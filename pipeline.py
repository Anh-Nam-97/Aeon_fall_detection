"""Core AI engine of Aeon_fall_detection.

The engine chains three stages for every video frame:

1. **Open-vocabulary detection** with YOLO-World v2 (a YOLOv8 detector whose
   classification head is driven by CLIP text embeddings). A plain COCO
   YOLOv8 has no ``floor`` class, so the zero-shot requirement ("detect
   'person' and 'floor' without retraining") can only be met by the
   open-vocabulary YOLOv8 variant.
2. **Relation prediction** with RelateAnything (``relsgg``), which scores
   every ``(subject, predicate, object)`` combination for the detected boxes
   against a predicate vocabulary chosen at run time (e.g. ``lying on``,
   ``standing on``). Two interchangeable backends are provided:
   PyTorch on CUDA (production GPU) and ONNX Runtime / OpenVINO on CPU
   (Intel Core i7-1260P edge box).
3. **Spatial reasoning + temporal filtering**: relation evidence is fused
   with box geometry (aspect ratio, overlap with the floor, height drop over
   time) and smoothed per tracked person, which turns noisy per-frame scores
   into a stable ``NORMAL / SUSPECTED / FALL`` status.

Typical use::

    engine = FallDetectionEngine(FallDetectionConfig(),
                                 mode=InferenceMode.ONNX_CPU)
    result = engine.process_frame(frame_bgr)
    print(result.status, result.alerts)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (Any, Deque, Dict, List, Optional, Sequence, Tuple,
                    Union)

import cv2
import numpy as np

LOGGER = logging.getLogger("aeon_fall_detection.pipeline")

#: Repository root of this project; every relative path is resolved from it
#: so the engine behaves identically when launched from another directory
#: (Gradio, Docker entrypoint, pytest).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

#: Default location of the cloned RelateAnything repository (setup.sh clones
#: it here). It is added to ``sys.path`` only when ``relsgg`` is not already
#: installed as a package, so an editable install always wins.
DEFAULT_RELATE_ANYTHING_DIR: Path = PROJECT_ROOT / "RelateAnything"

#: Default directory produced by ``export_onnx.py``.
DEFAULT_ARTIFACTS_DIR: Path = PROJECT_ROOT / "models" / "relsgg-vits16plus"

#: Predicates whose presence between a person and the floor indicates that
#: the person is on the ground rather than upright. They are deliberately
#: phrased as the model was trained ("<verb> on"), because RelateAnything's
#: text student embeds whole phrases and short canonical phrases transfer
#: best.
DEFAULT_FALL_PREDICATES: Tuple[str, ...] = (
    "lying on", "laying on", "resting on", "sitting on",
)

#: Predicates that indicate an upright, safe posture. They act as the
#: contrastive class: relation scores are normalised against them, which
#: cancels the global score offset that differs between scenes.
DEFAULT_UPRIGHT_PREDICATES: Tuple[str, ...] = (
    "standing on", "walking on", "walking past", "standing beside",
)

#: Detector vocabulary. ``floor`` is the support surface of interest; more
#: surfaces (``bed``, ``sofa``) can be added through the UI.
DEFAULT_OBJECT_CLASSES: Tuple[str, ...] = ("person", "floor")

#: Surfaces on which lying down is normal behaviour. If a person is "lying
#: on" one of them more strongly than on the floor, the alarm is suppressed;
#: without this rule every nap on a bed would trigger an alert.
DEFAULT_SAFE_SURFACES: Tuple[str, ...] = ("bed", "sofa", "couch", "mattress")

#: Labels treated as the floor. Synonyms are accepted so that a custom
#: vocabulary such as "ground" keeps working without code changes.
FLOOR_LABELS: Tuple[str, ...] = ("floor", "ground", "carpet", "rug")

VIRTUAL_FLOOR_LABEL: str = "floor (virtual)"


# ---------------------------------------------------------------------------
# Enumerations and configuration
# ---------------------------------------------------------------------------


class InferenceMode(str, Enum):
    """Execution backend of the relation model.

    Attributes:
        ONNX_CPU: ONNX graph executed on the CPU, through OpenVINO when an IR
            is available (fastest on Intel Alder Lake) or ONNX Runtime
            otherwise. Torch is not required at run time.
        TORCH_GPU: Native PyTorch model on CUDA. Supports any free-text
            predicate because the text encoder runs in-process.
    """

    ONNX_CPU = "onnx_cpu"
    TORCH_GPU = "torch_gpu"


class FallStatus(str, Enum):
    """Fall status of a person or of a whole frame.

    The enum value order is the severity order used when aggregating the
    status of several people into one frame-level status.

    Attributes:
        NORMAL: No evidence of a fall.
        SUSPECTED: The current frame looks like a fall but the evidence has
            not persisted long enough to raise an alarm.
        FALL: Fall confirmed by the temporal filter; an alert is raised.
    """

    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    FALL = "FALL"

    @property
    def severity(self) -> int:
        """Return an integer rank used to pick the worst status.

        Returns:
            int: 0 for NORMAL, 1 for SUSPECTED, 2 for FALL.
        """
        return {"NORMAL": 0, "SUSPECTED": 1, "FALL": 2}[self.value]


@dataclass
class VocabularySpec:
    """Run-time vocabularies of the detector and of the relation model.

    Attributes:
        objects: Classes requested from the open-vocabulary detector. Must
            contain ``person``.
        fall_predicates: Relation phrases indicating a person on the ground.
        upright_predicates: Relation phrases indicating an upright person.
    """

    objects: List[str] = field(
        default_factory=lambda: list(DEFAULT_OBJECT_CLASSES))
    fall_predicates: List[str] = field(
        default_factory=lambda: list(DEFAULT_FALL_PREDICATES))
    upright_predicates: List[str] = field(
        default_factory=lambda: list(DEFAULT_UPRIGHT_PREDICATES))

    def relation_vocabulary(self) -> List[str]:
        """Return the de-duplicated predicate list fed to RelateAnything.

        Returns:
            List[str]: Fall predicates followed by upright predicates, with
            duplicates removed while preserving order (order matters because
            column indices are derived from it).
        """
        merged: List[str] = []
        for phrase in [*self.fall_predicates, *self.upright_predicates]:
            if phrase not in merged:
                merged.append(phrase)
        return merged


@dataclass
class FallDetectionConfig:
    """Tunable parameters of :class:`FallDetectionEngine`.

    Attributes:
        detector_weights: YOLO-World checkpoint name or path. Ultralytics
            downloads official names automatically.
        detector_conf: Minimum detector confidence for a box to be kept.
        detector_imgsz: Detector input resolution (pixels, square).
        detector_openvino: In ``ONNX_CPU`` mode, export the detector to
            OpenVINO (cached per vocabulary) for a ~2x CPU speed-up.
        use_tracking: Track people with ByteTrack so temporal filtering is
            done per person instead of per frame.
        relation_model_id: Hugging Face id of the RelateAnything checkpoint
            used by the Torch backend.
        artifacts_dir: Directory produced by ``export_onnx.py`` (ONNX graph,
            sidecar JSON, predicate bank, optional OpenVINO IR).
        relate_anything_dir: Path of the cloned RelateAnything repository.
        prefer_openvino: In ``ONNX_CPU`` mode, run the relation graph with
            OpenVINO when an IR exists in ``artifacts_dir``.
        cpu_threads: Intra-op threads for ONNX Runtime / OpenVINO. ``0``
            lets the runtime decide (it uses all physical cores).
        max_boxes: Maximum number of boxes sent to the relation model.
        vocabulary: Detector and relation vocabularies.
        fall_threshold: Fused fall score above which a frame is "fall-like".
            0.5 separated upright (<= 0.44) from lying (>= 0.52) people in
            the validation images used during development.
        relation_weight: Weight of relation evidence in the fused score; the
            remainder goes to geometric evidence.
        relation_display_threshold: Minimum relation score for a triplet to
            be drawn on the frame. Calibrated RelateAnything probabilities
            are low in absolute terms (top relations typically 0.1-0.3,
            because a real frame has few true pairs), hence the low value.
        use_virtual_floor: Synthesise a floor region at the bottom of the
            frame when the detector finds none (YOLO-World recall on large
            amorphous "stuff" classes such as floor is low).
        virtual_floor_ratio: Height of the virtual floor as a fraction of
            the frame height.
        safe_surfaces: Surfaces on which lying down is not a fall.
        smoothing_window: Number of recent frames averaged per person.
        min_fall_frames: Number of fall-like frames inside the window
            required to confirm a fall.
        alert_cooldown_s: Minimum time between two alerts for the same
            person, to avoid flooding the alert log.
        track_ttl_s: Time after which an unseen track is forgotten.
        torch_device: CUDA device string for ``TORCH_GPU`` mode.
    """

    detector_weights: str = "yolov8s-worldv2.pt"
    detector_conf: float = 0.25
    detector_imgsz: int = 640
    detector_openvino: bool = True
    use_tracking: bool = True
    relation_model_id: str = "maelic/relsgg-vits16plus"
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    relate_anything_dir: Path = DEFAULT_RELATE_ANYTHING_DIR
    prefer_openvino: bool = True
    cpu_threads: int = 0
    max_boxes: int = 16
    vocabulary: VocabularySpec = field(default_factory=VocabularySpec)
    fall_threshold: float = 0.5
    relation_weight: float = 0.6
    relation_display_threshold: float = 0.1
    use_virtual_floor: bool = True
    virtual_floor_ratio: float = 0.25
    safe_surfaces: Tuple[str, ...] = DEFAULT_SAFE_SURFACES
    smoothing_window: int = 8
    min_fall_frames: int = 4
    alert_cooldown_s: float = 10.0
    track_ttl_s: float = 5.0
    torch_device: str = "cuda:0"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Detection:
    """One detected (or synthesised) region.

    Attributes:
        box: ``[x1, y1, x2, y2]`` in original-frame pixels.
        label: Class name.
        confidence: Detector confidence (1.0 for the virtual floor).
        track_id: ByteTrack identity, or ``None`` when tracking is off.
    """

    box: np.ndarray
    label: str
    confidence: float
    track_id: Optional[int] = None


@dataclass
class RelationTriplet:
    """A scored ``subject --predicate--> object`` relation.

    Attributes:
        subject_index: Index of the subject in the detection list.
        object_index: Index of the object in the detection list.
        predicate: Predicate phrase.
        score: Calibrated relation probability in ``[0, 1]``.
    """

    subject_index: int
    object_index: int
    predicate: str
    score: float


@dataclass
class PersonAssessment:
    """Per-person evidence and decision for one frame.

    Attributes:
        detection_index: Index of the person in the detection list.
        track_key: Key of the temporal state (track id or slot index).
        relation_score: Normalised relation evidence of being on the ground.
        geometry_score: Geometric evidence (posture, floor contact, drop).
        fused_score: Weighted fusion of both evidences.
        smoothed_score: Temporal mean of ``fused_score``.
        status: Decision after temporal filtering.
        safe_surface: Name of a safe surface the person lies on, if any.
        best_predicate: Strongest person-floor predicate.
    """

    detection_index: int
    track_key: str
    relation_score: Optional[float]
    geometry_score: float
    fused_score: float
    smoothed_score: float
    status: FallStatus
    safe_surface: Optional[str] = None
    best_predicate: Optional[str] = None


@dataclass
class FrameResult:
    """Output of :meth:`FallDetectionEngine.process_frame`.

    Attributes:
        annotated_frame: BGR frame with boxes, relations and status banner.
        status: Worst status over all people in the frame.
        detections: Detections used for reasoning (incl. virtual floor).
        persons: Per-person assessments.
        triplets: Relation triplets above the display threshold.
        alerts: Alert messages raised by this frame (usually empty).
        warnings: Non-fatal problems (missing predicates, fallbacks).
        timings_ms: Latency per stage in milliseconds.
    """

    annotated_frame: np.ndarray
    status: FallStatus
    detections: List[Detection]
    persons: List[PersonAssessment]
    triplets: List[RelationTriplet]
    alerts: List[str]
    warnings: List[str]
    timings_ms: Dict[str, float]


@dataclass
class RelationOutput:
    """Raw relation scores in a backend-independent layout.

    Attributes:
        scores: ``[K, V]`` calibrated probabilities for K candidate pairs and
            V predicates.
        sub_idx: ``[K]`` subject index of each pair.
        obj_idx: ``[K]`` object index of each pair.
        valid: ``[K]`` boolean mask of real (non-padding) pairs.
        predicates: Predicate names, aligned with the V axis.
    """

    scores: np.ndarray
    sub_idx: np.ndarray
    obj_idx: np.ndarray
    valid: np.ndarray
    predicates: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_vocabulary_spec(text: str,
                          base: Optional[VocabularySpec] = None
                          ) -> VocabularySpec:
    """Parse the free-text vocabulary typed in the UI.

    Accepted format, one section per line (sections are optional; a missing
    section keeps the value from ``base``)::

        objects: person, floor, bed
        fall: lying on, resting on
        upright: standing on, walking on

    A single line without a ``key:`` prefix is interpreted as the object list,
    which is the most common edit.

    Args:
        text: Raw text from the UI textbox.
        base: Vocabulary used for missing sections. Defaults to the built-in
            vocabulary.

    Returns:
        VocabularySpec: The parsed vocabulary. ``person`` is always present
        in ``objects`` because the engine cannot work without it.

    Raises:
        ValueError: If a section is present but empty, or the key is unknown.
    """
    base = base or VocabularySpec()
    spec = VocabularySpec(objects=list(base.objects),
                          fall_predicates=list(base.fall_predicates),
                          upright_predicates=list(base.upright_predicates))
    aliases = {
        "objects": "objects", "object": "objects", "classes": "objects",
        "fall": "fall_predicates", "falls": "fall_predicates",
        "upright": "upright_predicates", "normal": "upright_predicates",
    }
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for line in lines:
        if ":" in line:
            key, _, values = line.partition(":")
            attr = aliases.get(key.strip().lower())
            if attr is None:
                raise ValueError(
                    f"Unknown vocabulary section '{key.strip()}'. "
                    f"Use one of: objects, fall, upright.")
        else:
            attr, values = "objects", line
        # Lower-casing keeps detector prompts and predicate-bank lookups
        # consistent: the bank is keyed by lower-case phrases.
        items = [v.strip().lower() for v in values.split(",") if v.strip()]
        if not items:
            raise ValueError(f"Vocabulary section '{attr}' is empty.")
        setattr(spec, attr, list(dict.fromkeys(items)))
    if "person" not in spec.objects:
        # The pipeline reasons about people; silently dropping them would
        # make the detector "work" while the alarm can never fire.
        spec.objects.insert(0, "person")
    return spec


def box_area(box: np.ndarray) -> float:
    """Compute the area of an ``xyxy`` box.

    Args:
        box: ``[x1, y1, x2, y2]``.

    Returns:
        float: Area in squared pixels (0 for degenerate boxes).
    """
    return float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))


def intersection_over_first(first: np.ndarray, second: np.ndarray) -> float:
    """Fraction of ``first`` covered by ``second`` (IoA, not IoU).

    IoA is used instead of IoU because a person box is tiny compared with a
    floor box; IoU would stay near zero even for a person lying entirely on
    the floor, whereas IoA measures exactly "how much of the person is on the
    floor region".

    Args:
        first: ``xyxy`` box whose coverage is measured (the person).
        second: ``xyxy`` covering box (the floor).

    Returns:
        float: Value in ``[0, 1]``.
    """
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area = box_area(first)
    return float(inter / area) if area > 0 else 0.0


def clip01(value: float) -> float:
    """Clamp a number to ``[0, 1]``.

    Args:
        value: Any float.

    Returns:
        float: ``value`` clamped to the unit interval.
    """
    return float(min(max(value, 0.0), 1.0))


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    The relation graph masks invalid predicate columns with ``-inf``; the
    naive ``1 / (1 + exp(-z))`` overflows for large negative inputs, so the
    two branches below are evaluated separately.

    Args:
        z: Logits of any shape.

    Returns:
        np.ndarray: Probabilities with the same shape, ``float32``.
    """
    z = np.asarray(z, dtype=np.float32)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def boxes_to_normalized_cxcywh(boxes_xyxy: np.ndarray, width: int,
                               height: int) -> np.ndarray:
    """Convert pixel ``xyxy`` boxes to the relation model's box format.

    RelateAnything consumes boxes as normalised ``(cx, cy, w, h)`` in
    ``[0, 1]``. The image is resized to a square without letterboxing, and
    normalising boxes by the *original* width/height distorts them by exactly
    the same factor, so geometry stays consistent with the pixels.

    Args:
        boxes_xyxy: ``[N, 4]`` pixel boxes.
        width: Original frame width.
        height: Original frame height.

    Returns:
        np.ndarray: ``[N, 4]`` float32 normalised ``cxcywh`` boxes.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] /= max(width, 1)
    b[:, [1, 3]] /= max(height, 1)
    b = np.clip(b, 0.0, 1.0)
    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    return np.stack([cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]],
                    axis=-1).astype(np.float32)


def preprocess_image(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Turn a BGR frame into the relation model's image tensor.

    Mirrors ``relsgg.api.RelateAnything._to_chw`` and the upstream ONNX
    runtime exactly (plain square resize, RGB, ``[0, 1]`` range, NCHW).
    ImageNet normalisation happens *inside* the exported graph, so it must
    not be applied here or the inputs would be normalised twice.

    Args:
        frame_bgr: ``HxWx3`` uint8 BGR frame.
        size: Square side length expected by the model.

    Returns:
        np.ndarray: ``[1, 3, size, size]`` float32 contiguous array.
    """
    resized = cv2.resize(frame_bgr, (size, size),
                         interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(rgb.astype(np.float32) / 255.0)


def ensure_relate_anything_importable(repo_dir: Path) -> None:
    """Make ``import relsgg`` work from a plain clone.

    ``setup.sh`` installs the clone in editable mode, so this is a safety net
    for environments where only ``git clone`` was run (e.g. a quick test on
    a laptop).

    Args:
        repo_dir: Path of the cloned RelateAnything repository.

    Raises:
        ImportError: If ``relsgg`` is neither installed nor present in
            ``repo_dir``.
    """
    try:
        import relsgg  # noqa: F401  (import probes availability only)
        return
    except ImportError:
        pass
    if (repo_dir / "relsgg" / "__init__.py").is_file():
        sys.path.insert(0, str(repo_dir))
        import relsgg  # noqa: F401
        return
    raise ImportError(
        "Package 'relsgg' (RelateAnything) is not installed and was not "
        f"found in {repo_dir}. Run setup.sh first.")


# ---------------------------------------------------------------------------
# Stage 1: open-vocabulary detector
# ---------------------------------------------------------------------------


class OpenVocabularyDetector:
    """YOLO-World v2 (YOLOv8 family) detector with a run-time vocabulary.

    In CPU mode the detector is optionally exported to OpenVINO once per
    vocabulary. The text embeddings of the vocabulary are baked into the
    exported graph, which removes the CLIP text encoder from the per-frame
    path and lets OpenVINO fuse the whole network for Alder Lake cores.
    """

    def __init__(self, config: FallDetectionConfig, device: str,
                 use_openvino: bool) -> None:
        """Load the detector and apply the configured vocabulary.

        Args:
            config: Engine configuration.
            device: Ultralytics device string (``"cpu"`` or ``"cuda:0"``).
            use_openvino: Export/load an OpenVINO copy for CPU inference.

        Raises:
            RuntimeError: If the detector weights cannot be loaded.
        """
        self.config = config
        self.device = device
        self.use_openvino = use_openvino
        self.half = device.startswith("cuda")
        self.classes: List[str] = []
        self.backend_name = "ultralytics-torch"
        self._model: Any = None
        self.set_classes(config.vocabulary.objects)

    def _load_world_model(self, classes: Sequence[str]) -> Any:
        """Load YOLO-World and set its text-prompted classes.

        Args:
            classes: Class names.

        Returns:
            Any: An ``ultralytics.YOLOWorld`` instance.

        Raises:
            RuntimeError: If ultralytics or the weights are unavailable.
        """
        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(self.config.detector_weights)
            model.set_classes(list(classes))
            return model
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                f"Cannot load detector '{self.config.detector_weights}': "
                f"{exc}") from exc

    def _openvino_cache_dir(self, classes: Sequence[str]) -> Path:
        """Return the cache directory of an OpenVINO export.

        The key hashes the weights name, class list and resolution, because
        each of them changes the exported graph.

        Args:
            classes: Class names baked into the export.

        Returns:
            Path: Directory that holds (or will hold) the IR files.
        """
        key = json.dumps([self.config.detector_weights, list(classes),
                          self.config.detector_imgsz])
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        # The "_openvino_model" suffix is mandatory: Ultralytics infers the
        # backend of a model directory from that suffix.
        return (self.config.artifacts_dir.parent / "detector_openvino"
                / f"{digest}_openvino_model")

    def _load_openvino_model(self, classes: Sequence[str]) -> Any:
        """Load (exporting first if needed) an OpenVINO detector.

        Args:
            classes: Class names baked into the export.

        Returns:
            Any: An ``ultralytics.YOLO`` instance backed by OpenVINO.

        Raises:
            RuntimeError: If the export or the load fails.
        """
        from ultralytics import YOLO
        cache_dir = self._openvino_cache_dir(classes)
        if not (cache_dir / "metadata.yaml").is_file():
            LOGGER.info("Exporting detector to OpenVINO (one-off, cached in "
                        "%s)", cache_dir)
            world = self._load_world_model(classes)
            exported = world.export(format="openvino",
                                    imgsz=self.config.detector_imgsz,
                                    half=False, dynamic=False,
                                    verbose=False)
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            # Ultralytics writes next to the .pt file; moving the folder into
            # our cache keeps exports for different vocabularies apart.
            shutil.move(str(exported), str(cache_dir))
        return YOLO(str(cache_dir), task="detect")

    def set_classes(self, classes: Sequence[str]) -> None:
        """Change the detector vocabulary.

        Args:
            classes: New class names.

        Raises:
            RuntimeError: If neither backend can be loaded.
        """
        classes = list(dict.fromkeys(c.strip() for c in classes if c))
        if classes == self.classes and self._model is not None:
            return
        model: Any = None
        if self.use_openvino:
            try:
                model = self._load_openvino_model(classes)
                self.backend_name = "ultralytics-openvino"
            except Exception as exc:  # noqa: BLE001 - graceful fallback
                # OpenVINO export can fail on exotic platforms; the torch
                # detector still works on CPU, only slower.
                LOGGER.warning("OpenVINO detector unavailable (%s); "
                               "falling back to PyTorch on CPU.", exc)
        if model is None:
            model = self._load_world_model(classes)
            self.backend_name = "ultralytics-torch"
        self._model = model
        self.classes = classes

    def reset_tracker(self) -> None:
        """Forget all tracks, e.g. when a new video starts.

        Ultralytics keeps tracker state inside its predictor; without a reset
        the IDs of a previous video would leak into the next one.
        """
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            try:
                tracker.reset()
            except Exception as exc:  # noqa: BLE001 - best effort
                LOGGER.debug("Tracker reset failed: %s", exc)

    def __call__(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame.

        Returns:
            List[Detection]: Detections sorted by decreasing confidence.

        Raises:
            RuntimeError: If inference fails.
        """
        kwargs: Dict[str, Any] = dict(
            conf=self.config.detector_conf, imgsz=self.config.detector_imgsz,
            device=self.device, verbose=False)
        if self.half and self.backend_name == "ultralytics-torch":
            # FP16 halves memory traffic on Ada/Blackwell tensor cores with
            # no measurable accuracy loss for detection.
            kwargs["half"] = True
        try:
            if self.config.use_tracking:
                results = self._model.track(frame_bgr, persist=True,
                                            tracker="bytetrack.yaml",
                                            **kwargs)
            else:
                results = self._model.predict(frame_bgr, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(f"Detector inference failed: {exc}") from exc

        detections: List[Detection] = []
        if not results:
            return detections
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections
        names: Dict[int, str] = res.names
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32)
        cls = res.boxes.cls.cpu().numpy().astype(np.int64)
        ids = (res.boxes.id.cpu().numpy().astype(np.int64)
               if getattr(res.boxes, "id", None) is not None else None)
        for i in np.argsort(-conf):
            detections.append(Detection(
                box=xyxy[i], label=str(names.get(int(cls[i]), cls[i])),
                confidence=float(conf[i]),
                track_id=int(ids[i]) if ids is not None else None))
        return detections


# ---------------------------------------------------------------------------
# Stage 2: relation backends
# ---------------------------------------------------------------------------


class RelationBackend:
    """Interface of a RelateAnything backend."""

    name: str = "abstract"

    def set_predicates(self, predicates: Sequence[str]) -> List[str]:
        """Select the predicate vocabulary.

        Args:
            predicates: Requested predicate phrases.

        Returns:
            List[str]: Phrases that could NOT be loaded (empty when all
            were accepted).
        """
        raise NotImplementedError

    @property
    def predicates(self) -> List[str]:
        """Return the active predicate vocabulary.

        Returns:
            List[str]: Active predicate phrases.
        """
        raise NotImplementedError

    def infer(self, frame_bgr: np.ndarray,
              boxes_xyxy: np.ndarray) -> RelationOutput:
        """Score all relations between the given boxes.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame.
            boxes_xyxy: ``[N, 4]`` pixel boxes, ``N >= 2``.

        Returns:
            RelationOutput: Scores for every candidate pair.
        """
        raise NotImplementedError


class TorchRelationBackend(RelationBackend):
    """RelateAnything in PyTorch on a CUDA GPU.

    Any free-text predicate is accepted: the text student encodes new phrases
    once (milliseconds) and inference afterwards is vision-only.
    """

    name = "torch-cuda"

    def __init__(self, config: FallDetectionConfig) -> None:
        """Download/load the checkpoint and move it to the GPU.

        Args:
            config: Engine configuration.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        ensure_relate_anything_importable(config.relate_anything_dir)
        import torch
        from relsgg import RelateAnything

        self._torch = torch
        device = config.torch_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            # Keeping the service alive on CPU is preferable to crashing a
            # container whose GPU passthrough is misconfigured; the warning
            # makes the degraded state visible in logs and in the UI.
            LOGGER.warning("CUDA is not available; TORCH_GPU runs on CPU.")
            device = "cpu"
        if device.startswith("cuda"):
            # TF32 matmuls are ~2x faster on Ada/Blackwell tensor cores and
            # keep 10 mantissa bits, far more than the relation scores need.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(device)
        # Report the real device in the UI banner/logs (e.g. "torch-cpu"
        # after a CUDA fallback) instead of the intended one.
        self.name = f"torch-{self.device.type}"
        try:
            self._ra = RelateAnything.from_pretrained(
                config.relation_model_id,
                predicates=config.vocabulary.relation_vocabulary(),
                device=str(self.device))
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                f"Cannot load RelateAnything '{config.relation_model_id}': "
                f"{exc}") from exc
        self.img_size: int = int(self._ra.img_size)

    @property
    def predicates(self) -> List[str]:
        """Return the active predicate vocabulary.

        Returns:
            List[str]: Active predicate phrases.
        """
        return list(self._ra.predicates)

    def set_predicates(self, predicates: Sequence[str]) -> List[str]:
        """Encode and install a new predicate vocabulary.

        Args:
            predicates: Requested predicate phrases (any text).

        Returns:
            List[str]: Always empty; the text encoder accepts any phrase.

        Raises:
            RuntimeError: If encoding fails.
        """
        predicates = list(predicates)
        if predicates == self._ra.predicates:
            return []
        try:
            self._ra.set_vocabulary(predicates)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(f"Vocabulary encoding failed: {exc}") from exc
        return []

    def infer(self, frame_bgr: np.ndarray,
              boxes_xyxy: np.ndarray) -> RelationOutput:
        """Run the PyTorch relation model.

        The public ``RelateAnything.predict`` keeps only the best predicate
        per pair; fall reasoning needs the full ``[K, V]`` score matrix (to
        compare fall vs. upright predicates), so the underlying module is
        called directly with the same preprocessing.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame.
            boxes_xyxy: ``[N, 4]`` pixel boxes.

        Returns:
            RelationOutput: Scores for every candidate pair.

        Raises:
            RuntimeError: If the forward pass fails.
        """
        torch = self._torch
        height, width = frame_bgr.shape[:2]
        image = torch.from_numpy(preprocess_image(frame_bgr, self.img_size))
        boxes = torch.from_numpy(
            boxes_to_normalized_cxcywh(boxes_xyxy, width, height))[None]
        count = torch.tensor([len(boxes_xyxy)], dtype=torch.long)
        try:
            with torch.inference_mode():
                out = self._ra.model(image.to(self.device),
                                     boxes.to(self.device),
                                     box_counts=count.to(self.device),
                                     targets=None)
                logits = out["logits"][0].float()
                pair = out["pair_logits"][0].float()
                # The calibrated score contract shipped with the checkpoint:
                # sigmoid(a * (pred + pair) + b).
                scores = self._ra.contract.scores(logits, pair)
                result = RelationOutput(
                    scores=scores.cpu().numpy().astype(np.float32),
                    sub_idx=out["sub_idx"][0].cpu().numpy().astype(np.int64),
                    obj_idx=out["obj_idx"][0].cpu().numpy().astype(np.int64),
                    valid=out["valid_mask"][0].cpu().numpy().astype(bool),
                    predicates=self.predicates)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(f"Relation inference failed: {exc}") from exc
        return result


class OnnxRelationBackend(RelationBackend):
    """RelateAnything exported by ``export_onnx.py``, executed on the CPU.

    The graph takes the predicate embedding matrix ``W`` and routing weights
    ``alpha`` as *inputs*, so switching vocabulary is a row lookup in the
    pre-computed predicate bank: no text encoder and no torch at run time.
    The trade-off is that only phrases present in the bank can be used.
    """

    def __init__(self, config: FallDetectionConfig) -> None:
        """Load the sidecar, the predicate bank and the inference engine.

        Args:
            config: Engine configuration.

        Raises:
            FileNotFoundError: If the export artifacts are missing.
            RuntimeError: If no runtime can load the graph.
        """
        art = Path(config.artifacts_dir)
        onnx_path = art / "relateanything.onnx"
        meta_path = art / "relateanything.json"
        bank_path = art / "predicate_bank.npz"
        for path in (onnx_path, meta_path, bank_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Run: python export_onnx.py "
                    f"--out-dir {art}")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self.img_size: int = int(meta["img_size"])
        self.max_boxes: int = int(meta["max_boxes"])
        calib = meta.get("calibration") or {}
        self.calib_a: float = float(calib.get("a", 1.0))
        self.calib_b: float = float(calib.get("b", 0.0))

        with np.load(bank_path, allow_pickle=False) as bank:
            self._bank_names: List[str] = [str(n) for n in bank["names"]]
            self._bank_w: np.ndarray = bank["W"].astype(np.float32)
            self._bank_alpha: np.ndarray = bank["alpha"].astype(np.float32)
        self._bank_index = {n: i for i, n in enumerate(self._bank_names)}
        self._predicates: List[str] = []
        self._w = np.zeros((0, self._bank_w.shape[1]), np.float32)
        self._alpha = np.zeros((0,), np.float32)

        self._ov_request: Any = None
        self._ort_session: Any = None
        ir_path = art / "relateanything_fp16.xml"
        if config.prefer_openvino and ir_path.is_file():
            try:
                self._init_openvino(ir_path, config.cpu_threads)
            except Exception as exc:  # noqa: BLE001 - graceful fallback
                LOGGER.warning("OpenVINO load failed (%s); using ONNX "
                               "Runtime.", exc)
        if self._ov_request is None:
            self._init_onnxruntime(onnx_path, config.cpu_threads)

    def _init_openvino(self, ir_path: Path, threads: int) -> None:
        """Compile the OpenVINO IR for the CPU.

        ``PERFORMANCE_HINT=LATENCY`` makes OpenVINO schedule one request on
        the P-cores of a hybrid Alder Lake CPU instead of spreading streams
        over E-cores, which is what a single live camera needs.

        Args:
            ir_path: Path of the ``.xml`` IR.
            threads: Inference threads (0 = automatic).
        """
        import openvino as ov
        core = ov.Core()
        cfg: Dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = int(threads)
        compiled = core.compile_model(core.read_model(str(ir_path)), "CPU",
                                      cfg)
        self._ov_compiled = compiled
        self._ov_request = compiled.create_infer_request()
        self.name = "openvino-cpu"

    def _init_onnxruntime(self, onnx_path: Path, threads: int) -> None:
        """Create an ONNX Runtime CPU session.

        Args:
            onnx_path: Path of the ``.onnx`` graph.
            threads: Intra-op threads (0 = automatic).

        Raises:
            RuntimeError: If the session cannot be created.
        """
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            if threads:
                opts.intra_op_num_threads = int(threads)
            self._ort_session = ort.InferenceSession(
                str(onnx_path), opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(f"Cannot load {onnx_path}: {exc}") from exc
        self.name = "onnxruntime-cpu"

    @property
    def predicates(self) -> List[str]:
        """Return the active predicate vocabulary.

        Returns:
            List[str]: Active predicate phrases.
        """
        return list(self._predicates)

    def available_predicates(self) -> List[str]:
        """Return every phrase stored in the predicate bank.

        Returns:
            List[str]: Phrases usable in ONNX mode.
        """
        return list(self._bank_names)

    def set_predicates(self, predicates: Sequence[str]) -> List[str]:
        """Slice the requested phrases out of the predicate bank.

        Args:
            predicates: Requested predicate phrases.

        Returns:
            List[str]: Phrases missing from the bank (ignored). Re-run
            ``export_onnx.py --predicates ...`` to add them.

        Raises:
            ValueError: If none of the requested phrases is in the bank.
        """
        missing = [p for p in predicates if p not in self._bank_index]
        kept = [p for p in predicates if p in self._bank_index]
        if not kept:
            raise ValueError(
                "None of the requested predicates is in the predicate bank. "
                f"Available examples: {self._bank_names[:10]}")
        idx = np.array([self._bank_index[p] for p in kept], dtype=np.int64)
        self._predicates = kept
        # Contiguous copies: both runtimes reject strided views.
        self._w = np.ascontiguousarray(self._bank_w[idx])
        self._alpha = np.ascontiguousarray(self._bank_alpha[idx])
        return missing

    def infer(self, frame_bgr: np.ndarray,
              boxes_xyxy: np.ndarray) -> RelationOutput:
        """Run the exported graph.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame.
            boxes_xyxy: ``[N, 4]`` pixel boxes.

        Returns:
            RelationOutput: Scores for every candidate pair.

        Raises:
            RuntimeError: If the forward pass fails.
        """
        height, width = frame_bgr.shape[:2]
        boxes = boxes_to_normalized_cxcywh(boxes_xyxy, width, height)
        count = min(len(boxes), self.max_boxes)
        # The graph is traced for a fixed number of box slots; real boxes
        # go first, padding is zero, and ``box_counts`` tells the model how
        # many slots are real (exactly what the training collate does).
        padded = np.zeros((1, self.max_boxes, 4), dtype=np.float32)
        padded[0, :count] = boxes[:count]
        feed = {
            "image": preprocess_image(frame_bgr, self.img_size),
            "boxes": padded,
            "box_counts": np.array([count], dtype=np.int64),
            "W": self._w,
            "alpha": self._alpha,
        }
        names = ["pred_logits", "pair_logits", "sub_idx", "obj_idx",
                 "valid_mask"]
        try:
            if self._ov_request is not None:
                res = self._ov_request.infer(feed)
                outs = [np.asarray(res[self._ov_compiled.output(n)])
                        for n in names]
            else:
                outs = self._ort_session.run(names, feed)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(f"Relation inference failed: {exc}") from exc
        pred, pair, sub, obj, valid = (o[0] for o in outs)
        z = self.calib_a * (pred.astype(np.float32)
                            + pair.astype(np.float32)[:, None]) + self.calib_b
        return RelationOutput(scores=stable_sigmoid(z),
                              sub_idx=sub.astype(np.int64),
                              obj_idx=obj.astype(np.int64),
                              valid=valid.astype(bool),
                              predicates=self.predicates)


# ---------------------------------------------------------------------------
# Stage 3: temporal state per person
# ---------------------------------------------------------------------------


@dataclass
class _TrackState:
    """Sliding-window memory of one person.

    Attributes:
        scores: Recent fused scores.
        heights: Recent box heights (pixels), used to detect a sudden drop.
        last_seen: Timestamp of the last observation.
        last_alert: Timestamp of the last alert, ``-inf`` if never.
        status: Last decided status.
    """

    scores: Deque[float]
    heights: Deque[float]
    last_seen: float
    last_alert: float = -math.inf
    status: FallStatus = FallStatus.NORMAL


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class FallDetectionEngine:
    """End-to-end zero-shot fall detection engine.

    Thread-safety: :meth:`process_frame` holds an internal lock because the
    tracker and the temporal state are mutable. One engine therefore serves
    one stream at a time; create one engine per camera for parallelism.
    """

    def __init__(self, config: Optional[FallDetectionConfig] = None,
                 mode: Union[InferenceMode, str] = InferenceMode.ONNX_CPU
                 ) -> None:
        """Build the detector and the relation backend for ``mode``.

        Args:
            config: Engine configuration (defaults are production values).
            mode: ``InferenceMode`` or its string value.

        Raises:
            RuntimeError: If a model cannot be loaded.
            FileNotFoundError: If ONNX artifacts are missing in ONNX mode.
        """
        self.config = config or FallDetectionConfig()
        self.mode = InferenceMode(mode)
        self._lock = threading.Lock()
        self._tracks: Dict[str, _TrackState] = {}
        self.vocab_warnings: List[str] = []

        if self.mode is InferenceMode.TORCH_GPU:
            self.relation: RelationBackend = TorchRelationBackend(
                self.config)
            det_device = (self.config.torch_device
                          if getattr(self.relation, "device", None) is not
                          None and self.relation.device.type == "cuda"
                          else "cpu")
            use_ov = False
        else:
            self.relation = OnnxRelationBackend(self.config)
            det_device, use_ov = "cpu", self.config.detector_openvino
        self.detector = OpenVocabularyDetector(self.config, det_device,
                                               use_ov)
        self._apply_relation_vocabulary(self.config.vocabulary)

    # -- vocabulary --------------------------------------------------------

    def _apply_relation_vocabulary(self, spec: VocabularySpec) -> None:
        """Install the relation vocabulary and record skipped phrases.

        Args:
            spec: Vocabulary to install.

        Raises:
            ValueError: If no usable fall predicate remains.
        """
        missing = self.relation.set_predicates(spec.relation_vocabulary())
        active = set(self.relation.predicates)
        if not any(p in active for p in spec.fall_predicates):
            raise ValueError(
                "No fall predicate is available in the current backend; "
                f"requested {spec.fall_predicates}.")
        self.vocab_warnings = (
            [f"Predicates not in ONNX bank (ignored): {missing}"]
            if missing else [])

    def update_vocabulary(self, spec: VocabularySpec) -> List[str]:
        """Switch detector classes and relation predicates at run time.

        Args:
            spec: New vocabulary.

        Returns:
            List[str]: Warnings (e.g. predicates missing from the bank).

        Raises:
            ValueError: If the vocabulary is unusable.
            RuntimeError: If the detector cannot be reloaded.
        """
        with self._lock:
            self.detector.set_classes(spec.objects)
            self._apply_relation_vocabulary(spec)
            self.config.vocabulary = spec
            return list(self.vocab_warnings)

    def reset(self) -> None:
        """Clear tracker and temporal state (call when the source changes)."""
        with self._lock:
            self._tracks.clear()
            self.detector.reset_tracker()

    # -- reasoning helpers -------------------------------------------------

    def _select_boxes(self, detections: List[Detection],
                      frame_shape: Tuple[int, ...]
                      ) -> Tuple[List[Detection], bool]:
        """Choose the regions given to the relation model.

        People are prioritised, then floors and safe surfaces, then other
        objects, because the relation graph has a fixed number of box slots
        and every slot spent on irrelevant clutter is a lost person.

        Args:
            detections: Raw detections sorted by confidence.
            frame_shape: Shape of the frame ``(H, W, C)``.

        Returns:
            Tuple[List[Detection], bool]: Selected detections and whether a
            virtual floor was added.
        """
        height, width = frame_shape[:2]
        # Partition by label (not by object equality: Detection holds numpy
        # arrays, whose element-wise ``==`` cannot be used as a boolean).
        safe_labels = set(self.config.safe_surfaces)
        persons = [d for d in detections if d.label == "person"]
        floors = [d for d in detections if d.label in FLOOR_LABELS]
        safe = [d for d in detections if d.label in safe_labels]
        others = [d for d in detections
                  if d.label != "person" and d.label not in FLOOR_LABELS
                  and d.label not in safe_labels]
        virtual = False
        if not floors and self.config.use_virtual_floor:
            # The lowest band of a fixed camera view is almost always floor
            # in indoor retail/care settings; using it as a region keeps the
            # person->floor relation computable when the detector misses the
            # amorphous floor class.
            y1 = height * (1.0 - self.config.virtual_floor_ratio)
            floors = [Detection(
                box=np.array([0.0, y1, width - 1.0, height - 1.0],
                             dtype=np.float32),
                label=VIRTUAL_FLOOR_LABEL, confidence=1.0)]
            virtual = True
        # Keep at most two floor regions: more add slots but not evidence.
        ordered = persons + floors[:2] + safe + others
        return ordered[: self.config.max_boxes], virtual

    def _relation_evidence(self, rel: RelationOutput, person_idx: int,
                           floor_idx: List[int], safe_idx: List[int],
                           detections: List[Detection]
                           ) -> Tuple[Optional[float], Optional[str],
                                      Optional[str]]:
        """Compute normalised relation evidence for one person.

        ``evidence = fall / (fall + upright)`` compares the two predicate
        groups for the same pair. Raw scores share a scene-dependent offset
        (crowded scenes score lower overall); the ratio cancels it, so one
        threshold works across cameras.

        Args:
            rel: Relation scores.
            person_idx: Index of the person among the selected detections.
            floor_idx: Indices of floor regions.
            safe_idx: Indices of safe surfaces.
            detections: Selected detections (for labels).

        Returns:
            Tuple[Optional[float], Optional[str], Optional[str]]: Evidence
            in ``[0, 1]`` (``None`` if no person-floor pair was scored), the
            best floor predicate, and the label of a safe surface on which
            the person lies more strongly than on the floor.
        """
        preds = rel.predicates
        spec = self.config.vocabulary
        fall_cols = [i for i, p in enumerate(preds)
                     if p in spec.fall_predicates]
        up_cols = [i for i, p in enumerate(preds)
                   if p in spec.upright_predicates]
        if not fall_cols:
            return None, None, None
        mask = rel.valid & (rel.sub_idx == person_idx)

        def best(targets: List[int], cols: List[int]
                 ) -> Tuple[float, Optional[str]]:
            """Best score among ``cols`` for pairs to ``targets``."""
            sel = mask & np.isin(rel.obj_idx, targets)
            if not cols or not np.any(sel):
                return -1.0, None
            block = rel.scores[sel][:, cols]
            flat = int(np.argmax(block))
            return (float(block.flat[flat]),
                    preds[cols[flat % len(cols)]])

        fall_s, fall_p = best(floor_idx, fall_cols)
        if fall_s < 0:
            return None, None, None
        up_s, up_p = best(floor_idx, up_cols)
        evidence = (fall_s / (fall_s + up_s + 1e-6)) if up_s >= 0 else fall_s
        best_pred = fall_p if up_s < fall_s else up_p
        safe_label = None
        for j in safe_idx:
            s_safe, _ = best([j], fall_cols)
            if s_safe > fall_s:
                safe_label = detections[j].label
                break
        return clip01(evidence), best_pred, safe_label

    def _geometry_evidence(self, person: Detection,
                           floors: List[Detection],
                           state: Optional[_TrackState]) -> float:
        """Compute geometric fall evidence for one person.

        Three cues, each in ``[0, 1]``:

        * ``posture``: a lying body has a box wider than tall; the aspect
          ratio is mapped linearly from 0.6 (upright) to 1.4 (lying).
        * ``contact``: share of the person box inside a floor region (IoA).
        * ``drop``: relative height loss versus the tallest height in the
          track history, which captures the *transition* of a fall and not
          only the final pose (someone already sitting stays low).

        Args:
            person: Person detection.
            floors: Floor regions.
            state: Temporal state of the person, if tracked.

        Returns:
            float: Weighted geometric evidence in ``[0, 1]``.
        """
        x1, y1, x2, y2 = (float(v) for v in person.box)
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        posture = clip01((w / h - 0.6) / 0.8)
        contact = max((intersection_over_first(person.box, f.box)
                       for f in floors), default=0.0)
        drop = 0.0
        if state is not None and state.heights:
            drop = clip01(1.0 - h / max(max(state.heights), 1.0))
        return clip01(0.5 * posture + 0.25 * contact + 0.25 * drop)

    def _update_track(self, key: str, fused: float, height: float,
                      now: float) -> Tuple[_TrackState, float]:
        """Push a new observation and decide the temporal status.

        Args:
            key: Track key.
            fused: Fused score of this frame.
            height: Person box height (pixels).
            now: Timestamp in seconds.

        Returns:
            Tuple[_TrackState, float]: Updated state and smoothed score.
        """
        cfg = self.config
        state = self._tracks.get(key)
        if state is None:
            state = _TrackState(scores=deque(maxlen=cfg.smoothing_window),
                                heights=deque(maxlen=cfg.smoothing_window),
                                last_seen=now)
            self._tracks[key] = state
        state.scores.append(fused)
        state.heights.append(height)
        state.last_seen = now
        smoothed = float(np.mean(state.scores))
        hits = sum(1 for s in state.scores if s >= cfg.fall_threshold)
        # Hysteresis: confirming needs several fall-like frames, while a
        # confirmed fall is only released once the mean drops clearly below
        # the threshold. This prevents a flickering alarm when the score
        # hovers around the threshold.
        if hits >= cfg.min_fall_frames and smoothed >= cfg.fall_threshold:
            state.status = FallStatus.FALL
        elif (state.status is FallStatus.FALL
              and smoothed >= 0.8 * cfg.fall_threshold):
            state.status = FallStatus.FALL
        elif fused >= cfg.fall_threshold:
            state.status = FallStatus.SUSPECTED
        else:
            state.status = FallStatus.NORMAL
        return state, smoothed

    def _expire_tracks(self, now: float) -> None:
        """Drop tracks not seen for ``track_ttl_s`` seconds.

        Args:
            now: Current timestamp in seconds.
        """
        ttl = self.config.track_ttl_s
        for key in [k for k, s in self._tracks.items()
                    if now - s.last_seen > ttl]:
            del self._tracks[key]

    # -- drawing -----------------------------------------------------------

    @staticmethod
    def _draw(frame: np.ndarray, detections: List[Detection],
              persons: List[PersonAssessment],
              triplets: List[RelationTriplet], status: FallStatus,
              timings: Dict[str, float], backend: str) -> np.ndarray:
        """Render boxes, relations and a status banner.

        Args:
            frame: BGR frame (not modified).
            detections: Selected detections.
            persons: Person assessments.
            triplets: Relations to draw.
            status: Frame status.
            timings: Latency per stage.
            backend: Backend description for the banner.

        Returns:
            np.ndarray: Annotated BGR copy of ``frame``.
        """
        out = frame.copy()
        colors = {FallStatus.NORMAL: (60, 200, 60),
                  FallStatus.SUSPECTED: (0, 200, 255),
                  FallStatus.FALL: (0, 0, 255)}
        by_index = {p.detection_index: p for p in persons}
        overlay = out.copy()
        for det in detections:
            if det.label in FLOOR_LABELS or det.label == VIRTUAL_FLOOR_LABEL:
                x1, y1, x2, y2 = det.box.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 160, 0), -1)
        # Floors are drawn as a translucent fill so that they do not hide
        # the people standing on them.
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.box.astype(int)
            if i in by_index:
                pa = by_index[i]
                color = colors[pa.status]
                text = (f"person {pa.track_key} {pa.status.value} "
                        f"{pa.smoothed_score:.2f}")
                if pa.safe_surface:
                    text += f" (on {pa.safe_surface})"
                thickness = 3
            else:
                color, text, thickness = (255, 160, 0), det.label, 1
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(out, text, (x1 + 2, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        cv2.LINE_AA)
        for t in triplets[:6]:
            s, o = detections[t.subject_index], detections[t.object_index]
            p1 = (int((s.box[0] + s.box[2]) / 2), int(s.box[3]))
            p2 = (int((o.box[0] + o.box[2]) / 2),
                  int((o.box[1] + o.box[3]) / 2))
            cv2.arrowedLine(out, p1, p2, (255, 255, 255), 1, cv2.LINE_AA,
                            tipLength=0.03)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(out, f"{t.predicate} {t.score:.2f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        banner = (f"{status.value} | {backend} | total "
                  f"{timings.get('total', 0.0):.0f} ms")
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), colors[status], -1)
        cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    # -- main entry point --------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray,
                      timestamp: Optional[float] = None) -> FrameResult:
        """Run detection, relation prediction and fall reasoning.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame (OpenCV convention).
            timestamp: Frame time in seconds. Use the video clock for files
                so temporal parameters are independent of processing speed;
                defaults to wall-clock time for live sources.

        Returns:
            FrameResult: Annotated frame, status, evidence and alerts.

        Raises:
            ValueError: If ``frame_bgr`` is not an ``HxWx3`` uint8 array.
        """
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3):
            raise ValueError("frame_bgr must be an HxWx3 array.")
        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            return self._process_locked(frame_bgr, now)

    def _process_locked(self, frame: np.ndarray, now: float) -> FrameResult:
        """Body of :meth:`process_frame`, executed under the lock.

        Args:
            frame: Validated BGR frame.
            now: Timestamp in seconds.

        Returns:
            FrameResult: See :meth:`process_frame`.
        """
        timings: Dict[str, float] = {}
        warnings: List[str] = list(self.vocab_warnings)
        t0 = time.perf_counter()
        try:
            raw = self.detector(frame)
        except RuntimeError as exc:
            # A single failed frame must not kill a 24/7 stream: report it
            # and return the unannotated frame with a NORMAL status.
            LOGGER.error("%s", exc)
            raw, warnings = [], warnings + [str(exc)]
        timings["detect"] = (time.perf_counter() - t0) * 1e3

        selected, virtual = self._select_boxes(raw, frame.shape)
        if virtual:
            warnings.append("No floor detected: virtual floor in use.")
        person_idx = [i for i, d in enumerate(selected)
                      if d.label == "person"]
        floor_idx = [i for i, d in enumerate(selected)
                     if d.label in FLOOR_LABELS
                     or d.label == VIRTUAL_FLOOR_LABEL]
        safe_idx = [i for i, d in enumerate(selected)
                    if d.label in self.config.safe_surfaces]

        rel: Optional[RelationOutput] = None
        t1 = time.perf_counter()
        if person_idx and len(selected) >= 2:
            boxes = np.stack([d.box for d in selected]).astype(np.float32)
            try:
                rel = self.relation.infer(frame, boxes)
            except RuntimeError as exc:
                LOGGER.error("%s", exc)
                warnings.append(str(exc))
        timings["relation"] = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        persons: List[PersonAssessment] = []
        alerts: List[str] = []
        floors = [selected[j] for j in floor_idx]
        w_rel = clip01(self.config.relation_weight)
        for slot, i in enumerate(person_idx):
            det = selected[i]
            key = (str(det.track_id) if det.track_id is not None
                   else f"#{slot}")
            state = self._tracks.get(key)
            geom = self._geometry_evidence(det, floors, state)
            rel_s, best_pred, safe = (None, None, None)
            if rel is not None:
                rel_s, best_pred, safe = self._relation_evidence(
                    rel, i, floor_idx, safe_idx, selected)
            # Without relation evidence the decision falls back to geometry
            # alone instead of treating the missing term as zero (which would
            # make falls impossible to detect when a pair is pruned).
            fused = (geom if rel_s is None
                     else w_rel * rel_s + (1.0 - w_rel) * geom)
            if safe is not None:
                fused *= 0.3
            height = float(det.box[3] - det.box[1])
            state, smoothed = self._update_track(key, fused, height, now)
            if (state.status is FallStatus.FALL
                    and now - state.last_alert
                    >= self.config.alert_cooldown_s):
                state.last_alert = now
                # Wall-clock timestamps (live sources) are epoch seconds;
                # video files pass their own clock (seconds since start),
                # which must not be rendered as a time of day.
                stamp = (time.strftime("%H:%M:%S", time.localtime(now))
                         if now > 1e9 else f"t={now:.1f}s")
                alerts.append(
                    f"[{stamp}] FALL person={key} score={smoothed:.2f} "
                    f"relation={'n/a' if rel_s is None else f'{rel_s:.2f}'}"
                    f" geometry={geom:.2f} predicate={best_pred}")
            persons.append(PersonAssessment(
                detection_index=i, track_key=key, relation_score=rel_s,
                geometry_score=geom, fused_score=fused,
                smoothed_score=smoothed, status=state.status,
                safe_surface=safe, best_predicate=best_pred))
        self._expire_tracks(now)

        triplets: List[RelationTriplet] = []
        if rel is not None:
            thr = self.config.relation_display_threshold
            n = len(selected)
            best_p = np.argmax(rel.scores, axis=1)
            best_s = rel.scores[np.arange(len(best_p)), best_p]
            for k in np.argsort(-best_s):
                si, oi = int(rel.sub_idx[k]), int(rel.obj_idx[k])
                if (not rel.valid[k] or best_s[k] < thr or si >= n
                        or oi >= n or si == oi):
                    continue
                triplets.append(RelationTriplet(
                    si, oi, rel.predicates[int(best_p[k])],
                    float(best_s[k])))
                if len(triplets) >= 20:
                    break
        status = max((p.status for p in persons),
                     key=lambda s: s.severity, default=FallStatus.NORMAL)
        timings["reasoning"] = (time.perf_counter() - t2) * 1e3
        timings["total"] = (time.perf_counter() - t0) * 1e3
        annotated = self._draw(
            frame, selected, persons, triplets, status, timings,
            f"{self.mode.value}:{self.relation.name}/"
            f"{self.detector.backend_name}")
        return FrameResult(annotated_frame=annotated, status=status,
                           detections=selected, persons=persons,
                           triplets=triplets, alerts=alerts,
                           warnings=warnings, timings_ms=timings)


def main() -> int:
    """Command-line smoke test: process an image or a video file.

    Returns:
        int: Process exit code (0 on success).
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("source", help="Image or video path.")
    parser.add_argument("--mode", default=InferenceMode.ONNX_CPU.value,
                        choices=[m.value for m in InferenceMode])
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--output", default="annotated_output.jpg",
                        help="Output image (image source) or mp4 (video).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = whole video).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = FallDetectionConfig(artifacts_dir=Path(args.artifacts_dir))
        engine = FallDetectionEngine(cfg, mode=args.mode)
        image = cv2.imread(args.source)
        if image is not None:
            res = engine.process_frame(image)
            cv2.imwrite(args.output, res.annotated_frame)
            print(f"status={res.status.value} timings={res.timings_ms}")
            for t in res.triplets[:10]:
                print(f"  {res.detections[t.subject_index].label} "
                      f"--{t.predicate} [{t.score:.2f}]--> "
                      f"{res.detections[t.object_index].label}")
            return 0
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Cannot open {args.source}", file=sys.stderr)
            return 2
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer: Optional[cv2.VideoWriter] = None
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and index >= args.max_frames):
                break
            res = engine.process_frame(frame, timestamp=index / fps)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (w, h))
            writer.write(res.annotated_frame)
            for alert in res.alerts:
                print(alert)
            index += 1
        cap.release()
        if writer is not None:
            writer.release()
        print(f"Processed {index} frames -> {args.output}")
        return 0
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
