"""Model-free unit tests of the fall-reasoning logic in ``pipeline.py``.

The detector and relation backends are replaced by deterministic fakes so
that the reasoning, temporal filter and drawing code are tested in seconds
without downloading any weights.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import (Detection, FallDetectionConfig,  # noqa: E402
                      FallDetectionEngine, FallStatus, InferenceMode,
                      RelationBackend, RelationOutput, VocabularySpec,
                      boxes_to_normalized_cxcywh, intersection_over_first,
                      parse_vocabulary_spec, stable_sigmoid)


class FakeDetector:
    """Returns a scripted list of detections."""

    backend_name = "fake"

    def __init__(self) -> None:
        """Start with no detections."""
        self.detections: List[Detection] = []

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        """Return the scripted detections.

        Args:
            frame: Ignored.

        Returns:
            List[Detection]: Scripted detections.
        """
        return list(self.detections)

    def set_classes(self, classes: Sequence[str]) -> None:
        """Accept any vocabulary.

        Args:
            classes: Ignored.
        """

    def reset_tracker(self) -> None:
        """No state to reset."""


class FakeRelation(RelationBackend):
    """Scores ``lying on`` high when ``lying`` is True, else ``standing on``.
    """

    name = "fake"

    def __init__(self) -> None:
        """Start upright."""
        self._preds: List[str] = []
        self.lying = False

    @property
    def predicates(self) -> List[str]:
        """Return the active predicates.

        Returns:
            List[str]: Active predicates.
        """
        return list(self._preds)

    def set_predicates(self, predicates: Sequence[str]) -> List[str]:
        """Accept every predicate.

        Args:
            predicates: Requested predicates.

        Returns:
            List[str]: Empty list.
        """
        self._preds = list(predicates)
        return []

    def infer(self, frame_bgr: np.ndarray,
              boxes_xyxy: np.ndarray) -> RelationOutput:
        """Produce scores for every ordered pair.

        Args:
            frame_bgr: Ignored.
            boxes_xyxy: ``[N, 4]`` boxes.

        Returns:
            RelationOutput: Synthetic scores.
        """
        n = len(boxes_xyxy)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        scores = np.full((len(pairs), len(self._preds)), 0.05, np.float32)
        hot = "lying on" if self.lying else "standing on"
        col = self._preds.index(hot)
        scores[:, col] = 0.9
        return RelationOutput(
            scores=scores,
            sub_idx=np.array([p[0] for p in pairs], np.int64),
            obj_idx=np.array([p[1] for p in pairs], np.int64),
            valid=np.ones(len(pairs), bool), predicates=self.predicates)


def make_engine(**overrides: float) -> FallDetectionEngine:
    """Build an engine around the fakes, bypassing model loading.

    Args:
        **overrides: ``FallDetectionConfig`` field overrides.

    Returns:
        FallDetectionEngine: Engine with fake backends.
    """
    engine = FallDetectionEngine.__new__(FallDetectionEngine)
    engine.config = FallDetectionConfig(use_virtual_floor=True, **overrides)
    engine.mode = InferenceMode.ONNX_CPU
    engine._lock = threading.Lock()
    engine._tracks = {}
    engine.vocab_warnings = []
    engine.detector = FakeDetector()
    engine.relation = FakeRelation()
    engine._apply_relation_vocabulary(engine.config.vocabulary)
    return engine


FRAME = np.zeros((480, 640, 3), np.uint8)
STANDING = np.array([300, 100, 360, 400], np.float32)   # tall box
LYING = np.array([200, 380, 440, 460], np.float32)      # wide, on floor


def test_parse_vocabulary_sections_and_person_forced() -> None:
    """Sections are parsed and ``person`` is always kept."""
    spec = parse_vocabulary_spec("objects: floor, bed\nfall: lying on")
    assert spec.objects == ["person", "floor", "bed"]
    assert spec.fall_predicates == ["lying on"]
    assert spec.upright_predicates == list(VocabularySpec().upright_predicates)


def test_parse_vocabulary_plain_line_and_errors() -> None:
    """A bare line is the object list; bad input raises ValueError."""
    assert parse_vocabulary_spec("person, Ground").objects == [
        "person", "ground"]
    with pytest.raises(ValueError):
        parse_vocabulary_spec("colour: red")
    with pytest.raises(ValueError):
        parse_vocabulary_spec("fall: ,")


def test_geometry_helpers() -> None:
    """IoA, box conversion and sigmoid behave as documented."""
    floor = np.array([0, 360, 640, 480], np.float32)
    assert intersection_over_first(LYING, floor) == pytest.approx(1.0)
    assert 0.0 < intersection_over_first(STANDING, floor) < 0.2
    cxcywh = boxes_to_normalized_cxcywh(np.array([[0, 0, 640, 480]]), 640,
                                        480)
    np.testing.assert_allclose(cxcywh, [[0.5, 0.5, 1.0, 1.0]])
    sig = stable_sigmoid(np.array([-np.inf, -1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(sig, [0.0, 0.0, 0.5, 1.0])


def test_standing_person_stays_normal() -> None:
    """An upright person never raises an alert."""
    engine = make_engine()
    engine.detector.detections = [Detection(STANDING, "person", 0.9, 1)]
    for t in range(10):
        res = engine.process_frame(FRAME, timestamp=t * 0.1)
    assert res.status is FallStatus.NORMAL
    assert not res.alerts
    assert res.detections[-1].label == "floor (virtual)"


def test_fall_requires_persistence_then_alerts_once() -> None:
    """A fall is SUSPECTED first, confirmed after ``min_fall_frames``, and
    alerted once inside the cooldown."""
    engine = make_engine(min_fall_frames=4, smoothing_window=6)
    det = engine.detector
    # Two upright frames establish the person's standing height.
    det.detections = [Detection(STANDING, "person", 0.9, 7)]
    for t in range(2):
        engine.process_frame(FRAME, timestamp=t * 0.1)
    engine.relation.lying = True
    det.detections = [Detection(LYING, "person", 0.9, 7)]
    statuses, alerts = [], []
    for t in range(2, 12):
        res = engine.process_frame(FRAME, timestamp=t * 0.1)
        statuses.append(res.status)
        alerts += res.alerts
    assert statuses[0] is FallStatus.SUSPECTED
    assert statuses[-1] is FallStatus.FALL
    assert len(alerts) == 1 and "person=7" in alerts[0]
    assert res.annotated_frame.shape == FRAME.shape


def test_safe_surface_suppresses_alarm() -> None:
    """Lying on a bed more strongly than on the floor is not a fall."""
    engine = make_engine()
    engine.relation.lying = True

    original = engine.relation.infer

    def bed_preferred(frame: np.ndarray,
                      boxes: np.ndarray) -> RelationOutput:
        """Make person->bed the strongest ``lying on`` pair."""
        out = original(frame, boxes)
        col = out.predicates.index("lying on")
        bed_pairs = (out.sub_idx == 0) & (out.obj_idx == 2)
        out.scores[bed_pairs, col] = 0.99
        return out

    engine.relation.infer = bed_preferred
    bed = np.array([150, 300, 500, 470], np.float32)
    engine.detector.detections = [
        Detection(LYING, "person", 0.9, 3),
        Detection(np.array([0, 360, 640, 480], np.float32), "floor", 0.5),
        Detection(bed, "bed", 0.8)]
    for t in range(10):
        res = engine.process_frame(FRAME, timestamp=t * 0.1)
    assert res.persons[0].safe_surface == "bed"
    assert res.status is not FallStatus.FALL


def test_track_expiry_and_invalid_frame() -> None:
    """Old tracks are dropped; malformed frames are rejected."""
    engine = make_engine(track_ttl_s=1.0)
    engine.detector.detections = [Detection(STANDING, "person", 0.9, 5)]
    engine.process_frame(FRAME, timestamp=0.0)
    assert "5" in engine._tracks
    engine.detector.detections = []
    engine.process_frame(FRAME, timestamp=5.0)
    assert "5" not in engine._tracks
    with pytest.raises(ValueError):
        engine.process_frame(np.zeros((10, 10), np.uint8))
