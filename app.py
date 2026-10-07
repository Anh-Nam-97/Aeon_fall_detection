"""Gradio real-time testing UI for Aeon_fall_detection.

Two input paths share one engine per inference mode:

* **Webcam** - the browser streams frames to the server (works through
  Docker and remote servers because capture happens client-side).
* **Video file** - frames are decoded server-side, annotated, streamed back
  live, and the full annotated clip is returned at the end.

Run::

    python app.py --mode onnx_cpu          # Intel i7-1260P edge box
    python app.py --mode torch_gpu --host 0.0.0.0   # RTX 4090 / 5090
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from pipeline import (DEFAULT_ARTIFACTS_DIR, FallDetectionConfig,
                      FallDetectionEngine, FallStatus, FrameResult,
                      InferenceMode, VocabularySpec, parse_vocabulary_spec)

LOGGER = logging.getLogger("aeon_fall_detection.app")

#: Text pre-filled in the vocabulary box; it documents the syntax by example.
DEFAULT_VOCAB_TEXT: str = (
    "objects: person, floor, bed, sofa\n"
    "fall: lying on, laying on, resting on, sitting on\n"
    "upright: standing on, walking on, walking past, standing beside")

#: Maximum lines kept in the alert log, so a long session does not grow the
#: payload sent to the browser on every frame.
MAX_LOG_LINES: int = 200

#: For video files, only every n-th annotated frame is pushed to the
#: browser. Every frame is still analysed (temporal logic needs them all);
#: this only bounds network traffic.
UI_FRAME_STRIDE: int = 2


class EngineRegistry:
    """Lazily creates and caches one engine per inference mode.

    Loading a model takes seconds to minutes (download + compile), so
    engines are created on first use and reused. A lock prevents two
    browser tabs from building the same engine concurrently.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        """Create an empty registry.

        Args:
            artifacts_dir: Directory produced by ``export_onnx.py``.
        """
        self.artifacts_dir = artifacts_dir
        self._engines: Dict[InferenceMode, FallDetectionEngine] = {}
        self._vocab_text: Dict[InferenceMode, str] = {}
        self._lock = threading.Lock()

    def get(self, mode: InferenceMode, vocab_text: str
            ) -> Tuple[FallDetectionEngine, List[str]]:
        """Return a ready engine with the requested vocabulary.

        Args:
            mode: Inference mode.
            vocab_text: Raw vocabulary text from the UI.

        Returns:
            Tuple[FallDetectionEngine, List[str]]: Engine and vocabulary
            warnings (e.g. phrases missing from the ONNX bank).

        Raises:
            ValueError: If the vocabulary text is malformed.
            RuntimeError: If the engine cannot be built.
            FileNotFoundError: If ONNX artifacts are missing.
        """
        spec = parse_vocabulary_spec(vocab_text, VocabularySpec())
        with self._lock:
            engine = self._engines.get(mode)
            if engine is None:
                cfg = FallDetectionConfig(artifacts_dir=self.artifacts_dir,
                                          vocabulary=spec)
                engine = FallDetectionEngine(cfg, mode=mode)
                self._engines[mode] = engine
                self._vocab_text[mode] = vocab_text
                return engine, list(engine.vocab_warnings)
            if self._vocab_text.get(mode) != vocab_text:
                warnings = engine.update_vocabulary(spec)
                self._vocab_text[mode] = vocab_text
                return engine, warnings
            return engine, list(engine.vocab_warnings)


class AlertLog:
    """Bounded, de-duplicated text log shown in the UI."""

    def __init__(self) -> None:
        """Create an empty log."""
        self._lines: Deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._seen_warnings: set = set()

    def add(self, line: str) -> None:
        """Append a line.

        Args:
            line: Text to append.
        """
        self._lines.append(line)

    def add_result(self, result: FrameResult) -> None:
        """Append alerts and first-seen warnings of a frame.

        Warnings such as "virtual floor in use" repeat every frame; logging
        them once keeps real alerts visible.

        Args:
            result: Frame result.
        """
        for alert in result.alerts:
            self._lines.append(alert)
        for warning in result.warnings:
            self.add_warning(warning)

    def add_warning(self, warning: str) -> None:
        """Append a warning only the first time it is seen.

        Args:
            warning: Warning text.
        """
        if warning not in self._seen_warnings:
            self._seen_warnings.add(warning)
            self._lines.append(f"[warn] {warning}")

    def text(self) -> str:
        """Return the log, newest line first.

        Returns:
            str: Log text.
        """
        return "\n".join(reversed(self._lines))


def status_markdown(result: Optional[FrameResult]) -> str:
    """Format the status panel.

    Args:
        result: Latest frame result, or ``None``.

    Returns:
        str: Markdown text.
    """
    if result is None:
        return "**Status:** waiting for frames"
    icon = {FallStatus.NORMAL: "🟢", FallStatus.SUSPECTED: "🟠",
            FallStatus.FALL: "🔴"}[result.status]
    t = result.timings_ms
    return (f"**Status:** {icon} {result.status.value} &nbsp;|&nbsp; "
            f"people: {len(result.persons)} &nbsp;|&nbsp; "
            f"detect {t.get('detect', 0):.0f} ms · relation "
            f"{t.get('relation', 0):.0f} ms · total "
            f"{t.get('total', 0):.0f} ms")


def build_ui(registry: EngineRegistry, default_mode: InferenceMode
             ) -> gr.Blocks:
    """Build the Gradio interface.

    Args:
        registry: Engine registry.
        default_mode: Mode pre-selected in the UI.

    Returns:
        gr.Blocks: The application.
    """
    modes = [m.value for m in InferenceMode]

    def analyse_video(video_path: Optional[str], mode: str, vocab: str,
                      threshold: float
                      ) -> Iterator[Tuple[Optional[np.ndarray],
                                          Optional[str], str, str]]:
        """Process a video file and stream annotated frames.

        Args:
            video_path: Uploaded file path.
            mode: Inference mode value.
            vocab: Vocabulary text.
            threshold: Fall threshold.

        Yields:
            Tuple: (annotated RGB frame, output video path, alert log,
            status markdown).
        """
        log = AlertLog()
        if not video_path:
            yield None, None, "Please upload a video.", status_markdown(None)
            return
        try:
            engine, warns = registry.get(InferenceMode(mode), vocab)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            yield None, None, f"[error] {exc}", status_markdown(None)
            return
        for w in warns:
            log.add_warning(w)
        engine.config.fall_threshold = float(threshold)
        engine.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield None, None, f"[error] cannot open {video_path}", \
                status_markdown(None)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_path = os.path.join(tempfile.mkdtemp(prefix="aeon_"),
                                "annotated.mp4")
        writer: Optional[cv2.VideoWriter] = None
        index, last = 0, None
        log.add(f"[info] processing {Path(video_path).name} at "
                f"{fps:.1f} fps, mode={mode}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # Video clock, not wall clock: temporal thresholds then mean
                # "seconds of footage" regardless of processing speed.
                last = engine.process_frame(frame, timestamp=index / fps)
                log.add_result(last)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (w, h))
                writer.write(last.annotated_frame)
                if index % UI_FRAME_STRIDE == 0:
                    yield (cv2.cvtColor(last.annotated_frame,
                                        cv2.COLOR_BGR2RGB), None,
                           log.text(), status_markdown(last))
                index += 1
        except Exception as exc:  # noqa: BLE001 - surface in the UI
            LOGGER.exception("Video processing failed")
            log.add(f"[error] {exc}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        log.add(f"[info] done: {index} frames")
        final = (cv2.cvtColor(last.annotated_frame, cv2.COLOR_BGR2RGB)
                 if last is not None else None)
        yield (final, out_path if index else None, log.text(),
               status_markdown(last))

    def analyse_webcam(frame_rgb: Optional[np.ndarray], mode: str,
                       vocab: str, threshold: float, log_state: AlertLog
                       ) -> Tuple[Optional[np.ndarray], str, str, AlertLog]:
        """Process one webcam frame.

        Args:
            frame_rgb: RGB frame from the browser.
            mode: Inference mode value.
            vocab: Vocabulary text.
            threshold: Fall threshold.
            log_state: Per-session alert log.

        Returns:
            Tuple: (annotated RGB frame, alert log, status, log state).
        """
        if log_state is None:
            log_state = AlertLog()
        if frame_rgb is None:
            return None, log_state.text(), status_markdown(None), log_state
        try:
            engine, warns = registry.get(InferenceMode(mode), vocab)
            for w in warns:
                log_state.add_warning(w)
            engine.config.fall_threshold = float(threshold)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result = engine.process_frame(frame_bgr, timestamp=time.time())
            log_state.add_result(result)
            return (cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                    log_state.text(), status_markdown(result), log_state)
        except Exception as exc:  # noqa: BLE001 - surface in the UI
            LOGGER.exception("Webcam frame failed")
            log_state.add(f"[error] {exc}")
            return frame_rgb, log_state.text(), status_markdown(None), \
                log_state

    with gr.Blocks(title="Aeon Fall Detection") as demo:
        gr.Markdown(
            "# Aeon Fall Detection\n"
            "Zero-shot fall detection: **YOLO-World (YOLOv8)** finds people "
            "and the floor, **RelateAnything** scores relations such as "
            "*person - lying on - floor*, and a temporal filter confirms "
            "the fall.")
        with gr.Row():
            mode_dd = gr.Dropdown(modes, value=default_mode.value,
                                  label="Inference mode")
            threshold = gr.Slider(0.1, 0.95, value=0.5, step=0.01,
                                  label="Fall threshold")
        vocab_box = gr.Textbox(value=DEFAULT_VOCAB_TEXT, lines=3,
                               label="Custom vocabulary (objects / fall / "
                                     "upright)")
        status_md = gr.Markdown(status_markdown(None))
        with gr.Tabs():
            with gr.Tab("Video file"):
                with gr.Row():
                    with gr.Column():
                        video_in = gr.Video(label="Input video",
                                            sources=["upload"])
                        run_btn = gr.Button("Analyse", variant="primary")
                    with gr.Column():
                        video_live = gr.Image(label="Annotated (live)",
                                              type="numpy")
                        video_out = gr.Video(label="Annotated video")
                video_log = gr.Textbox(label="Alert log", lines=12,
                                       max_lines=12, autoscroll=False)
                run_btn.click(analyse_video,
                              inputs=[video_in, mode_dd, vocab_box,
                                      threshold],
                              outputs=[video_live, video_out, video_log,
                                       status_md],
                              concurrency_limit=1)
            with gr.Tab("Webcam"):
                log_state = gr.State(None)
                with gr.Row():
                    cam = gr.Image(sources=["webcam"], streaming=True,
                                   type="numpy", label="Webcam")
                    cam_out = gr.Image(label="Annotated (live)",
                                       type="numpy")
                cam_log = gr.Textbox(label="Alert log", lines=12,
                                     max_lines=12, autoscroll=False)
                cam.stream(analyse_webcam,
                           inputs=[cam, mode_dd, vocab_box, threshold,
                                   log_state],
                           outputs=[cam_out, cam_log, status_md, log_state],
                           stream_every=0.1, concurrency_limit=1)
    return demo


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, optionally pre-load the engine, launch the UI.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        int: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default=os.environ.get(
        "AEON_MODE", InferenceMode.ONNX_CPU.value),
        choices=[m.value for m in InferenceMode])
    parser.add_argument("--host", default=os.environ.get("AEON_HOST",
                                                         "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("AEON_PORT", "7860")))
    parser.add_argument("--artifacts-dir", default=os.environ.get(
        "AEON_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))
    parser.add_argument("--preload", action="store_true",
                        help="Build the engine before serving (fail fast).")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    registry = EngineRegistry(Path(args.artifacts_dir))
    mode = InferenceMode(args.mode)
    if args.preload:
        try:
            registry.get(mode, DEFAULT_VOCAB_TEXT)
            LOGGER.info("Engine %s ready", mode.value)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            LOGGER.error("Engine preload failed: %s", exc)
            return 1
    demo = build_ui(registry, mode)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
