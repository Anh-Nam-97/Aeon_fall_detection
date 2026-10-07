"""Export RelateAnything to ONNX (and OpenVINO IR) for the Intel i7-1260P.

What the script produces in ``--out-dir``:

* ``relateanything.onnx``  - relation graph. The predicate embedding matrix
  ``W [V, D]`` and routing weights ``alpha [V]`` are graph *inputs*, so the
  fall vocabulary can change at run time without re-exporting.
* ``relateanything.json``  - sidecar metadata (image size, box slots, score
  calibration, predicate list, validation deltas).
* ``predicate_bank.npz``   - pre-encoded predicates (``names``, ``W``,
  ``alpha``, ``default``, ``is_spatial``). The CPU runtime never needs the
  text encoder or PyTorch.
* ``relateanything_fp16.xml/.bin`` - OpenVINO IR with FP16-compressed
  weights (optional, ``--openvino``). Alder Lake has no native FP16 compute;
  OpenVINO decompresses to FP32 at load time, so this halves disk/RAM
  footprint without accuracy loss.

Dynamic axes:

* ``boxes`` axis 1 (``num_boxes``) and ``W``/``alpha`` axis 0
  (``num_predicates``) are declared dynamic.
* The *number of real boxes* per frame is carried by ``box_counts``; boxes
  are zero-padded to ``--max-boxes`` slots. This is the contract used by the
  upstream runtime and the one validated for exact parity below. The
  ``num_boxes`` axis itself executes at other slot counts, but the traced
  pair sampler is only numerically faithful at the exported count (a
  24-slot run selected ~84% of PyTorch's pairs during development), so
  change capacity by re-exporting with ``--max-boxes`` rather than by
  feeding a different slot count. ``--max-boxes`` must satisfy
  ``N * (N - 1) >= K`` (K = 128 pair budget, i.e. N >= 12).

Usage::

    python export_onnx.py --out-dir models/relsgg-vits16plus --openvino
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("aeon_fall_detection.export")

PROJECT_ROOT: Path = Path(__file__).resolve().parent

#: Phrases always exported into the predicate bank. The fall-specific set is
#: merged with RelateAnything's default vocabulary so that the CPU runtime
#: can also render a general scene graph and so that users have a large pool
#: of phrases to choose from in the UI without re-exporting.
FALL_VOCABULARY: Tuple[str, ...] = (
    "lying on", "laying on", "resting on", "sitting on", "kneeling on",
    "crawling on", "fallen on", "lying next to", "standing on", "walking on",
    "walking past", "standing beside", "leaning against", "holding",
)

#: Absolute tolerance on logits between PyTorch and ONNX Runtime. FP32
#: reorderings in fused kernels typically yield ~1e-4 on ViT-S; 5e-3 leaves
#: margin while still catching real export bugs (which produce errors > 0.1).
LOGIT_TOLERANCE: float = 5e-3

#: Tolerance for the OpenVINO IR. FP16 weight storage keeps 11 significant
#: bits, which shifts logits by up to ~1e-1 on a ViT-S; ranking (what the
#: fall logic consumes) is unaffected at that level.
FP16_LOGIT_TOLERANCE: float = 0.25

#: Fraction of reference pairs the FP16 IR must also select. The sampler
#: keeps the top-K pairs, so FP16 noise may swap a few pairs at the K-th
#: rank; the person/floor pairs the engine needs rank far above that edge.
FP16_MIN_PAIR_OVERLAP: float = 0.9


def build_vocabulary(extra: Optional[Sequence[str]]) -> List[str]:
    """Merge fall predicates, upstream defaults and user extras.

    Args:
        extra: Additional phrases from ``--predicates``.

    Returns:
        List[str]: De-duplicated, order-preserving lower-case phrases.
    """
    from relsgg.vocabulary import DEFAULT_PREDICATES

    merged: List[str] = []
    for phrase in [*FALL_VOCABULARY, *DEFAULT_PREDICATES, *(extra or [])]:
        phrase = phrase.strip().lower()
        if phrase and phrase not in merged:
            merged.append(phrase)
    return merged


def make_export_wrapper(model: "torch.nn.Module") -> "torch.nn.Module":
    """Wrap the relation model with a flat, export-friendly signature.

    Args:
        model: ``relsgg`` ``RelSGG`` module in eval mode.

    Returns:
        torch.nn.Module: Module ``(image, boxes, box_counts, W, alpha) ->
        (pred_logits, pair_logits, sub_idx, obj_idx, valid_mask)``.
    """
    import torch

    class RelationExportWrapper(torch.nn.Module):
        """Inference-only wrapper returning raw logits.

        Thresholding is kept on the host: a threshold inside the graph would
        make output shapes data-dependent (NonZero), which breaks shape
        inference and static-memory backends, while the host-side compare
        costs microseconds.
        """

        def __init__(self, inner: torch.nn.Module) -> None:
            """Store the wrapped module.

            Args:
                inner: The ``RelSGG`` module.
            """
            super().__init__()
            self.inner = inner

        def forward(self, image: torch.Tensor, boxes: torch.Tensor,
                    box_counts: torch.Tensor, W: torch.Tensor,
                    alpha: torch.Tensor
                    ) -> Tuple[torch.Tensor, ...]:
            """Run the model with the vocabulary supplied as inputs.

            Args:
                image: ``[B, 3, S, S]`` float image in ``[0, 1]``.
                boxes: ``[B, N, 4]`` normalised ``cxcywh`` boxes.
                box_counts: ``[B]`` number of real boxes.
                W: ``[V, D]`` L2-normalised predicate embeddings.
                alpha: ``[V]`` spatial/semantic routing weights.

            Returns:
                Tuple[torch.Tensor, ...]: ``pred_logits [B, K, V]``,
                ``pair_logits [B, K]``, ``sub_idx [B, K]``,
                ``obj_idx [B, K]``, ``valid_mask [B, K]``.
            """
            # W/alpha are registered buffers of the vocabulary head. Plain
            # assignment is traced as data flow, so the exported matmul
            # reads the graph inputs instead of frozen constants.
            self.inner.vocab_head.W = W
            self.inner.vocab_head.alpha = alpha
            out = self.inner(image, boxes, box_counts=box_counts,
                             targets=None)
            pred = out["logits"]
            pair = out.get("pair_logits")
            if pair is None:
                # 0 is the identity of the additive score fusion.
                pair = torch.zeros_like(pred[..., 0])
            return (pred, pair, out["sub_idx"].to(torch.int64),
                    out["obj_idx"].to(torch.int64),
                    out["valid_mask"].to(torch.bool))

    return RelationExportWrapper(model).eval()


def dummy_inputs(num_slots: int, num_real: int, img_size: int,
                 W: np.ndarray, alpha: np.ndarray,
                 seed: int = 0) -> Dict[str, np.ndarray]:
    """Create a reproducible dummy feed.

    Boxes are random but well-formed (centres inside the image, positive
    sizes); padded slots are zero, as at run time.

    Args:
        num_slots: Padded box slots (graph ``num_boxes`` axis).
        num_real: Real boxes declared in ``box_counts``.
        img_size: Square image side.
        W: ``[V, D]`` predicate embeddings.
        alpha: ``[V]`` routing weights.
        seed: RNG seed.

    Returns:
        Dict[str, np.ndarray]: Feed keyed by graph input names.
    """
    rng = np.random.default_rng(seed)
    boxes = np.zeros((1, num_slots, 4), dtype=np.float32)
    centres = rng.uniform(0.25, 0.75, size=(num_real, 2))
    sizes = rng.uniform(0.1, 0.4, size=(num_real, 2))
    boxes[0, :num_real] = np.concatenate([centres, sizes], axis=1)
    return {
        "image": rng.random((1, 3, img_size, img_size), dtype=np.float32),
        "boxes": boxes,
        "box_counts": np.array([num_real], dtype=np.int64),
        "W": np.ascontiguousarray(W, dtype=np.float32),
        "alpha": np.ascontiguousarray(alpha, dtype=np.float32),
    }


OUTPUT_NAMES: List[str] = ["pred_logits", "pair_logits", "sub_idx",
                           "obj_idx", "valid_mask"]
INPUT_NAMES: List[str] = ["image", "boxes", "box_counts", "W", "alpha"]


def _pair_table(outs: Sequence[np.ndarray]
                ) -> Dict[Tuple[int, int], Tuple[np.ndarray, float]]:
    """Index valid pairs of one output set by ``(subject, object)``.

    Args:
        outs: Outputs in ``OUTPUT_NAMES`` order (batch of 1).

    Returns:
        Dict[Tuple[int, int], Tuple[np.ndarray, float]]: Predicate logits
        ``[V]`` and pair logit for each valid pair.
    """
    pred, pair, sub, obj, valid = (np.asarray(o)[0] for o in outs)
    return {(int(sub[k]), int(obj[k])): (pred[k], float(pair[k]))
            for k in range(len(valid)) if bool(valid[k])}


def compare_outputs(name: str, ref: Sequence[np.ndarray],
                    got: Sequence[np.ndarray], tolerance: float,
                    min_pair_overlap: float = 1.0) -> float:
    """Compare reference and candidate outputs pair by pair.

    The model's pair sampler keeps the top-K pairs by a learned score, so
    the *row order* of the outputs depends on tiny numeric differences (and
    with FP16 weights the last few pairs at the top-K boundary may differ).
    Rows are therefore matched by their ``(subject, object)`` key, never by
    position. Padded rows are ignored: they carry ``-inf`` by design.

    Args:
        name: Label of the check (for logs).
        ref: PyTorch outputs.
        got: Runtime outputs.
        tolerance: Maximum absolute logit difference on shared pairs.
        min_pair_overlap: Minimum fraction of reference pairs that must also
            be selected by the runtime (1.0 = identical pair sets).

    Returns:
        float: Worst absolute logit difference on shared pairs.

    Raises:
        AssertionError: If shapes, pair sets or logits disagree.
    """
    assert np.shape(ref[0]) == np.shape(got[0]), (
        f"[{name}] pred_logits shape {np.shape(got[0])} != "
        f"{np.shape(ref[0])}")
    r_tab, g_tab = _pair_table(ref), _pair_table(got)
    shared = r_tab.keys() & g_tab.keys()
    overlap = len(shared) / max(len(r_tab), 1)
    assert overlap >= min_pair_overlap, (
        f"[{name}] only {overlap:.1%} of reference pairs selected")
    worst = 0.0
    for key in shared:
        (r_pred, r_pair), (g_pred, g_pair) = r_tab[key], g_tab[key]
        finite = np.isfinite(r_pred) & np.isfinite(g_pred)
        if finite.any():
            worst = max(worst, float(np.abs(r_pred[finite]
                                            - g_pred[finite]).max()))
        worst = max(worst, abs(r_pair - g_pair))
    assert worst <= tolerance, (
        f"[{name}] max |delta| {worst:.3e} > {tolerance}")
    LOGGER.info("[check] %-30s pairs=%4d overlap=%5.1f%% max|delta|=%.2e "
                "OK", name, len(r_tab), overlap * 100, worst)
    return worst


def torch_reference(wrapper: "torch.nn.Module",
                    feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Run the PyTorch wrapper on a numpy feed.

    Args:
        wrapper: Export wrapper.
        feed: Inputs keyed by name.

    Returns:
        List[np.ndarray]: Outputs in ``OUTPUT_NAMES`` order.
    """
    import torch

    with torch.no_grad():
        outs = wrapper(*[torch.from_numpy(feed[n]) for n in INPUT_NAMES])
    return [o.detach().cpu().numpy() for o in outs]


def validate_onnx(onnx_path: Path, wrapper: "torch.nn.Module",
                  img_size: int, max_boxes: int, W: np.ndarray,
                  alpha: np.ndarray) -> Dict[str, float]:
    """Validate the exported graph with dummy inference.

    Checks, in order:

    1. ONNX structural checker.
    2. Full slots, full vocabulary: parity with PyTorch.
    3. Variable box count (2 real boxes in ``max_boxes`` slots): parity.
    4. Variable vocabulary size (V=5): parity, proving the dynamic V axis.
    5. Larger slot count (``max_boxes + 8``): the dynamic box axis.
       Reported as a warning on failure, because the runtime always pads to
       ``max_boxes`` and therefore never depends on it. Smaller slot counts
       are not tested on purpose: the pair sampler's top-K (``final_budget``,
       128 for the released checkpoints) is a graph constant, so a graph
       needs ``N * (N - 1) >= K`` candidate pairs (N >= 12).

    Args:
        onnx_path: Exported graph.
        wrapper: PyTorch wrapper used as reference.
        img_size: Square image side.
        max_boxes: Box slots used at export.
        W: Full ``[V, D]`` vocabulary matrix.
        alpha: Full ``[V]`` routing weights.

    Returns:
        Dict[str, float]: Worst logit delta per check.

    Raises:
        AssertionError: If a mandatory check fails.
    """
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(onnx_path))
    LOGGER.info("[check] onnx.checker: OK")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), opts,
                                providers=["CPUExecutionProvider"])
    deltas: Dict[str, float] = {}
    cases = [
        ("full slots / full vocab", max_boxes, max_boxes, W, alpha),
        ("2 real boxes (padded)", max_boxes, 2, W, alpha),
        ("vocab V=5 (dynamic axis)", max_boxes, 3, W[:5], alpha[:5]),
    ]
    for label, slots, real, w, a in cases:
        feed = dummy_inputs(slots, real, img_size, w, a)
        ref = torch_reference(wrapper, feed)
        t0 = time.perf_counter()
        got = sess.run(OUTPUT_NAMES, feed)
        LOGGER.info("[check] %-30s ORT latency %.1f ms", label,
                    (time.perf_counter() - t0) * 1e3)
        deltas[label] = compare_outputs(label, ref, got, LOGIT_TOLERANCE)
    more = max_boxes + 8
    label = f"{more} slots (dynamic axis)"
    try:
        feed = dummy_inputs(more, more, img_size, W, alpha)
        deltas[label] = compare_outputs(
            label, torch_reference(wrapper, feed),
            sess.run(OUTPUT_NAMES, feed), LOGIT_TOLERANCE)
    except Exception as exc:  # noqa: BLE001 - soft check, see docstring
        LOGGER.warning("[check] %s failed (%s). The runtime pads to %d "
                       "slots, so this is informational.", label, exc,
                       max_boxes)
    return deltas


def export_openvino(onnx_path: Path, wrapper: "torch.nn.Module",
                    img_size: int, max_boxes: int, W: np.ndarray,
                    alpha: np.ndarray) -> Optional[float]:
    """Convert the ONNX graph to an OpenVINO IR and validate it.

    Args:
        onnx_path: Exported ONNX graph.
        wrapper: PyTorch reference.
        img_size: Square image side.
        max_boxes: Box slots.
        W: Vocabulary matrix.
        alpha: Routing weights.

    Returns:
        Optional[float]: Worst logit delta, or ``None`` if OpenVINO failed.
    """
    try:
        import openvino as ov

        ov_model = ov.convert_model(str(onnx_path))
        xml_path = onnx_path.with_name("relateanything_fp16.xml")
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=True)
        compiled = ov.Core().compile_model(
            str(xml_path), "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        feed = dummy_inputs(max_boxes, max_boxes, img_size, W, alpha)
        request = compiled.create_infer_request()
        request.infer(feed)  # warm-up: first call includes lazy init
        t0 = time.perf_counter()
        res = request.infer(feed)
        LOGGER.info("[check] OpenVINO CPU latency %.1f ms",
                    (time.perf_counter() - t0) * 1e3)
        got = [np.asarray(res[compiled.output(n)]) for n in OUTPUT_NAMES]
        ref = torch_reference(wrapper, feed)
        delta = compare_outputs("OpenVINO FP16 IR", ref, got,
                                FP16_LOGIT_TOLERANCE, FP16_MIN_PAIR_OVERLAP)
        LOGGER.info("Wrote %s", xml_path)
        return delta
    except Exception as exc:  # noqa: BLE001 - optional stage
        # Never leave an unvalidated IR behind: the runtime prefers the IR
        # whenever it exists, so a broken one would silently be used.
        for suffix in (".xml", ".bin"):
            onnx_path.with_name("relateanything_fp16" + suffix).unlink(
                missing_ok=True)
        LOGGER.warning("OpenVINO export skipped: %s", exc)
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-id", default="maelic/relsgg-vits16plus",
                   help="Hugging Face checkpoint id.")
    p.add_argument("--checkpoint", default="",
                   help="Local model.pth (overrides --model-id).")
    p.add_argument("--repo-dir", default=str(PROJECT_ROOT / "RelateAnything"),
                   help="Cloned RelateAnything repository.")
    p.add_argument("--out-dir",
                   default=str(PROJECT_ROOT / "models" / "relsgg-vits16plus"))
    p.add_argument("--predicates", nargs="*", default=None,
                   help="Extra predicate phrases for the bank.")
    p.add_argument("--max-boxes", type=int, default=16,
                   help="Box slots of the graph (16 covers a care room).")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--openvino", action="store_true",
                   help="Also write and validate an OpenVINO FP16 IR.")
    p.add_argument("--skip-validate", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Export, write sidecars and validate.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        int: Exit code (0 success, 1 failure).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        repo = Path(args.repo_dir)
        if repo.is_dir() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import torch
        from relsgg import RelateAnything

        torch.set_grad_enabled(False)
        # nn.TransformerEncoderLayer's fused fast path traces to an op with
        # no ONNX symbolic; disabling it exports the same maths unfused.
        torch.backends.mha.set_fastpath_enabled(False)

        vocab = build_vocabulary(args.predicates)
        LOGGER.info("Loading RelateAnything (%s) with %d predicates",
                    args.checkpoint or args.model_id, len(vocab))
        if args.checkpoint:
            ra = RelateAnything.from_checkpoint(args.checkpoint, vocab,
                                                device="cpu")
        else:
            ra = RelateAnything.from_pretrained(args.model_id,
                                                predicates=vocab,
                                                device="cpu")
        model = ra.model.eval()
        W = model.vocab_head.W.detach().cpu().numpy().astype(np.float32)
        alpha = (model.vocab_head.alpha.detach().cpu().numpy()
                 .astype(np.float32))
        img_size = int(ra.img_size)
        LOGGER.info("V=%d  D=%d  img=%d  slots=%d", W.shape[0], W.shape[1],
                    img_size, args.max_boxes)

        budget = int(getattr(model.config, "final_budget", 0) or 0)
        min_slots = int(np.ceil((1 + np.sqrt(1 + 4 * budget)) / 2))
        if args.max_boxes * (args.max_boxes - 1) < budget:
            # The sampler's top-K over N*(N-1) candidate pairs is a graph
            # constant; fewer candidates make TopK fail at run time.
            raise ValueError(
                f"--max-boxes {args.max_boxes} gives "
                f"{args.max_boxes * (args.max_boxes - 1)} candidate pairs, "
                f"fewer than the model's pair budget K={budget}. Use "
                f"--max-boxes >= {min_slots}.")
        wrapper = make_export_wrapper(model)
        feed = dummy_inputs(args.max_boxes, args.max_boxes, img_size, W,
                            alpha)
        example = tuple(torch.from_numpy(feed[n]) for n in INPUT_NAMES)
        dynamic_axes = {
            "image": {0: "batch"},
            "boxes": {0: "batch", 1: "num_boxes"},
            "box_counts": {0: "batch"},
            "W": {0: "num_predicates"},
            "alpha": {0: "num_predicates"},
            "pred_logits": {0: "batch", 1: "num_pairs",
                            2: "num_predicates"},
            "pair_logits": {0: "batch", 1: "num_pairs"},
            "sub_idx": {0: "batch", 1: "num_pairs"},
            "obj_idx": {0: "batch", 1: "num_pairs"},
            "valid_mask": {0: "batch", 1: "num_pairs"},
        }
        onnx_path = out_dir / "relateanything.onnx"
        LOGGER.info("Exporting to %s (opset %d)", onnx_path, args.opset)
        t0 = time.perf_counter()
        # The TorchScript exporter (dynamo=False) is used because it honours
        # ``dynamic_axes`` and is the path the upstream release is validated
        # with; the dynamo exporter needs explicit ``Dim`` specs per axis.
        torch.onnx.export(wrapper, example, str(onnx_path),
                          input_names=INPUT_NAMES,
                          output_names=OUTPUT_NAMES,
                          dynamic_axes=dynamic_axes,
                          opset_version=args.opset,
                          do_constant_folding=True, dynamo=False)
        LOGGER.info("Exported in %.1f s (%.0f MB)", time.perf_counter() - t0,
                    onnx_path.stat().st_size / 1e6)

        # Spatial/semantic type of each predicate (two-graph rendering).
        try:
            is_spatial = np.asarray(ra._type_vector(), dtype=bool)
        except Exception as exc:  # noqa: BLE001 - optional metadata
            LOGGER.warning("Predicate type vector unavailable: %s", exc)
            is_spatial = alpha >= 0.5
        np.savez(out_dir / "predicate_bank.npz",
                 names=np.array(vocab, dtype=str), W=W, alpha=alpha,
                 default=np.array(vocab, dtype=str), is_spatial=is_spatial)

        meta = {
            "model_id": args.checkpoint or args.model_id,
            "img_size": img_size,
            "max_boxes": args.max_boxes,
            "vocab_mode": "input",
            "predicates": vocab,
            "text_dim": int(W.shape[1]),
            "inputs": INPUT_NAMES,
            "outputs": OUTPUT_NAMES,
            "output_kind": "logits",
            "score_contract": "sigmoid(a * (pred_logit + pair_logit) + b)",
            "calibration": {"a": float(ra.calib_a), "b": float(ra.calib_b)},
            "opset": args.opset,
            "torch": torch.__version__,
            "exported": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        if not args.skip_validate:
            meta["validation_max_abs_delta"] = validate_onnx(
                onnx_path, wrapper, img_size, args.max_boxes, W, alpha)
        if args.openvino:
            meta["openvino_max_abs_delta"] = export_openvino(
                onnx_path, wrapper, img_size, args.max_boxes, W, alpha)
        with open(out_dir / "relateanything.json", "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        LOGGER.info("Artifacts ready in %s", out_dir)
        return 0
    except AssertionError as exc:
        LOGGER.error("Validation FAILED: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        LOGGER.exception("Export failed: %s", exc)
        return 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
