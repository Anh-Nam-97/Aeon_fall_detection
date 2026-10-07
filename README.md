# Aeon_fall_detection

Zero-shot fall detection. YOLO-World v2 (YOLOv8) detects people and the floor,
and RelateAnything scores relations such as `person — lying on — floor`. A
per-person temporal filter then confirms the fall. See
[docs/architecture.md](docs/architecture.md) for the full design.

## Quick start

```bash
bash setup.sh                       # conda env "Aeon_fall_detection" (Python 3.12)
conda activate Aeon_fall_detection
python export_onnx.py --openvino    # once: ONNX + OpenVINO artifacts (~300 MB)
python app.py --mode onnx_cpu       # Edge CPU (i7-1260P)  -> http://127.0.0.1:7860
python app.py --mode torch_gpu --host 0.0.0.0   # RTX 4090 / 5090
```

### Command line

```bash
python pipeline.py image.jpg --mode onnx_cpu --output out.jpg
python pipeline.py clip.mp4 --mode torch_gpu --output out.mp4
```

### Docker (GPU)

```bash
docker compose up -d --build
docker compose --profile export run --rm export   # artifacts for edge devices
```

### Tests and documents

```bash
pytest -q tests
python generate_docx_report.py --out-dir docs --markdown docs/Kien_truc_He_thong_Aeon_Fall_Detection.md
```

## Custom vocabulary (UI textbox)

```
objects: person, floor, bed, sofa
fall: lying on, laying on, resting on, sitting on
upright: standing on, walking on, walking past, standing beside
```

* **TORCH_GPU:** accepts any phrase, because the text encoder runs in-process.
* **ONNX_CPU:** accepts only phrases in `predicate_bank.npz`. Phrases not in the
  bank are reported in the alert log. To add one, re-export with
  `python export_onnx.py --openvino --predicates "collapsed on"`.
* Adding a safe surface such as `bed` or `sofa` suppresses alarms for people
  resting on it.

## Notes

* Python 3.12 is required because RelateAnything declares
  `requires-python >= 3.12`.
* RelateAnything and Ultralytics are AGPL-3.0.

## License

Copyright (C) 2026 Anh-Nam-97

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE)
for the full text.

The project is AGPL-3.0 because it builds on
[Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO-World) and
[RelateAnything](https://github.com/Maelic/RelateAnything), both AGPL-3.0.
If you run a modified version as a network service (for example the Gradio
UI), section 13 requires offering its users the corresponding source code;
the UI header links to this repository for that purpose.
