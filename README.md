# Sotto models

Models that OgenticAI publishes for Sotto, each with where it came from and how to rebuild it. Sotto downloads them by HTTPS from this repository's releases and checks each against a pinned SHA-256 before loading it.

## NVIDIA TitaNet-large, ONNX

Release [`titanet-large-v1`](https://github.com/OgenticAI/sotto-models/releases/tag/titanet-large-v1), file `titanet_large.onnx`.

NVIDIA's TitaNet-large speaker model, exported to ONNX with NVIDIA NeMo's own exporter so Sotto can tell the voices on a call apart on the device. The weights are NVIDIA's, unchanged: only the file format differs.

- **Source:** [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large), `speakerverification_en_titanet_large.nemo`, SHA-256 `e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3`.
- **Licence:** CC-BY-4.0, as NVIDIA licenses the model.
- **File:** `titanet_large.onnx`, 101,404,593 bytes, SHA-256 `ff56e5482ae211b082e55280d129687e40ee629ad31bd345a1ea9bdd4a2ed310`.
- **Rebuild it:** [`scripts/export-titanet.py`](scripts/export-titanet.py) downloads NVIDIA's checkpoint, checks its hash and exports it. With NeMo 3.0.0 and PyTorch 2.15.0 the result is this file byte for byte.

```bash
uv venv -p 3.11 nemo && VIRTUAL_ENV=nemo uv pip install "nemo_toolkit[asr]==3.0.0" onnxruntime
nemo/bin/python scripts/export-titanet.py titanet_large.onnx
```

| Name | Type | Shape |
| --- | --- | --- |
| `audio_signal` (in) | float32 | batch × 80 × frames: NeMo's log-mel spectrogram, normalised per feature |
| `length` (in) | int64 | batch: valid frames |
| `embs` (out) | float32 | batch × 192: the speaker embedding |
| `logits` (out) | float32 | batch × 16681: training-speaker classes, unused |

TitaNet is by NVIDIA: N. Koluguri, T. Park and B. Ginsburg, "TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions and Global Context", ICASSP 2022. Built with NVIDIA NeMo.
