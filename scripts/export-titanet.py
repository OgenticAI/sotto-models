#!/usr/bin/env python3
"""Export NVIDIA's TitaNet-large speaker model to the ONNX file Pronto pins.

Pronto tells far-side voices apart with TitaNet-large (crates/pronto-engine/
src/speakers.rs). NVIDIA publishes it only as a NeMo checkpoint, which an app
cannot run without PyTorch, so OgenticAI exports it with NeMo's own exporter:

    uv venv -p 3.11 nemo && VIRTUAL_ENV=nemo uv pip install "nemo_toolkit[asr]==3.0.0" onnxruntime
    nemo/bin/python scripts/export-titanet.py titanet_large.onnx

Source: https://huggingface.co/nvidia/speakerverification_en_titanet_large,
NVIDIA, CC-BY-4.0. The checkpoint is checked against the hash below before it
is exported. With NeMo 3.0.0 and PyTorch 2.15.0 the export is byte for byte
the file the catalog pins (SHA-256 ff56e548…ed310), so anyone can rebuild it
from NVIDIA's checkpoint and compare. Other tool versions give a different file.
"""

import hashlib
import sys
import urllib.request

from nemo.collections.asr.models import EncDecSpeakerLabelModel

SOURCE = (
    "https://huggingface.co/nvidia/speakerverification_en_titanet_large"
    "/resolve/main/speakerverification_en_titanet_large.nemo"
)
SHA256 = "e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(out):
    checkpoint, _ = urllib.request.urlretrieve(SOURCE)
    if sha256(checkpoint) != SHA256:
        sys.exit(f"checkpoint is not NVIDIA's: {sha256(checkpoint)} != {SHA256}")
    model = EncDecSpeakerLabelModel.restore_from(checkpoint, map_location="cpu")
    model.eval()
    model.export(out)
    print(out, sha256(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "titanet_large.onnx")
