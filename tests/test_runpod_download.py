"""runpod/download.py against a fake S3 client (no network)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download", ROOT / "runpod" / "download.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


class FakeS3:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.downloads: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v)} for k, v in sorted(fake.files.items())
                                    if k.startswith(Prefix)]}
        return P()

    def download_file(self, bucket, key, dest):
        self.downloads.append(key)
        Path(dest).write_bytes(self.files[key])


def test_demo_preset_maps_volume_paths_into_local_data_and_is_resumable(tmp_path):
    files = {
        "results/20261111T1/dataset_card.md": b"card",
        "logs/run-1.log": b"log",
        "weld-physical-ai-demo/data/real/cache/worldmodels/assess/ab.json": b"{}",
        "weld-physical-ai-demo/data/real/curated/clips/c1.mp4": b"mp4",
        "weld-physical-ai-demo/data/real/generated/PP5_B02_SAW/meta.json": b"{}",
        "weld-physical-ai-demo/data/real/generated/": b"",                 # directory marker
        "weld-physical-ai-demo/data/real/raw/aimen_saw/big.hdf5": b"x" * 10,  # raw downloads never fetched
        "weld-physical-ai-demo/.venv/bin/python": b"x",                    # never downloaded
    }
    s3 = FakeS3(files)
    n_new, n_same, total = dl.download(s3, "vol", dl.PRESETS["demo"], tmp_path)
    assert n_new == 5 and n_same == 0 and total == len(b"card" + b"log" + b"{}" + b"mp4" + b"{}")
    assert (tmp_path / "runpod_results/20261111T1/dataset_card.md").read_bytes() == b"card"
    assert (tmp_path / "runpod_results/logs/run-1.log").exists()
    assert (tmp_path / "data/real/cache/worldmodels/assess/ab.json").exists()
    assert (tmp_path / "data/real/curated/clips/c1.mp4").read_bytes() == b"mp4"
    assert (tmp_path / "data/real/generated/PP5_B02_SAW/meta.json").exists()
    assert not (tmp_path / "data/real/raw").exists()
    assert not list(tmp_path.rglob("*.part"))
    # second run: nothing new
    s3.downloads.clear()
    assert dl.download(s3, "vol", dl.PRESETS["demo"], tmp_path)[:2] == (0, 5) and s3.downloads == []


def test_endpoint_and_supported_datacenters():
    assert dl.endpoint("US-NE-1") == "https://s3api-us-ne-1.runpod.io/"
    assert "US-NE-1" in dl.S3_DATACENTERS and "US-TX-3" not in dl.S3_DATACENTERS
