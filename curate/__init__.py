"""Stage 3 of WeldWorld: turn exported weld clips into a curated, captioned dataset.

Backends (``python -m curate.run --backend ...``):
  curator-next    NVIDIA Cosmos Curator Next recipe chain (GPU, S3/MinIO input)
  curator-legacy  Cosmos Curator legacy ``video-pipeline split`` (GPU, local input)
  lite            pure-Python CPU implementation of the same stages

Every backend ends by writing ``<out>/manifest.jsonl`` (one row per clip, see
``curate.manifest.MANIFEST_KEYS``) and ``<out>/dataset_card.md``.
"""

__all__ = ["MANIFEST_KEYS"]

from curate.manifest import MANIFEST_KEYS  # noqa: E402
