"""Cosmos-Embed1 remote-code compatibility patch (transformers 5 moved helpers to pytorch_utils)."""
from curate.curator_next import OPTIONAL_STAGES, patch_qformer_source, patch_remote_code

OLD = '''import torch
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.bert.configuration_bert import BertConfig
'''


def test_old_import_is_moved_to_pytorch_utils():
    new = patch_qformer_source(OLD)
    assert "from transformers.modeling_utils import (\n    PreTrainedModel,\n)" in new
    assert "from transformers.pytorch_utils import (\n    apply_chunking_to_forward," in new
    compile(new, "modeling_qformer.py", "exec")
    assert patch_qformer_source(new) is None          # idempotent


def test_patch_remote_code_rewrites_downloaded_copies(tmp_path):
    f = tmp_path / "Cosmos-Embed1-336p" / "modeling_qformer.py"
    f.parent.mkdir()
    f.write_text(OLD)
    assert patch_remote_code(tmp_path) == [f]
    assert "pytorch_utils" in f.read_text()
    assert patch_remote_code(tmp_path) == []
    assert patch_remote_code(tmp_path / "missing") == []


def test_only_cut_and_transcode_are_required():
    assert "01-video-cut" not in OPTIONAL_STAGES and "02-video-transcode" not in OPTIONAL_STAGES
    assert "04-video-embed" in OPTIONAL_STAGES and "03-video-caption" in OPTIONAL_STAGES
