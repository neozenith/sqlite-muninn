"""Shared GLiNER2 model cache — loads once per process.

Both PhaseNER and PhaseRE share this when backend='gliner2'.
The cache ensures the model is loaded at most once even when
both phases run sequentially in the same process.

Offline loading: snapshot_download returns the cached local path;
GLiNER2.from_pretrained(local_dir) short-circuits to local file access
via os.path.isdir() — no HF_HUB_OFFLINE patching required.
"""

from __future__ import annotations

from gliner2 import GLiNER2
from huggingface_hub import snapshot_download

DEFAULT_GLINER2_MODEL = "fastino/gliner2-base-v1"

_cache: dict[str, GLiNER2] = {}


def get_gliner2(model_name: str = DEFAULT_GLINER2_MODEL) -> GLiNER2:
    """Return a cached GLiNER2 instance, downloading on first call.

    Subsequent calls with the same model_name return the cached instance
    with no additional memory or disk I/O.
    """
    if model_name not in _cache:
        local_path = snapshot_download(model_name, local_files_only=True)
        _cache[model_name] = GLiNER2.from_pretrained(local_path)
    return _cache[model_name]
