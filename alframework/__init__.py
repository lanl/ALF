"""ALF: the Active Learning Framework."""

# Every ALF task imports this package before it touches the GPU runtime, which reads the visibility masks
# once, so this is where a mask that would hide an AMD worker's GPU is removed. See
# alframework.tools.gpu.normalize_amd_visibility; it changes nothing on NVIDIA nodes or without such a mask.
from alframework.tools.gpu import normalize_amd_visibility as _normalize_amd_visibility

_normalize_amd_visibility()
