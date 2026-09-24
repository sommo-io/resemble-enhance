"""Import-only stand-in for deepspeed.

resemble_enhance imports deepspeed at module level (training engine, distributed helpers), but
inference never calls it. This stub satisfies those imports so the image can skip the real
package, its CUDA build toolchain and the devel base image. Anything that actually tries to use
deepspeed (i.e. training) fails loudly.
"""


def _unavailable(*_args, **_kwargs):
    raise RuntimeError("deepspeed is stubbed out in this image; training is not supported")


init_distributed = _unavailable


class DeepSpeedConfig:
    def __init__(self, *args, **kwargs):
        _unavailable()
