import os

import torch
import torchrec_dynamic_embedding


__all__ = ["register_memory_io"]


MEMORY_IO_REGISTERED = False


def register_memory_io():
    global MEMORY_IO_REGISTERED
    if not MEMORY_IO_REGISTERED:
        torch.ops.tde.register_io(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memory_io/memory_io.so"
            )
        )
        MEMORY_IO_REGISTERED = True
