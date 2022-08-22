import json
import os
from typing import List

import torch

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = ["TensorList"]


class TensorList:
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensor_list = torch.classes.tde.TensorList()
        for tensor in tensors:
            self.tensor_list.append(tensor)

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, i):
        return self.tensor_list[i]
