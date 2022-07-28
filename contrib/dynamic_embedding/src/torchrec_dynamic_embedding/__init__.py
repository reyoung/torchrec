import json
import os

import torch

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


class IDTransformer:
    def __init__(self, num_embedding, **kwargs):
        if "lxu_strategy" not in kwargs:
            kwargs["lxu_strategy"] = {"type": "mixed_lru_lfu"}
        if "id_transformer" not in kwargs:
            kwargs["id_transformer"] = {"type": "naive"}
        config = json.dumps(kwargs)
        self._transformer = torch.classes.tde.IDTransformer(num_embedding, config)

    def transform(self, global_ids, cache_ids):
        return self._transformer.transform(global_ids, cache_ids)

    def get_ids_to_fetch(self):
        return self._transformer.get_ids_to_fetch()
