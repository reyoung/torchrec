import os
from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import ParameterSharding, ShardingPlan

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


class PS:
    def __init__(
        self,
        table_name: str,
        tensor: Union[torch.Tensor, ShardedTensor],
        ps_config: str,
    ):
        shards = torch.classes.tde.LocalShardList()
        if isinstance(tensor, torch.Tensor):
            shards.append(0, 0, tensor.shape[0], tensor.shape[1], tensor)
            col_size = tensor.shape[1]
        elif isinstance(tensor, ShardedTensor):
            for shard in tensor.local_shards():
                shards.append(
                    shard.metadata.shard_offsets[0],
                    shard.metadata.shard_offsets[1],
                    shard.metadata.shard_sizes[0],
                    shard.metadata.shard_sizes[1],
                    shard.tensor,
                )
                # This assumes all shard have the same column size.
                col_size = shard.tensor.shape[1]
        self._ps = torch.classes.tde.PS(table_name, shards, col_size, ps_config)

    def evict(self, ids_to_evict: torch.Tensor):
        self._ps.evict(ids_to_evict)

    def fetch(self, ids_to_fetch: torch.Tensor):
        self._ps.fetch(ids_to_fetch)


class PSCollection:
    """
    PS for one table.
    """

    def __init__(
        self,
        path: str,
        plan: Dict[str, Tuple[ParameterSharding, Union[torch.Tensor, ShardedTensor]]],
        ps_config: Union[str, Callable[[str], str]],
    ):
        self._path = path
        self._ps_collection = {}
        for table_name, (param_plan, tensor) in plan.items():
            if isinstance(ps_config, str):
                table_config = ps_config
            else:
                table_config = ps_config[table_name]
            self._ps_collection[table_name] = PS(
                f"{path}.{table_name}", tensor, table_config
            )

    def __getitem__(self, table_name):
        return self._ps_collection[table_name]


def get_sharded_modules_recursive(
    module: nn.Module,
    path: str,
    plan: ShardingPlan,
) -> Dict[str, nn.Module]:
    params_plan = plan.get_plan_for_module(path)
    if params_plan:
        return {path: (module, params_plan)}

    res = {}
    for name, child in module.named_children():
        new_path = f"{path}.{name}" if path else name
        res.update(get_sharded_modules_recursive(child, new_path, plan))
    return res


def get_ps(module: DMP, ps_config: Union[str, Callable[[str, str], str]]):
    plan = module.plan

    sharded_modules = get_sharded_modules_recursive(module.module, "", plan)

    ps_list = {}

    for path, (sharded_module, params_plan) in sharded_modules.items():
        state_dict = sharded_module.state_dict()
        tensor_infos = {}
        for key, tensor in state_dict.items():
            # Here we use the fact that state_dict will be shape of
            # `embeddings.xxx.weight` or `embeddingbags.xxx.weight`
            if len(key.split(".")) <= 1 or key.split(".")[1] not in params_plan:
                continue
            table_name = key.split(".")[1]
            param_plan = params_plan.pop(table_name)
            tensor_infos[table_name] = (param_plan, tensor)

        assert (
            len(params_plan) == 0
        ), f"There are sharded param not found, leaving: {params_plan}."

        if isinstance(ps_config, str):
            collection_config = ps_config
        else:
            collection_config = lambda table_name: ps_config(path, table_name)

        ps_list[path] = PSCollection(path, tensor_infos, collection_config)
    return ps_list
