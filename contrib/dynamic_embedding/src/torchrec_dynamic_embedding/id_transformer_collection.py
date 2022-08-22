from typing import List, Union

import torch
from torchrec import EmbeddingBagConfig, EmbeddingConfig, KeyedJaggedTensor

from .id_transformer import IDTransformer, TensorList
from .ps import PSCollection


__all__ = ["IDTransformerCollection"]


class IDTransformerCollection:
    def __init__(
        self,
        tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
        eviction_config=None,
        transform_config=None,
        ps_collection: PSCollection = None,
    ):
        self._configs = tables
        self._ps_collection = ps_collection

        self._transformers = []
        self._table_names = []
        feature_names = set()
        for config in tables:
            if config.name in self._table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            if not config.feature_names:
                config.feature_names = [config.name]
            self._table_names.append(config.name)
            for feature_name in config.feature_names:
                if feature_name in feature_names:
                    raise ValueError(f"Shared feature not allowed yet.")
                feature_names
            self._transformers.append(
                IDTransformer(
                    num_embedding=config.num_embeddings,
                    eviction_config=eviction_config,
                    transform_config=transform_config,
                )
            )
        self._feature_names: List[List[str]] = [
            config.feature_names for config in tables
        ]

    def transform(self, global_features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        global_values = global_features.values()
        cache_values = torch.empty_like(global_values)

        global_feature_indices = {
            feature_name: i for i, feature_name in enumerate(global_features.keys())
        }
        offset_per_key = global_features.offset_per_key()

        for i, transformer in enumerate(self._transformers):
            feature_names = self._feature_names[i]
            feature_indices = [
                global_feature_indices[feature_name] for feature_name in feature_names
            ]
            global_ids = [
                global_values[offset_per_key[idx] : offset_per_key[idx + 1]]
                for idx in feature_indices
            ]
            cache_ids = [
                cache_values[offset_per_key[idx] : offset_per_key[idx + 1]]
                for idx in feature_indices
            ]

            result = transformer.transform(
                TensorList(global_ids), TensorList(cache_ids)
            )
            if self._ps_collection is not None:
                table_name = self._table_names[i]
                ps = self._ps_collection[table_name]
                if result.ids_to_fetch is not None:
                    ps.fetch(result.ids_to_fetch)
                if not result.success:
                    # TODO(zilinzhu): make this configurable
                    ids_to_evict = transformer.evict(transformer._num_embedding // 2)
                    ps.evict(ids_to_evict)

                    # retry after eviction.
                    result = transformer.transform(
                        TensorList(global_ids), TensorList(cache_ids)
                    )
                    if not result.success:
                        raise RuntimeError(
                            "Failed to transform global ids after eviction. "
                            f"Maybe the num_embedding of table {table_name} is too small?"
                        )
                    if result.ids_to_fetch is not None:
                        ps.fetch(result.ids_to_fetch)

        cache_values = KeyedJaggedTensor(
            keys=global_features.keys(),
            values=cache_values,
            lengths=global_features.lengths(),
            weights=global_features.weights_or_none(),
        )
        return cache_values
