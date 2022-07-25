#include <torch/torch.h>
#include "benchmark/benchmark.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

static void BM_NaiveIDTransformer_Cold(benchmark::State& state) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag> transformer(1e8, 1e8);
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    global_ids.random_(1e10, 2e10);
    transformer.Transform(
        tcb::span{global_ids.template data_ptr<int64_t>(), global_ids.numel()},
        tcb::span{cache_ids.template data_ptr<int64_t>(), cache_ids.numel()},
        [](int64_t global_id) { return true; },
        [](Tag tag, int64_t global_id, int64_t cache_id) { return tag; },
        [](int64_t global_id, int64_t cache_id) {});
  }
}

BENCHMARK(BM_NaiveIDTransformer_Cold)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

static void BM_NaiveIDTransformer_Hot(benchmark::State& state) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag> transformer(1e8, 1e8);
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    global_ids.random_(1e6, 2e6);
    transformer.Transform(
        tcb::span{global_ids.template data_ptr<int64_t>(), global_ids.numel()},
        tcb::span{cache_ids.template data_ptr<int64_t>(), cache_ids.numel()},
        [](int64_t global_id) { return true; },
        [](Tag tag, int64_t global_id, int64_t cache_id) { return tag; },
        [](int64_t global_id, int64_t cache_id) {});
  }
}

BENCHMARK(BM_NaiveIDTransformer_Hot)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

} // namespace tde::details
