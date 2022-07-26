#include <torch/torch.h>
#include "benchmark/benchmark.h"
#include "tde/details/multithreaded_id_transformer.h"

namespace tde::details {

static void BM_MultiThreadedIDTransformer_Cold(benchmark::State& state) {
  using Tag = int32_t;
  MultiThreadedIDTransformer<Tag> transformer(1e8, state.range(0));
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    global_ids.random_(1e10, 2e10);
    transformer.Transform(
        tcb::span{global_ids.template data_ptr<int64_t>(), global_ids.numel()},
        tcb::span{cache_ids.template data_ptr<int64_t>(), cache_ids.numel()});
  }
}

BENCHMARK(BM_MultiThreadedIDTransformer_Cold)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

static void BM_MultiThreadedIDTransformer_Hot(benchmark::State& state) {
  using Tag = int32_t;
  MultiThreadedIDTransformer<Tag> transformer(1e8, state.range(0));
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    global_ids.random_(1e6, 2e6);
    transformer.Transform(
        tcb::span{global_ids.template data_ptr<int64_t>(), global_ids.numel()},
        tcb::span{cache_ids.template data_ptr<int64_t>(), cache_ids.numel()});
  }
}

BENCHMARK(BM_MultiThreadedIDTransformer_Hot)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

} // namespace tde::details
