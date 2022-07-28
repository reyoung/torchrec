#include <torch/torch.h>
#include "benchmark/benchmark.h"
#include "tde/details/id_transformer_variant.h"

namespace tde::details {

static void BM_IDTransformerVariant_Naive(benchmark::State& state) {
  using Tag = int32_t;
  IDTransformer transformer(
      LXUStrategy(nlohmann::json::parse(R"(
{
  "type": "mixed_lru_lfu"
}
)")),
      1e8,
      nlohmann::json::parse(R"(
{
  "type": "naive"
}
)"));
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    state.PauseTiming();
    global_ids.random_(state.range(0), state.range(1));
    state.ResumeTiming();
    transformer.Transform(
        tcb::span{
            global_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        tcb::span{
            cache_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())});
  }
}

BENCHMARK(BM_IDTransformerVariant_Naive)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"rand_from", "rand_to"})
    ->Args({static_cast<long long>(1e10), static_cast<long long>(2e10)})
    ->Args({static_cast<long long>(1e6), static_cast<long long>(2e6)});

static void BM_IDTransformerVariant_MultiThreaded(benchmark::State& state) {
  using Tag = int32_t;
  IDTransformer transformer(
      LXUStrategy(nlohmann::json::parse(R"(
{
  "type": "mixed_lru_lfu"
}
)")),
      1e8,
      nlohmann::json::parse(
          R"(
{
  "type": "thread",
  "underlying": {
    "type": "naive"
  },
  "num_threads": )" +
          std::to_string(state.range(2)) + R"(
}
)"));
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    state.PauseTiming();
    global_ids.random_(state.range(0), state.range(1));
    state.ResumeTiming();
    transformer.Transform(
        tcb::span{
            global_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        tcb::span{
            cache_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())});
  }
}

BENCHMARK(BM_IDTransformerVariant_MultiThreaded)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"rand_from", "rand_to"})
    ->Args({static_cast<long long>(1e10), static_cast<long long>(2e10), 1})
    ->Args({static_cast<long long>(1e10), static_cast<long long>(2e10), 2})
    ->Args({static_cast<long long>(1e10), static_cast<long long>(2e10), 4})
    ->Args({static_cast<long long>(1e6), static_cast<long long>(2e6), 1})
    ->Args({static_cast<long long>(1e6), static_cast<long long>(2e6), 2})
    ->Args({static_cast<long long>(1e6), static_cast<long long>(2e6), 4});

} // namespace tde::details
