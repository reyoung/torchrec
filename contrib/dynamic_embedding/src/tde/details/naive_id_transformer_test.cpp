#include "gtest/gtest.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

TEST(tde, NaiveThreadedIDTransformer_NoFilter) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, uint8_t> transformer(16, 3);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {3, 4, 3, 5, 4};
  int64_t num_transformed = transformer.Transform(
      tcb::span<const int64_t>{global_ids},
      tcb::span<int64_t>{cache_ids},
      [](int64_t global_id) { return true; },
      [](Tag tag, int64_t global_id, int64_t cache_id) { return tag; },
      [](int64_t global_id, int64_t cache_id) {});
  EXPECT_EQ(5, num_transformed);
  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, NaiveThreadedIDTransformer_Filter) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, uint8_t> transformer(16, 3);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {3, -1, 3, 4, -1};

  auto filter = [](int64_t global_id) { return global_id % 2 == 0; };

  int64_t num_transformed = transformer.Transform(
      tcb::span<const int64_t>{global_ids},
      tcb::span<int64_t>{cache_ids},
      filter,
      [](Tag tag, int64_t global_id, int64_t cache_id) { return tag; },
      [](int64_t global_id, int64_t cache_id) {});
  EXPECT_EQ(3, num_transformed);
  for (size_t i = 0; i < 5; i++) {
    if (filter(global_ids[i]))
      EXPECT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, NaiveThreadedIDTransformer_Full) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, uint8_t> transformer(4, 3);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {3, 4, 5, 6, -1};

  int64_t num_transformed = transformer.Transform(
      tcb::span<const int64_t>{global_ids},
      tcb::span<int64_t>{cache_ids},
      [](int64_t global_id) { return true; },
      [](Tag tag, int64_t global_id, int64_t cache_id) { return tag; },
      [](int64_t global_id, int64_t cache_id) {});
  EXPECT_EQ(4, num_transformed);
  for (size_t i = 0; i < num_transformed; i++) {
    EXPECT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, NaiveThreadedIDTransformer_Evict) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, uint8_t> transformer(4, 0);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];

  int64_t num_transformed = transformer.Transform(
      tcb::span<const int64_t>{global_ids},
      tcb::span<int64_t>{cache_ids},
      [](int64_t global_id) { return true; },
      [](Tag tag, int64_t global_id, int64_t cache_id) { return tag; },
      [](int64_t global_id, int64_t cache_id) {});

  EXPECT_EQ(4, num_transformed);

  const int64_t evict_global_ids[2] = {100, 102};
  transformer.Evict(evict_global_ids);

  const int64_t new_global_ids[4] = {101, 102, 103, 104};
  int64_t new_cache_ids[4];

  num_transformed = transformer.Transform(
      tcb::span<const int64_t>{new_global_ids},
      tcb::span<int64_t>{new_cache_ids},
      [](int64_t global_id) { return true; },
      [](Tag tag, int64_t global_id, int64_t cache_id) { return tag; },
      [](int64_t global_id, int64_t cache_id) {});

  int64_t expected_cache_ids[4] = {1, 0, 3, 2};

  EXPECT_EQ(4, num_transformed);
  for (size_t i = 0; i < num_transformed; i++) {
    EXPECT_EQ(expected_cache_ids[i], new_cache_ids[i]);
  }
}

} // namespace tde::details
