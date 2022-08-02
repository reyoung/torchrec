#include "gtest/gtest.h"
#include "tde/details/redis_io_v1.h"

namespace tde::details::redis_v1 {

TEST(TDE, redis_v1_Option) {
  auto opt = Option::Parse("192.168.3.1:3948/?db=3&&num_threads=2");
  ASSERT_EQ(opt.host_, "192.168.3.1");
  ASSERT_EQ(opt.port_, 3948);
  ASSERT_EQ(opt.db_, 3);
  ASSERT_EQ(opt.num_io_threads_, 2);
  ASSERT_TRUE(opt.prefix_.empty());
}

TEST(TDE, redis_v1_push_pull) {
  auto opt = Option::Parse("127.0.0.1:6379");
  RedisV1 redis(opt);

  constexpr static int64_t global_ids[] = {1, 3, 4};
  constexpr static uint32_t os_ids[] = {0};
  constexpr static float params[] = {1, 2, 3, 4, 5, 9, 8, 1};
  constexpr static uint64_t offsets[] = {0, 2, 4, 6, 8};

  IOPushParameter push{
      .table_name_ = "table",
      .num_cols_ = 0,
      .num_global_ids_ = sizeof(global_ids) / sizeof(global_ids[0]),
      .global_ids_ = global_ids,
      .col_ids_ = nullptr,
      .num_optimizer_stats_ = sizeof(os_ids) / sizeof(os_ids[0]),
      .num_offsets_ = sizeof(offsets) / sizeof(offsets[0]),
      .offsets_ = offsets,
      .data_ = params,
      .on_complete_context_ = nullptr,
      .on_push_complete = +[](void* ctx) {},
  };
}
} // namespace tde::details::redis_v1
