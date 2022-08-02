#include "tde/details/redis_io_v1.h"
#include <variant>
#include <vector>
#include "lexy/callback.hpp"
#include "lexy/dsl.hpp"
#include "tcb/span.hpp"
#include "tde/details/url.h"

namespace tde::details::redis_v1 {

struct NumThreadsOpt {
  uint32_t num_threads_;
};
struct DBOpt {
  uint32_t db_;
};
struct PrefixOpt {
  std::string prefix_;
};

using OptVar = std::variant<NumThreadsOpt, DBOpt, PrefixOpt>;

namespace option_rules {
namespace dsl = lexy::dsl;
struct NumThreads {
  constexpr static auto rule = LEXY_LIT("num_threads=") >>
      dsl::integer<uint32_t>(dsl::digits<>.no_leading_zero());
  constexpr static auto value = lexy::construct<NumThreadsOpt>;
};

struct DB {
  constexpr static auto rule = LEXY_LIT("db=") >>
      dsl::integer<uint32_t>(dsl::digits<>);
  constexpr static auto value = lexy::construct<DBOpt>;
};

struct Prefix {
  constexpr static auto rule = LEXY_LIT("prefix=") >>
      dsl::capture(dsl::token(dsl::identifier(
          dsl::ascii::alpha_underscore,
          dsl::ascii::alpha_digit_underscore)));
  constexpr static auto value =
      lexy::callback<PrefixOpt>([](auto&& str) -> PrefixOpt {
        return PrefixOpt{std::string(str.data(), str.size())};
      });
};

struct Option {
  constexpr static auto rule = dsl::p<NumThreads> | dsl::p<DB> | dsl::p<Prefix>;
  constexpr static auto value = lexy::construct<OptVar>;
};

struct Param {
  constexpr static auto rule =
      dsl::list(dsl::p<Option>, dsl::sep(LEXY_LIT("&&")));

  constexpr static auto value = lexy::as_list<std::vector<OptVar>>;
};

} // namespace option_rules

Option::Option(std::string_view config_str) {
  auto url = url_parser::ParseUrl(config_str);
  if (url.auth_.has_value()) {
    username_ = std::move(url.auth_->username_);
    if (url.auth_->password_.has_value()) {
      password_ = std::move(url.auth_->password_.value());
    }
  }

  host_ = std::move(url.host_);

  if (url.port_.has_value()) {
    port_ = url.port_.value();
  }

  if (url.param_.has_value()) {
    std::ostringstream err_oss_;
    url_parser::ErrorCollector collector{err_oss_};

    auto result = lexy::parse<option_rules::Param>(
        lexy::string_input(url.param_.value()), collector);
    TORCH_CHECK(result.has_value(), "parse param error %s", err_oss_.str());

    for (auto&& opt_var : result.value()) {
      std::visit(
          [this](auto&& opt) {
            using T = std::decay_t<decltype(opt)>;

            if constexpr (std::is_same_v<T, NumThreadsOpt>) {
              num_io_threads_ = opt.num_threads_;
            } else if constexpr (std::is_same_v<T, DBOpt>) {
              db_ = opt.db_;
            } else {
              prefix_ = std::move(opt.prefix_);
            }
          },
          opt_var);
    }
  }
}

RedisV1::RedisV1(Option opt) : opt_(std::move(opt)) {
  TORCH_CHECK(opt_.num_io_threads_ != 0, "num_io_threads must not be empty");
  TORCH_CHECK(
      opt_.heart_beat_interval_ms_ != 0,
      "heart beat interval must not be zero.");
  for (size_t i = 0; i < opt_.num_io_threads_; ++i) {
    StartThread();
  }
}

void RedisV1::StartThread() {
  auto connection = Connect();
  HeartBeat(connection);

  io_threads_.emplace_back(
      [connection = std::move(connection), this]() mutable {
        std::chrono::milliseconds heart_beat(opt_.heart_beat_interval_ms_);
        while (true) {
          MoveOnlyFunction<void(redis::ContextPtr&)> todo;
          bool heartbeat_timeout;
          {
            std::unique_lock<std::mutex> lock(this->jobs_mutex_);
            heartbeat_timeout = !jobs_not_empty_.wait_for(
                lock, heart_beat, [this] { return !jobs_.empty(); });
            if (!heartbeat_timeout) {
              todo = std::move(jobs_.front());
              jobs_.pop_front();
            }
          }

          if (heartbeat_timeout) {
            HeartBeat(connection);
            continue;
          }

          if (!todo) {
            break;
          }
          todo(connection);
        }
      });
}

void RedisV1::HeartBeat(redis::ContextPtr& connection) {
  for (uint32_t retry = 0; retry < opt_.retry_limit_; ++retry) {
    try {
      auto reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(
          redisCommand(connection.get(), "PING")));
      TORCH_CHECK(
          reply && reply->type == REDIS_REPLY_STRING,
          "Ping should return string");
      auto rsp = std::string_view(reply->str, reply->len);
      TORCH_CHECK(rsp == "PONG", "ping/pong error");
    } catch (...) {
      // reconnect if heart beat error
      connection = Connect();
    }
  }
}

redis::ContextPtr RedisV1::Connect() const {
  redis::ContextPtr connection;
  if (opt_.timeout_ms_ == 0) {
    connection =
        redis::ContextPtr(redisConnect(opt_.host_.c_str(), opt_.port_));
  } else {
    struct timeval interval {};
    interval.tv_sec = opt_.timeout_ms_ / 1000;
    interval.tv_usec = opt_.timeout_ms_ % 1000 * 1000;
    connection = redis::ContextPtr(
        redisConnectWithTimeout(opt_.host_.c_str(), opt_.port_, interval));
  }
  TORCH_CHECK(
      !connection->err,
      "connect to %s:%d error occurred %s",
      opt_.host_,
      opt_.port_,
      connection->errstr);

  if (!opt_.password_.empty()) {
    redis::ReplyPtr reply;
    if (opt_.username_.empty()) {
      reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(
          redisCommand(connection.get(), "AUTH %s", opt_.password_.c_str())));
    } else {
      reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(redisCommand(
          connection.get(),
          "AUTH %s %s",
          opt_.username_.c_str(),
          opt_.password_.c_str())));
    }
    CheckReplyString("auth error", connection, reply, "OK");
  }

  if (opt_.db_ != 0) {
    auto reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(
        redisCommand(connection.get(), "SELECT %d", opt_.db_)));
    CheckReplyString("select db error", connection, reply, "OK");
  }

  return connection;
}

RedisV1::~RedisV1() {
  for (uint32_t i = 0; i < opt_.num_io_threads_; ++i) {
    jobs_.emplace_back();
  }
  for (auto& th : io_threads_) {
    th.join();
  }
}

void RedisV1::CheckReplyString(
    std::string_view label,
    redis::ContextPtr& connection,
    redis::ReplyPtr& reply,
    std::string_view expect) const {
  TORCH_CHECK(
      !connection->err,
      "%s error: %s. from redis://%s:%d",
      label,
      connection->errstr,
      opt_.host_,
      opt_.port_);
  TORCH_CHECK(
      reply->type == REDIS_REPLY_STRING,
      "%s reply should be string, but type %d. from redis://%s:%d",
      label,
      reply->type,
      opt_.host_,
      opt_.port_);

  auto actual = std::string_view(reply->str, reply->len);
  TORCH_CHECK(
      actual == expect,
      "%s reply not as expect. Expect %s, Actual %s, from redis://%s:%d",
      label,
      expect,
      actual,
      opt_.host_,
      opt_.port_);
}

static uint32_t CalculateChunkSizeByGlobalIDs(
    uint32_t chunk_size,
    uint32_t num_cols,
    uint32_t num_os) {
  static constexpr uint32_t low = 1;
  return std::max(
      chunk_size / std::max(num_cols, low) / std::max(num_os, low), low);
}

struct RedisV1PullContext {
  std::atomic<uint32_t> num_complete_ids_{0};
  uint32_t chunk_size_;
  std::string table_name_;
  std::vector<int64_t> global_ids_;
  std::vector<int64_t> col_ids_;
  uint32_t num_optimizer_stats_;
  void* on_complete_context_;
  void (*on_global_id_fetched_)(
      void* ctx,
      uint32_t gid_offset,
      uint32_t optimizer_state,
      void* data,
      uint32_t data_len);
  void (*on_all_fetched_)(void* ctx);

  explicit RedisV1PullContext(uint32_t chunk_size, IOPullParameter param)
      : chunk_size_(CalculateChunkSizeByGlobalIDs(
            chunk_size,
            param.num_cols_,
            param.num_optimizer_stats_)),
        table_name_(param.table_name_),
        global_ids_(
            param.global_ids_,
            param.global_ids_ + param.num_global_ids_),
        num_optimizer_stats_(param.num_optimizer_stats_),
        on_complete_context_(param.on_complete_context_),
        on_global_id_fetched_(param.on_global_id_fetched_),
        on_all_fetched_(param.on_all_fetched_) {
    if (param.num_cols_ == 0) {
      col_ids_.emplace_back(-1);
    } else {
      col_ids_ = std::vector<int64_t>(
          param.col_ids_, param.col_ids_ + param.num_cols_);
    }
  }
};

void RedisV1::Pull(IOPullParameter param) {
  auto* fetch_param = new RedisV1PullContext(opt_.chunk_size_, param);
  {
    std::lock_guard<std::mutex> guard(this->jobs_mutex_);
    for (uint32_t i = 0; i < param.num_global_ids_;
         i += fetch_param->chunk_size_) {
      jobs_.emplace_back([i, fetch_param, this](redis::ContextPtr& connection) {
        DoFetch(i, fetch_param, connection);
      });
    }
  }
  jobs_not_empty_.notify_all();
}

void RedisV1::DoFetch(
    uint32_t gid_offset,
    void* fetch_param_void,
    redis::ContextPtr& connection) const {
  auto& fetch_param = *reinterpret_cast<RedisV1PullContext*>(fetch_param_void);

  uint32_t end = std::min(
      gid_offset + fetch_param.chunk_size_,
      static_cast<uint32_t>(fetch_param.global_ids_.size()));

  auto loop = [&](auto&& callback) {
    for (uint32_t i = gid_offset; i < end; ++i) {
      int64_t gid = fetch_param.global_ids_[i];
      for (auto& col_id : fetch_param.col_ids_) {
        for (uint32_t os_id = 0; os_id < fetch_param.num_optimizer_stats_;
             ++os_id) {
          callback(
              i * fetch_param.col_ids_.size() + col_id, gid, col_id, os_id);
        }
      }
    }
  };

  loop([&](uint32_t offset, int64_t gid, uint32_t col_id, uint32_t os_id) {
    redisAppendCommand(
        connection.get(),
        "GET %s_table_%s_gid_%d_cid_%d_osid_%d",
        opt_.prefix_.c_str(),
        fetch_param.table_name_.c_str(),
        gid,
        col_id,
        os_id);
  });

  void* reply;
  loop([&](uint32_t offset, int64_t gid, uint32_t col_id, uint32_t os_id) {
    int status = redisGetReply(connection.get(), &reply);
    TORCH_CHECK(
        status != REDIS_ERR,
        "get reply error: %s, from redis %s, %d",
        connection->errstr,
        opt_.host_,
        opt_.port_);
    auto reply_ptr = redis::ReplyPtr(reinterpret_cast<redisReply*>(reply));

    if (reply_ptr->type == REDIS_REPLY_NIL) {
      fetch_param.on_global_id_fetched_(
          fetch_param.on_complete_context_, offset, os_id, nullptr, 0);
    } else {
      fetch_param.on_global_id_fetched_(
          fetch_param.on_complete_context_,
          offset,
          os_id,
          reply_ptr->str,
          reply_ptr->len);
    }
  });

  uint32_t n = end - gid_offset;
  uint32_t target = fetch_param.global_ids_.size();

  if (fetch_param.num_complete_ids_.fetch_add(n) + n ==
      target) { // last fetch complete
    fetch_param.on_all_fetched_(fetch_param.on_complete_context_);
    delete &fetch_param;
  }
}

struct RedisV1PushContext {
  std::atomic<uint32_t> num_complete_ids_{0};
  uint32_t chunk_size_;
  std::string table_name_;
  tcb::span<const int64_t> global_ids_;
  std::vector<int64_t> col_ids_;
  tcb::span<const uint32_t> os_ids_;
  tcb::span<const uint64_t> offsets_;
  const void* data_;
  void* on_complete_context_;
  void (*on_push_complete_)(void*);

  RedisV1PushContext(uint32_t chunk_size, IOPushParameter param)
      : chunk_size_(CalculateChunkSizeByGlobalIDs(
            chunk_size,
            param.num_cols_,
            param.num_optimizer_stats_)),
        table_name_(param.table_name_),
        global_ids_(param.global_ids_, param.num_global_ids_),
        os_ids_(param.optimizer_stats_ids_, param.num_optimizer_stats_),
        offsets_(param.offsets_, param.num_offsets_),
        data_(param.data_),
        on_complete_context_(param.on_complete_context_),
        on_push_complete_(param.on_push_complete) {
    if (param.num_cols_ != 0) {
      col_ids_ = std::vector<int64_t>(
          param.col_ids_, param.col_ids_ + param.num_cols_);
    } else {
      col_ids_.emplace_back(-1);
    }
  }
};

void RedisV1::Push(IOPushParameter param) {
  auto* ctx = new RedisV1PushContext(opt_.chunk_size_, param);
  {
    std::lock_guard<std::mutex> guard(this->jobs_mutex_);
    for (uint32_t i = 0; i < param.num_global_ids_; i += ctx->chunk_size_) {
      jobs_.emplace_back([i, ctx, this](redis::ContextPtr& connection) {
        DoPush(i, ctx, connection);
      });
    }
  }
  jobs_not_empty_.notify_all();
}
void RedisV1::DoPush(
    uint32_t gid_offset,
    void* push_ctx_ptr,
    redis::ContextPtr& connection) const {
  auto& push_ctx = *reinterpret_cast<RedisV1PushContext*>(push_ctx_ptr);

  uint32_t end = gid_offset + push_ctx.chunk_size_;
  if (end > push_ctx.global_ids_.size()) {
    end = push_ctx.global_ids_.size();
  }

  auto loop = [&](auto&& callback) {
    for (uint32_t i = gid_offset; i < end; ++i) {
      int64_t gid = push_ctx.global_ids_[i];
      for (uint32_t j = 0; j < push_ctx.col_ids_.size(); ++j) {
        int64_t cid = push_ctx.col_ids_[j];
        for (uint32_t k = 0; k < push_ctx.os_ids_.size(); ++k) {
          uint32_t os_id = push_ctx.os_ids_[k];

          uint32_t offset = k + j * push_ctx.os_ids_.size() +
              i * push_ctx.col_ids_.size() * push_ctx.os_ids_.size();
          callback(offset, gid, cid, os_id);
        }
      }
    }
  };

  loop([&](uint32_t o, int64_t gid, int64_t cid, uint32_t os_id) {
    uint64_t beg = push_ctx.offsets_[o];
    uint64_t end = push_ctx.offsets_[o + 1];

    redisAppendCommand(
        connection.get(),
        "SET %s_table_%s_gid_%d_cid_%d_osid_%d %b",
        opt_.prefix_.c_str(),
        push_ctx.table_name_.c_str(),
        gid,
        cid,
        os_id,
        reinterpret_cast<uint8_t*>(push_ctx.data_) + beg,
        static_cast<size_t>(end - beg));
  });

  void* replay_ptr;
  loop([&](...) {
    int status = redisGetReply(connection.get(), &replay_ptr);
    TORCH_CHECK(
        status != REDIS_ERR,
        "get reply error: %s, from redis %s, %d",
        connection->errstr,
        opt_.host_,
        opt_.port_);
    redis::ReplyPtr reply(reinterpret_cast<redisReply*>(replay_ptr));
    CheckReplyString("reply should be ok", connection, reply, "OK");
  });

  uint32_t n = end - gid_offset;
  uint32_t target = push_ctx.global_ids_.size();
  if (push_ctx.num_complete_ids_.fetch_add(n) + n == target) {
    push_ctx.on_push_complete_(push_ctx.on_complete_context_);
    delete &push_ctx;
  }
}

} // namespace tde::details::redis_v1
