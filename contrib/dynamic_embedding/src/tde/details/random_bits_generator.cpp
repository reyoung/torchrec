#include "random_bits_generator.h"

namespace tde::details {

bool BitScanner::IsNextNBitsAllZero(uint16_t& n_bits) {
  if (n_bits == 0) {
    return true;
  }
  if (array_idx_ == size_) [[unlikely]] {
    return true;
  }

  auto val = array[array_idx_];
  val &= static_cast<uint64_t>(-1) >>
      bit_idx; // mask higher bits to zeros if already scan.
  uint16_t remaining_bits =
      static_cast<uint16_t>(sizeof(uint64_t) * 8) - bit_idx;

  bool scan_result;
  if (remaining_bits <= n_bits) {
    scan_result = val == 0;
    n_bits -= remaining_bits;
    bit_idx = 0;
    array_idx_ += 1;
    if (n_bits == 0) {
      return scan_result;
    } else {
      if (!scan_result) { // already false, then just set indices.
        uint16_t next_array_idx = array_idx_ + n_bits / 64;
        if (next_array_idx >= size_) {
          array_idx_ = size_;
        } else {
          array_idx_ = next_array_idx;
          bit_idx = n_bits % 64;
        }
        return false;
      }
      return IsNextNBitsAllZero(n_bits);
    }
  } else {
    if (val == 0) {
      scan_result = true;
      bit_idx += n_bits;
    } else {
      int clz = std::__libcpp_clz(val);
      if (clz >= n_bits + bit_idx) {
        scan_result = true;
        bit_idx += n_bits;
      } else {
        scan_result = false;
        bit_idx += std::min(static_cast<uint16_t>(clz + 1 - bit_idx), n_bits);
      }
    }
    n_bits = 0;
    return scan_result;
  }
}
BitScanner::BitScanner(size_t n) : array(new uint64_t[n]), size_(n) {}

static ThreadPool g_random_pool_(1);
constexpr static size_t k_n_random_elems = 8;

class NElemsRandom {
 public:
  explicit NElemsRandom(std::mt19937_64& engine) : engine_(engine) {}
  void operator()(tcb::span<uint64_t> elems) {
    for (auto& elem : elems) {
      elem = engine_();
    }
  }

 private:
  std::mt19937_64& engine_;
};

RandomBitsGenerator::RandomBitsGenerator()
    : engine_(std::random_device()()),
      scanner_(new BitScanner(k_n_random_elems)) {
  scanner_->ResetArray(NElemsRandom(engine_));
  scanner_future_ =
      GeneratePendingFuture(std::make_unique<BitScanner>(k_n_random_elems));
}
bool RandomBitsGenerator::IsNextNBitsAllZero(uint16_t n_bits) {
  bool ok = scanner_->IsNextNBitsAllZero(n_bits);
  if (n_bits != 0) { // scanner is end.
    auto scanner = scanner_future_.get();
    scanner_future_ = GeneratePendingFuture(std::move(scanner_));
    scanner_ = std::move(scanner);
  }
  if (!ok) {
    return false;
  }
  if (n_bits != 0) [[unlikely]] {
    return IsNextNBitsAllZero(n_bits);
  } else {
    return true;
  }
}

std::future<std::unique_ptr<BitScanner>> RandomBitsGenerator::
    GeneratePendingFuture(std::unique_ptr<BitScanner> scanner) {
  return g_random_pool_.Enqueue(
      [scanner = std::move(scanner), this]() -> std::unique_ptr<BitScanner> {
        scanner->ResetArray(NElemsRandom(engine_));
        return std::move(const_cast<std::unique_ptr<BitScanner>&>(scanner));
      });
}
} // namespace tde::details
