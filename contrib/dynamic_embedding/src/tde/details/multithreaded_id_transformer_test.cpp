#include "catch2/catch_all.hpp"
#include "tde/details/multithreaded_id_transformer.h"

namespace tde::details {

TEST_CASE("tde_details_MultiThreadedIDTransformer") {
  // Some common code here

  SECTION("aspect 1") {}

  SECTION("aspect 2") {}

  BENCHMARK_ADVANCED("benchmark 1")(Catch::Benchmark::Chronometer meter) {
    // some set up code here

    meter.measure([]() -> int {
      // benchmark part here. This lambda will be run many times.

      // return some value related to benchmark, so the compiler will not
      // optimize whole lambda.
      return 0;
    });
  };
}

} // namespace tde::details
