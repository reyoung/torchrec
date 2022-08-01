#pragma once
#include <optional>
#include "lexy/action/parse.hpp"
#include "lexy/callback.hpp"
#include "lexy/dsl.hpp"
#include "lexy/input/string_input.hpp"
#include "lexy_ext/report_error.hpp"

/**
 * A simple URL/extendable parser
 */
namespace tde::details::url_parser {

struct Auth {
  std::string username_;
  std::optional<std::string> password_;
};

struct Url {
  std::optional<Auth> auth_;
  std::string host_;
  std::optional<uint16_t> port_;
  std::optional<std::string> param_;
};

namespace rules {
namespace dsl = lexy::dsl;
using AuthValue = Auth;
using UrlValue = Url;
struct UrlToken {
  static constexpr auto rule = dsl::percent_sign >>
          dsl::integer<uint8_t>(dsl::n_digits<2, dsl::hex>) |
      dsl::capture(dsl::ascii::alpha_digit_underscore);

  static constexpr auto value = lexy::callback<char>([](auto&& v) -> char {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, uint8_t>) {
      return v;
    } else {
      return v[0];
    }
  });
};

struct UrlString {
  static constexpr auto rule = dsl::list(dsl::p<UrlToken>);
  static constexpr auto value = lexy::as_string<std::string>;
};

struct Auth {
  static constexpr auto rule = dsl::p<UrlString> >>
      dsl::opt(dsl::lit_c<':'> >> dsl::p<UrlString>) >> dsl::lit_c<'@'>;
  static constexpr auto value = lexy::construct<AuthValue>;
};
struct Host {
  static constexpr auto rule =
      dsl::list(dsl::p<UrlToken> | dsl::capture(dsl::lit_c<'.'>));
  static constexpr auto value = lexy::as_string<std::string>;
};

struct Url {
  static constexpr auto rule =
      dsl::opt(dsl::lookahead(dsl::lit_c<'@'>, dsl::newline) >> dsl::p<Auth>) +
      dsl::p<Host> +
      dsl::opt(dsl::lit_c<':'> >> dsl::integer<uint16_t>(dsl::digits<>)) +
      dsl::opt(dsl::lit_c<'/'> >> dsl::any);

  static constexpr auto value = lexy::construct<UrlValue>;
};

} // namespace rules

inline std::optional<Url> ParseUrl(std::string_view url_str) {
  // NOTE: cannot using rule = xxx here. It seems a bug of clang or lexy.
  auto input = lexy::string_input(url_str);
  auto parsed = lexy::parse<rules::Url>(input, lexy_ext::report_error);
  if (parsed.has_value()) {
    return parsed.value();
  } else {
    return std::nullopt;
  }
}

} // namespace tde::details::url_parser
