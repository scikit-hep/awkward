// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_DATETIME_UTIL_H_
#define AWKWARD_DATETIME_UTIL_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <array>

#include "awkward/common.h"

#ifndef _MSC_VER
  #include "dlfcn.h"
#endif

namespace awkward {

  namespace util {
    /// @brief Convert a format string into a datetime units string.
    const std::string
    format_to_units(const std::string& format);

    /// @brief Convert a datetime units string into a format string.
    const std::string
    units_to_format(dtype dt, const std::string& units, int64_t step);

    template <class T>
    struct NameValuePair {
      using value_type = T;
      const T value;
      const char* const name;
      const int64_t scale_up;
      const int64_t scale_down;
    };

    template <class Mapping, class V>
    std::string name(Mapping a, V value) {
      auto pos = std::find_if(
        std::begin(a), std::end(a), [&value](const typename Mapping::value_type& t) { return ((V)t.value == value); });
      if (pos != std::end(a)) {
        return pos->name;
      }
      return std::begin(a)->name;
    }

    template <class Mapping>
    typename Mapping::value_type::value_type value(Mapping a, const std::string& name) {
      auto unit_name(name);
      std::string chars = "[]1234567890";
      unit_name.erase(remove_if(unit_name.begin(), unit_name.end(),
                        [&chars](const char &c) {
                            return chars.find(c) != std::string::npos;
                        }),
                        unit_name.end());
      auto pos = std::find_if(
        std::begin(a), std::end(a), [&](const typename Mapping::value_type& t) {
          return (unit_name == t.name);
        });
      if (pos != std::end(a)) {
        return pos->value;
      }
      return std::begin(a)->value;
    }

    /// @brief Valid datetime units.
    /// Different units of two datetime type Arrays will be normalized to
    /// the smallest unit (== a larger enum value).
    enum class datetime_units {
      unknown = -1, // unknown
      Y = 1,   // year
      M = 2,   // month
      W = 3,   // week
      D = 4,   // day
      h = 5,   // hour
      m = 6,   // minute
      s = 7,   // second
      ms = 8,  // millisecond
      us = 9,  // microsecond: note, 'μs' string is not supported
      ns = 10, // nanosecond
      ps = 11, // picosecond
      fs = 12, // femtosecond
      as = 13, // attosecond
    };

    // One calendar common year has 365 days: 31536000 seconds
    // One calendar leap year has 366 days: 31622400 seconds
    const std::array<const NameValuePair<datetime_units>, 14> units_map {
      {{datetime_units::unknown, "unknown", 1, 1},
      {datetime_units::Y, "Y", 31556952, 1}, // seconds in average Gregorian year
      {datetime_units::M, "M", 2629746, 1},
      {datetime_units::W, "W", 604800, 1},
      {datetime_units::D, "D", 86400, 1},
      {datetime_units::h, "h", 3600, 1},
      {datetime_units::m, "m", 60, 1},
      {datetime_units::s, "s", 1, 1},
      {datetime_units::ms, "ms", 1, 1000},
      {datetime_units::us, "us", 1, 1000000},
      {datetime_units::ns, "ns", 1, 1000000000},
      {datetime_units::ps, "ps", 1, 1000000000000},
      {datetime_units::fs, "fs", 1, 1000000000000000},
      {datetime_units::as, "as", 1, 1000000000000000000}}
    };

    std::tuple<std::string, int64_t> datetime_data(const std::string& format);

    double scale_from_units(const std::string& format, uint64_t index);
  }
}

#endif // AWKWARD_DATETIME_UTIL_H_
