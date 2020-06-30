// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <set>

#include "rapidjson/document.h"

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/cpu-kernels/reducers.h"
#include "awkward/cpu-kernels/sorting.h"

#include "awkward/util.h"
#include "awkward/Identities.h"

namespace rj = rapidjson;

namespace awkward {
  namespace util {
    void
    handle_error(const struct Error& err,
                 const std::string& classname,
                 const Identities* identities) {
      if(err.pass_through == true) {
        throw std::invalid_argument(err.str);
      }
      else {
        if (err.str != nullptr) {
          std::stringstream out;
          out << "in " << classname;
          if (err.identity != kSliceNone && identities != nullptr) {
            if (0 <= err.identity && err.identity < identities->length()) {
              out << " with identity ["
                  << identities->identity_at(err.identity) << "]";
            } else {
              out << " with invalid identity";
            }
          }
          if (err.attempt != kSliceNone) {
            out << " attempting to get " << err.attempt;
          }
          out << ", " << err.str;
          throw std::invalid_argument(out.str());
        }
      }
    }

    template<typename T>
    IndexOf<T> make_starts(const IndexOf<T> &offsets) {
      return IndexOf<T>(offsets.ptr(),
                        offsets.offset(),
                        offsets.length() - 1,
                        offsets.ptr_lib());
    }

    template<typename T>
    IndexOf<T> make_stops(const IndexOf<T> &offsets) {
      return IndexOf<T>(offsets.ptr(),
                        offsets.offset() + 1,
                        offsets.length() - 1,
                        offsets.ptr_lib());
    }

    template IndexOf<int32_t> make_starts(const IndexOf<int32_t> &offsets);

    template IndexOf<uint32_t> make_starts(const IndexOf<uint32_t> &offsets);

    template IndexOf<int64_t> make_starts(const IndexOf<int64_t> &offsets);

    template IndexOf<int32_t> make_stops(const IndexOf<int32_t> &offsets);

    template IndexOf<uint32_t> make_stops(const IndexOf<uint32_t> &offsets);

    template IndexOf<int64_t> make_stops(const IndexOf<int64_t> &offsets);

    std::string
    quote(const std::string &x, bool doublequote) {
      // TODO: escape characters, possibly using RapidJSON.
      if (doublequote) {
        return std::string("\"") + x + std::string("\"");
      } else {
        return std::string("'") + x + std::string("'");
      }
    }

    RecordLookupPtr
    init_recordlookup(int64_t numfields) {
      RecordLookupPtr out = std::make_shared<RecordLookup>();
      for (int64_t i = 0; i < numfields; i++) {
        out.get()->push_back(std::to_string(i));
      }
      return out;
    }

    int64_t
    fieldindex(const RecordLookupPtr &recordlookup,
               const std::string &key,
               int64_t numfields) {
      int64_t out = -1;
      if (recordlookup.get() != nullptr) {
        for (size_t i = 0; i < recordlookup.get()->size(); i++) {
          if (recordlookup.get()->at(i) == key) {
            out = (int64_t) i;
            break;
          }
        }
      }
      if (out == -1) {
        try {
          out = (int64_t) std::stoi(key);
        }
        catch (std::invalid_argument err) {
          throw std::invalid_argument(
            std::string("key ") + quote(key, true)
            + std::string(" does not exist (not in record)"));
        }
        if (!(0 <= out && out < numfields)) {
          throw std::invalid_argument(
            std::string("key interpreted as fieldindex ") + key
            + std::string(" for records with only " + std::to_string(numfields)
                          + std::string(" fields")));
        }
      }
      return out;
    }

    const std::string
    key(const RecordLookupPtr &recordlookup,
        int64_t fieldindex,
        int64_t numfields) {
      if (fieldindex >= numfields) {
        throw std::invalid_argument(
          std::string("fieldindex ") + std::to_string(fieldindex)
          + std::string(" for records with only " + std::to_string(numfields)
                        + std::string(" fields")));
      }
      if (recordlookup.get() != nullptr) {
        return recordlookup.get()->at((size_t) fieldindex);
      } else {
        return std::to_string(fieldindex);
      }
    }

    bool
    haskey(const RecordLookupPtr &recordlookup,
           const std::string &key,
           int64_t numfields) {
      try {
        fieldindex(recordlookup, key, numfields);
      }
      catch (std::invalid_argument err) {
        return false;
      }
      return true;
    }

    const std::vector<std::string>
    keys(const RecordLookupPtr &recordlookup, int64_t numfields) {
      std::vector<std::string> out;
      if (recordlookup.get() != nullptr) {
        out.insert(out.end(),
                   recordlookup.get()->begin(),
                   recordlookup.get()->end());
      } else {
        int64_t cols = numfields;
        for (int64_t j = 0; j < cols; j++) {
          out.push_back(std::to_string(j));
        }
      }
      return out;
    }

    bool
    parameter_equals(const Parameters &parameters,
                     const std::string &key,
                     const std::string &value) {
      auto item = parameters.find(key);
      std::string myvalue;
      if (item == parameters.end()) {
        myvalue = "null";
      } else {
        myvalue = item->second;
      }
      rj::Document mine;
      rj::Document yours;
      mine.Parse<rj::kParseNanAndInfFlag>(myvalue.c_str());
      yours.Parse<rj::kParseNanAndInfFlag>(value.c_str());
      return mine == yours;
    }

    bool
    parameters_equal(const Parameters &self, const Parameters &other) {
      std::set<std::string> checked;
      for (auto pair : self) {
        if (!parameter_equals(other, pair.first, pair.second)) {
          return false;
        }
        checked.insert(pair.first);
      }
      for (auto pair : other) {
        if (checked.find(pair.first) == checked.end()) {
          if (!parameter_equals(self, pair.first, pair.second)) {
            return false;
          }
        }
      }
      return true;
    }

    bool
    parameter_isstring(const Parameters &parameters, const std::string &key) {
      auto item = parameters.find(key);
      if (item == parameters.end()) {
        return false;
      }
      rj::Document mine;
      mine.Parse<rj::kParseNanAndInfFlag>(item->second.c_str());
      return mine.IsString();
    }

    bool
    parameter_isname(const Parameters &parameters, const std::string &key) {
      auto item = parameters.find(key);
      if (item == parameters.end()) {
        return false;
      }
      rj::Document mine;
      mine.Parse<rj::kParseNanAndInfFlag>(item->second.c_str());
      if (!mine.IsString()) {
        return false;
      }
      std::string value = mine.GetString();
      if (value.empty()) {
        return false;
      }
      if (!((value[0] >= 'a' && value[0] <= 'z') ||
            (value[0] >= 'A' && value[0] <= 'Z') ||
            (value[0] == '_'))) {
        return false;
      }
      for (size_t i = 1; i < value.length(); i++) {
        if (!((value[i] >= 'a' && value[i] <= 'z') ||
              (value[i] >= 'A' && value[i] <= 'Z') ||
              (value[i] >= '0' && value[i] <= '9') ||
              (value[i] == '_'))) {
          return false;
        }
      }
      return true;
    }

    const std::string
    parameter_asstring(const Parameters &parameters, const std::string &key) {
      auto item = parameters.find(key);
      if (item == parameters.end()) {
        throw std::runtime_error("parameter is null");
      }
      rj::Document mine;
      mine.Parse<rj::kParseNanAndInfFlag>(item->second.c_str());
      if (!mine.IsString()) {
        throw std::runtime_error("parameter is not a string");
      }
      return mine.GetString();
    }

    std::string
    gettypestr(const Parameters &parameters, const TypeStrs &typestrs) {
      auto item = parameters.find("__record__");
      if (item != parameters.end()) {
        std::string source = item->second;
        rj::Document recname;
        recname.Parse<rj::kParseNanAndInfFlag>(source.c_str());
        if (recname.IsString()) {
          std::string name = recname.GetString();
          for (auto pair : typestrs) {
            if (pair.first == name) {
              return pair.second;
            }
          }
        }
      }
      item = parameters.find("__array__");
      if (item != parameters.end()) {
        std::string source = item->second;
        rj::Document recname;
        recname.Parse<rj::kParseNanAndInfFlag>(source.c_str());
        if (recname.IsString()) {
          std::string name = recname.GetString();
          for (auto pair : typestrs) {
            if (pair.first == name) {
              return pair.second;
            }
          }
        }
      }
      return std::string();
    }
  }
}
