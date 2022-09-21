// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/Content.cpp", line)

#include <sstream>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "awkward/io/json.h"

#include "awkward/kernels.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"


#include "awkward/Content.h"

namespace rj = rapidjson;

namespace awkward {
  ////////// Form

  FormPtr
  Form::fromnumpy(char kind,
                  int64_t itemsize,
                  const std::vector<int64_t>& inner_shape) {
    switch (kind) {
    case 'b':
      if (itemsize == 1) {
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 1,
                 util::dtype_to_format(util::dtype::boolean),
                 util::dtype::boolean);
      }
      else {
        throw std::invalid_argument(
          std::string("cannot convert NumPy bool dtype with itemsize ")
          + std::to_string(itemsize) + std::string(" into a NumpyForm")
          + FILENAME(__LINE__));
      }
    case 'i':
      switch (itemsize) {
      case 1:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 1,
                 util::dtype_to_format(util::dtype::int8),
                 util::dtype::int8);
      case 2:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 2,
                 util::dtype_to_format(util::dtype::int16),
                 util::dtype::int16);
      case 4:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 4,
                 util::dtype_to_format(util::dtype::int32),
                 util::dtype::int32);
      case 8:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 8,
                 util::dtype_to_format(util::dtype::int64),
                 util::dtype::int64);
      default:
        throw std::invalid_argument(
          std::string("cannot convert NumPy int dtype with itemsize ")
          + std::to_string(itemsize) + std::string(" into a NumpyForm")
          + FILENAME(__LINE__));
      }
    case 'u':
      switch (itemsize) {
      case 1:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 1,
                 util::dtype_to_format(util::dtype::uint8),
                 util::dtype::uint8);
      case 2:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 2,
                 util::dtype_to_format(util::dtype::uint16),
                 util::dtype::uint16);
      case 4:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 4,
                 util::dtype_to_format(util::dtype::uint32),
                 util::dtype::uint32);
      case 8:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 8,
                 util::dtype_to_format(util::dtype::uint64),
                 util::dtype::uint64);
      default:
        throw std::invalid_argument(
          std::string("cannot convert NumPy int dtype with itemsize ")
          + std::to_string(itemsize) + std::string(" into a NumpyForm")
          + FILENAME(__LINE__));
      }
    case 'f':
      switch (itemsize) {
      case 2:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 2,
                 util::dtype_to_format(util::dtype::float16),
                 util::dtype::float16);
      case 4:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 4,
                 util::dtype_to_format(util::dtype::float32),
                 util::dtype::float32);
      case 8:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 8,
                 util::dtype_to_format(util::dtype::float64),
                 util::dtype::float64);
      case 16:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 16,
                 util::dtype_to_format(util::dtype::float128),
                 util::dtype::float128);
      default:
        throw std::invalid_argument(
          std::string("cannot convert NumPy floating-point dtype with itemsize ")
          + std::to_string(itemsize) + std::string(" into a NumpyForm")
          + FILENAME(__LINE__));
      }

    case 'c':
      switch (itemsize) {
      case 8:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 8,
                 util::dtype_to_format(util::dtype::complex64),
                 util::dtype::complex64);
      case 16:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 16,
                 util::dtype_to_format(util::dtype::complex128),
                 util::dtype::complex128);
      case 32:
        return std::make_shared<NumpyForm>(
                 false,
                 util::Parameters(),
                 FormKey(nullptr),
                 inner_shape,
                 32,
                 util::dtype_to_format(util::dtype::complex256),
                 util::dtype::complex256);
      default:
        throw std::invalid_argument(
          std::string("cannot convert NumPy complex dtype with itemsize ")
          + std::to_string(itemsize) + std::string(" into a NumpyForm")
          + FILENAME(__LINE__));
      }

    // case 'M': handle datetime64
    // case 'm': handle timedelta64

    default:
      throw std::invalid_argument(
        std::string("cannot convert NumPy dtype with kind ")
        + std::string(1, kind) + std::string(" into a NumpyForm")
        + FILENAME(__LINE__));
    }
  }

  template <typename JSON>
  FormPtr
  fromjson_part(const JSON& json) {
    if (json.IsString()) {
      std::string primitive(json.GetString());
      if (primitive.find("datetime64") == 0) {
        return std::make_shared<NumpyForm>(false,
                                           util::Parameters(),
                                           FormKey(nullptr),
                                           std::vector<int64_t>(),
                                           8,
                                           "M8" + primitive.substr(10, std::string::npos),
                                           util::dtype::datetime64);
      }
      else if (primitive.find("timedelta64") == 0) {
        return std::make_shared<NumpyForm>(false,
                                           util::Parameters(),
                                           FormKey(nullptr),
                                           std::vector<int64_t>(),
                                           8,
                                           "m8" + primitive.substr(11, std::string::npos),
                                           util::dtype::timedelta64);
      }
      else {
        util::dtype dtype = util::name_to_dtype(primitive);
        int64_t itemsize = util::dtype_to_itemsize(dtype);
        if (dtype != util::dtype::NOT_PRIMITIVE) {
          return std::make_shared<NumpyForm>(false,
                                             util::Parameters(),
                                             FormKey(nullptr),
                                             std::vector<int64_t>(),
                                             itemsize,
                                             util::dtype_to_format(dtype),
                                             dtype);
        }
      }
    }

    if (json.IsObject()  &&
        json.HasMember("class")  &&
        json["class"].IsString()) {

      bool h = false;
      if (json.HasMember("has_identifier")) {
        if (json["has_identifier"].IsBool()) {
          h = json["has_identifier"].GetBool();
        }
        else {
          throw std::invalid_argument(
            std::string("'has_identifier' must be boolean") + FILENAME(__LINE__));
        }
      }
      else if (json.HasMember("has_identities")) {
        if (json["has_identities"].IsBool()) {
          h = json["has_identities"].GetBool();
        }
        else {
          throw std::invalid_argument(
            std::string("'has_identities' must be boolean") + FILENAME(__LINE__));
        }
      }

      util::Parameters p;
      if (json.HasMember("parameters")) {
        if (json["parameters"].IsObject()) {
          for (auto& pair : json["parameters"].GetObject()) {
            rj::StringBuffer stringbuffer;
            rj::Writer<rj::StringBuffer> writer(stringbuffer);
            pair.value.Accept(writer);
            p[pair.name.GetString()] = stringbuffer.GetString();
          }
        }
        else {
          throw std::invalid_argument(
            std::string("'parameters' must be a JSON object") + FILENAME(__LINE__));
        }
      }

      FormKey f(nullptr);
      if (json.HasMember("form_key")) {
        if (json["form_key"].IsNull()) { }
        else if (json["form_key"].IsString()) {
          f = std::make_shared<std::string>(json["form_key"].GetString());
        }
        else {
          throw std::invalid_argument(
            std::string("'form_key' must be null or a string") + FILENAME(__LINE__));
        }
      }

      bool isgen;
      bool is64;
      bool isU32;
      bool is32;
      std::string cls = json["class"].GetString();

      if (cls == std::string("NumpyArray")) {
        std::string format;
        int64_t itemsize;
        util::dtype dtype;
        if (json.HasMember("primitive")  &&  json["primitive"].IsString()) {
          FormPtr tmp = fromjson_part(json["primitive"]);
          NumpyForm* raw = dynamic_cast<NumpyForm*>(tmp.get());
          format = raw->format();
          itemsize = raw->itemsize();
          dtype = util::format_to_dtype(format, itemsize);
          if ((raw->dtype() == util::dtype::datetime64 ||
               raw->dtype() == util::dtype::timedelta64)
              && json.HasMember("format")  &&  json["format"].IsString()) {
            format = json["format"].GetString();
          }
        }
        else if (json.HasMember("format")  &&  json["format"].IsString()  &&
                 json.HasMember("itemsize")  &&  json["itemsize"].IsInt()) {
          format = json["format"].GetString();
          itemsize = json["itemsize"].GetInt64();
          dtype = util::format_to_dtype(format, itemsize);
        }
        else {
          throw std::invalid_argument(
            std::string("NumpyForm must have a 'primitive' field or 'format' "
                        "and 'itemsize'") + FILENAME(__LINE__));
        }
        std::vector<int64_t> s;
        if (json.HasMember("inner_shape")  &&  json["inner_shape"].IsArray()) {
          for (auto& x : json["inner_shape"].GetArray()) {
            if (x.IsInt()) {
              s.push_back(x.GetInt64());
            }
            else {
              throw std::invalid_argument(
                std::string("NumpyForm 'inner_shape' must only contain integers")
                + FILENAME(__LINE__));
            }
          }
        }
        return std::make_shared<NumpyForm>(h, p, f, s, itemsize, format, dtype);
      }

      if (cls == std::string("RecordArray")) {
        util::RecordLookupPtr recordlookup(nullptr);
        std::vector<FormPtr> contents;
        if (json.HasMember("contents")  &&  json["contents"].IsArray()) {
          for (auto& x : json["contents"].GetArray()) {
            contents.push_back(fromjson_part(x));
          }
        }
        else if (json.HasMember("contents")  &&  json["contents"].IsObject()) {
          recordlookup = std::make_shared<util::RecordLookup>();
          for (auto& pair : json["contents"].GetObject()) {
            recordlookup.get()->push_back(pair.name.GetString());
            contents.push_back(fromjson_part(pair.value));
          }
        }
        else {
          throw std::invalid_argument(
            std::string("RecordArray 'contents' must be a JSON list or a "
                        "JSON object") + FILENAME(__LINE__));
        }
        return std::make_shared<RecordForm>(h, p, f, recordlookup, contents);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((isgen = (cls == std::string("ListOffsetArray")))  ||
          (is64  = (cls == std::string("ListOffsetArray64")))  ||
          (isU32 = (cls == std::string("ListOffsetArrayU32")))  ||
          (is32  = (cls == std::string("ListOffsetArray32")))) {
        Index::Form offsets = (is64  ? Index::Form::i64 :
                               isU32 ? Index::Form::u32 :
                               is32  ? Index::Form::i32 :
                                       Index::Form::kNumIndexForm);
        if (json.HasMember("offsets")  &&  json["offsets"].IsString()) {
          Index::Form tmp = Index::str2form(json["offsets"].GetString());
          if (offsets != Index::Form::kNumIndexForm  &&  offsets != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'offsets' type: ")
              + json["offsets"].GetString() + FILENAME(__LINE__));
          }
          offsets = tmp;
        }
        if (offsets == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing an 'offsets' specification")
            + FILENAME(__LINE__));
        }
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        return std::make_shared<ListOffsetForm>(h, p, f, offsets, content);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((isgen = (cls == std::string("ListArray")))  ||
          (is64  = (cls == std::string("ListArray64")))  ||
          (isU32 = (cls == std::string("ListArrayU32")))  ||
          (is32  = (cls == std::string("ListArray32")))) {
        Index::Form starts = (is64  ? Index::Form::i64 :
                              isU32 ? Index::Form::u32 :
                              is32  ? Index::Form::i32 :
                                      Index::Form::kNumIndexForm);
        Index::Form stops  = (is64  ? Index::Form::i64 :
                              isU32 ? Index::Form::u32 :
                              is32  ? Index::Form::i32 :
                                      Index::Form::kNumIndexForm);
        if (json.HasMember("starts")  &&  json["starts"].IsString()) {
          Index::Form tmp = Index::str2form(json["starts"].GetString());
          if (starts != Index::Form::kNumIndexForm  &&  starts != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'starts' type: ")
              + json["starts"].GetString() + FILENAME(__LINE__));
          }
          starts = tmp;
        }
        if (json.HasMember("stops")  &&  json["stops"].IsString()) {
          Index::Form tmp = Index::str2form(json["stops"].GetString());
          if (stops != Index::Form::kNumIndexForm  &&  stops != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'stops' type: ")
              + json["stops"].GetString() + FILENAME(__LINE__));
          }
          stops = tmp;
        }
        if (starts == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'starts' specification")
            + FILENAME(__LINE__));
        }
        if (stops == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'stops' specification")
            + FILENAME(__LINE__));
        }
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'")
            + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        return std::make_shared<ListForm>(h, p, f, starts, stops, content);
      }

      if (cls == std::string("RegularArray")) {
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        if (!json.HasMember("size")  ||  !json["size"].IsInt()) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'size'") + FILENAME(__LINE__));
        }
        int64_t size = json["size"].GetInt64();
        return std::make_shared<RegularForm>(h, p, f, content, size);
      }

      isgen = is64 = is32 = false;
      if ((isgen = (cls == std::string("IndexedOptionArray")))  ||
          (is64  = (cls == std::string("IndexedOptionArray64")))  ||
          (is32  = (cls == std::string("IndexedOptionArray32")))) {
        Index::Form index = (is64  ? Index::Form::i64 :
                             is32  ? Index::Form::i32 :
                                     Index::Form::kNumIndexForm);
        if (json.HasMember("index")  &&  json["index"].IsString()) {
          Index::Form tmp = Index::str2form(json["index"].GetString());
          if (index != Index::Form::kNumIndexForm  &&  index != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'index' type: ")
              + json["index"].GetString() + FILENAME(__LINE__));
          }
          index = tmp;
        }
        if (index == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing an 'index' specification")
            + FILENAME(__LINE__));
        }
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        return std::make_shared<IndexedOptionForm>(h, p, f, index, content);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((isgen = (cls == std::string("IndexedArray")))  ||
          (is64  = (cls == std::string("IndexedArray64")))  ||
          (isU32 = (cls == std::string("IndexedArrayU32")))  ||
          (is32  = (cls == std::string("IndexedArray32")))) {
        Index::Form index = (is64  ? Index::Form::i64 :
                             isU32 ? Index::Form::u32 :
                             is32  ? Index::Form::i32 :
                                     Index::Form::kNumIndexForm);
        if (json.HasMember("index")  &&  json["index"].IsString()) {
          Index::Form tmp = Index::str2form(json["index"].GetString());
          if (index != Index::Form::kNumIndexForm  &&  index != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'index' type: ")
              + json["index"].GetString() + FILENAME(__LINE__));
          }
          index = tmp;
        }
        if (index == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing an 'index' specification")
             + FILENAME(__LINE__));
        }
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        return std::make_shared<IndexedForm>(h, p, f, index, content);
      }

      if (cls == std::string("ByteMaskedArray")) {
        Index::Form mask = (is64  ? Index::Form::i64 :
                            isU32 ? Index::Form::u32 :
                            is32  ? Index::Form::i32 :
                                    Index::Form::kNumIndexForm);
        if (json.HasMember("mask")  &&  json["mask"].IsString()) {
          Index::Form tmp = Index::str2form(json["mask"].GetString());
          if (mask != Index::Form::kNumIndexForm  &&  mask != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'mask' type: ")
              + json["mask"].GetString() + FILENAME(__LINE__));
          }
          mask = tmp;
        }
        if (mask == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'mask' specification")
             + FILENAME(__LINE__));
        }
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        if (!json.HasMember("valid_when")  ||  !json["valid_when"].IsBool()) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'valid_when'") + FILENAME(__LINE__));
        }
        bool valid_when = json["valid_when"].GetBool();
        return std::make_shared<ByteMaskedForm>(h, p, f, mask, content,
                                                valid_when);
      }

      if (cls == std::string("BitMaskedArray")) {
        Index::Form mask = (is64  ? Index::Form::i64 :
                            isU32 ? Index::Form::u32 :
                            is32  ? Index::Form::i32 :
                                    Index::Form::kNumIndexForm);
        if (json.HasMember("mask")  &&  json["mask"].IsString()) {
          Index::Form tmp = Index::str2form(json["mask"].GetString());
          if (mask != Index::Form::kNumIndexForm  &&  mask != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'mask' type: ")
              + json["mask"].GetString() + FILENAME(__LINE__));
          }
          mask = tmp;
        }
        if (mask == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'mask' specification")
            + FILENAME(__LINE__));
        }
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'")
            + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        if (!json.HasMember("valid_when")  ||  !json["valid_when"].IsBool()) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'valid_when'")
            + FILENAME(__LINE__));
        }
        bool valid_when = json["valid_when"].GetBool();
        if (!json.HasMember("lsb_order")  ||  !json["lsb_order"].IsBool()) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'lsb_order'")
            + FILENAME(__LINE__));
        }
        bool lsb_order = json["lsb_order"].GetBool();
        return std::make_shared<BitMaskedForm>(h, p, f, mask, content,
                                               valid_when, lsb_order);
      }

      if (cls == std::string("UnmaskedArray")) {
        if (!json.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        FormPtr content = fromjson_part(json["content"]);
        return std::make_shared<UnmaskedForm>(h, p, f, content);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((isgen = (cls == std::string("UnionArray")))  ||
          (is64  = (cls == std::string("UnionArray8_64")))  ||
          (isU32 = (cls == std::string("UnionArray8_U32")))  ||
          (is32  = (cls == std::string("UnionArray8_32")))) {
        Index::Form tags = (is64  ? Index::Form::i8 :
                            isU32 ? Index::Form::i8 :
                            is32  ? Index::Form::i8 :
                                    Index::Form::kNumIndexForm);
        if (json.HasMember("tags")  &&  json["tags"].IsString()) {
          Index::Form tmp = Index::str2form(json["tags"].GetString());
          if (tags != Index::Form::kNumIndexForm  &&  tags != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'tags' type: ")
              + json["tags"].GetString() + FILENAME(__LINE__));
          }
          tags = tmp;
        }
        Index::Form index = (is64  ? Index::Form::i64 :
                             isU32 ? Index::Form::u32 :
                             is32  ? Index::Form::i32 :
                                     Index::Form::kNumIndexForm);
        if (json.HasMember("index")  &&  json["index"].IsString()) {
          Index::Form tmp = Index::str2form(json["index"].GetString());
          if (index != Index::Form::kNumIndexForm  &&  index != tmp) {
            throw std::invalid_argument(
              cls + std::string(" has conflicting 'index' type: ")
              + json["index"].GetString() + FILENAME(__LINE__));
          }
          index = tmp;
        }
        if (tags == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'tags' specification")
            + FILENAME(__LINE__));
        }
        if (index == Index::Form::kNumIndexForm) {
          throw std::invalid_argument(
            cls + std::string(" is missing an 'index' specification")
            + FILENAME(__LINE__));
        }
        std::vector<FormPtr> contents;
        if (json.HasMember("contents")  &&  json["contents"].IsArray()) {
          for (auto& x : json["contents"].GetArray()) {
            contents.push_back(fromjson_part(x));
          }
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" 'contents' must be a JSON list")
            + FILENAME(__LINE__));
        }
        return std::make_shared<UnionForm>(h, p, f, tags, index, contents);
      }

      if (cls == std::string("EmptyArray")) {
        return std::make_shared<EmptyForm>(h, p, f);
      }

    }

    rj::StringBuffer stringbuffer;
    rj::PrettyWriter<rj::StringBuffer> writer(stringbuffer);
    json.Accept(writer);
    throw std::invalid_argument(
            std::string("JSON cannot be recognized as a Form:\n\n")
            + stringbuffer.GetString() + FILENAME(__LINE__));
  }

  FormPtr
  Form::fromjson(const std::string& data) {
    rj::Document doc;
    doc.Parse<rj::kParseNanAndInfFlag>(data.c_str());
    return fromjson_part(doc);
  }

  Form::Form(bool has_identities,
             const util::Parameters& parameters,
             const FormKey& form_key)
      : has_identities_(has_identities)
      , parameters_(parameters)
      , form_key_(form_key) { }

  const std::string
  Form::tostring() const {
    return tojson(true, false);
  }

  const std::string
  Form::tojson(bool pretty, bool verbose) const {
    if (pretty) {
      ToJsonPrettyString builder(-1);
      tojson_part(builder, verbose);
      return builder.tostring();
    }
    else {
      ToJsonString builder(-1);
      tojson_part(builder, verbose);
      return builder.tostring();
    }
  }

  bool
  Form::has_identities() const {
    return has_identities_;
  }

  const util::Parameters
  Form::parameters() const {
    return parameters_;
  }

  const std::string
  Form::parameter(const std::string& key) const {
    auto item = parameters_.find(key);
    if (item == parameters_.end()) {
      return "null";
    }
    return item->second;
  }

  bool
  Form::parameter_equals(const std::string& key,
                         const std::string& value) const {
    return util::parameter_equals(parameters_, key, value);
  }

  const FormKey
  Form::form_key() const {
    return form_key_;
  }

  bool
  Form::form_key_equals(const FormKey& other_form_key) const {
    if (form_key_.get() == nullptr  &&  other_form_key.get() == nullptr) {
      return true;
    }
    else if (form_key_.get() != nullptr  &&
             other_form_key.get() != nullptr  &&
             *form_key_.get() == *(other_form_key.get())) {
      return true;
    }
    else {
      return false;
    }
  }

  void
  Form::identities_tojson(ToJson& builder, bool verbose) const {
    if (verbose  ||  has_identities_) {
      builder.field("has_identities");
      builder.boolean(has_identities_);
    }
  }

  void
  Form::parameters_tojson(ToJson& builder, bool verbose) const {
    if (verbose  ||  !parameters_.empty()) {
      builder.field("parameters");
      builder.beginrecord();
      for (auto pair : parameters_) {
        builder.field(pair.first.c_str());
        builder.json(pair.second.c_str());
      }
      builder.endrecord();
    }
  }

  void
  Form::form_key_tojson(ToJson& builder, bool verbose) const {
    if (form_key_.get() != nullptr) {
      builder.field("form_key");
      builder.string(*form_key_.get());
    }
    else if (verbose) {
      builder.field("form_key");
      builder.null();
    }
  }

  const FormPtr
  Form::getitem_range() const {
    return shallow_copy();
  }
}
