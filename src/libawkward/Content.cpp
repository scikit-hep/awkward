// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/Content.cpp", line)

#include <sstream>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"

#include "awkward/kernels/operations.h"
#include "awkward/kernels/reducers.h"
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
#include "awkward/array/VirtualArray.h"
#include "awkward/type/ArrayType.h"

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
      util::dtype dtype = util::name_to_dtype(json.GetString());
      int64_t itemsize = util::dtype_to_itemsize(dtype);
      std::string format = util::dtype_to_format(dtype);
      if (dtype != util::dtype::NOT_PRIMITIVE) {
        return std::make_shared<NumpyForm>(false,
                                           util::Parameters(),
                                           FormKey(nullptr),
                                           std::vector<int64_t>(),
                                           itemsize,
                                           format,
                                           dtype);
      }
    }

    if (json.IsObject()  &&
        json.HasMember("class")  &&
        json["class"].IsString()) {

      bool h = false;
      if (json.HasMember("has_identities")) {
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

      if (cls == std::string("VirtualArray")) {
        if (!json.HasMember("form")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'form'") + FILENAME(__LINE__));
        }
        FormPtr form(nullptr);
        if (!json["form"].IsNull()) {
          form = fromjson_part(json["form"]);
        }
        if (!json.HasMember("has_length")  ||  !json["has_length"].IsBool()) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'has_length'") + FILENAME(__LINE__));
        }
        bool has_length = json["has_length"].GetBool();
        return std::make_shared<VirtualForm>(h, p, f, form, has_length);
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

  ////////// Content

  Content::Content(const IdentitiesPtr& identities,
                   const util::Parameters& parameters)
      : identities_(identities)
      , parameters_(parameters) { }

  bool
  Content::isscalar() const {
    return false;
  }

  const IdentitiesPtr
  Content::identities() const {
    return identities_;
  }

  const std::string
  Content::tostring() const {
    return tostring_part("", "", "");
  }

  const std::string
  Content::tojson(bool pretty, int64_t maxdecimals) const {
    if (pretty) {
      ToJsonPrettyString builder(maxdecimals);
      tojson_part(builder, true);
      return builder.tostring();
    }
    else {
      ToJsonString builder(maxdecimals);
      tojson_part(builder, true);
      return builder.tostring();
    }
  }

  void
  Content::tojson(FILE* destination,
                  bool pretty,
                  int64_t maxdecimals,
                  int64_t buffersize) const {
    if (pretty) {
      ToJsonPrettyFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder, true);
      builder.endlist();
    }
    else {
      ToJsonFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder, true);
      builder.endlist();
    }
  }

  int64_t
  Content::nbytes() const {
    // FIXME: this is only accurate if all subintervals of allocated arrays are
    // nested (which is likely, but not guaranteed). In general, it's <= the
    // correct nbytes.
    std::map<size_t, int64_t> largest;
    nbytes_part(largest);
    int64_t out = 0;
    for (auto pair : largest) {
      out += pair.second;
    }
    return out;
  }

  const std::string
  Content::purelist_parameter(const std::string& key) const {
    return form(false).get()->purelist_parameter(key);
  }

  bool
  Content::purelist_isregular() const {
    return form(true).get()->purelist_isregular();
  }

  int64_t
  Content::purelist_depth() const {
    return form(true).get()->purelist_depth();
  }

  const std::pair<int64_t, int64_t>
  Content::minmax_depth() const {
    return form(true).get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  Content::branch_depth() const {
    return form(true).get()->branch_depth();
  }

  const ContentPtr
  Content::reduce(const Reducer& reducer,
                  int64_t axis,
                  bool mask,
                  bool keepdims) const {
    int64_t negaxis = -axis;
    std::pair<bool, int64_t> branchdepth = branch_depth();
    bool branch = branchdepth.first;
    int64_t depth = branchdepth.second;

    if (branch) {
      if (negaxis <= 0) {
        throw std::invalid_argument(
          std::string("cannot use non-negative axis on a nested list structure "
                      "of variable depth (negative axis counts from the leaves "
                      "of the tree; non-negative from the root)")
          + FILENAME(__LINE__));
      }
      if (negaxis > depth) {
        throw std::invalid_argument(
          std::string("cannot use axis=") + std::to_string(axis)
          + std::string(" on a nested list structure that splits into "
                        "different depths, the minimum of which is depth=")
          + std::to_string(depth) + std::string(" from the leaves")
          + FILENAME(__LINE__));
      }
    }
    else {
      if (negaxis <= 0) {
        negaxis += depth;
      }
      if (!(0 < negaxis  &&  negaxis <= depth)) {
        throw std::invalid_argument(
          std::string("axis=") + std::to_string(axis)
          + std::string(" exceeds the depth of the nested list structure "
                        "(which is ")
          + std::to_string(depth) + std::string(")") + FILENAME(__LINE__));
      }
    }

    Index64 starts(1);
    starts.setitem_at_nowrap(0, 0);

    Index64 shifts(0);

    Index64 parents(length());
    struct Error err = kernel::content_reduce_zeroparents_64(
      kernel::lib::cpu,   // DERIVE
      parents.data(),
      length());
    util::handle_error(err, classname(), identities_.get());

    ContentPtr next = reduce_next(reducer,
                                  negaxis,
                                  starts,
                                  shifts,
                                  parents,
                                  1,
                                  mask,
                                  keepdims);
    return next.get()->getitem_at_nowrap(0);
  }

  const ContentPtr
  Content::argsort(int64_t axis, bool ascending, bool stable) const {
    int64_t negaxis = -axis;
    std::pair<bool, int64_t> branchdepth = branch_depth();
    bool branch = branchdepth.first;
    int64_t depth = branchdepth.second;

    if (branch) {
      if (negaxis <= 0) {
        throw std::invalid_argument(
          std::string("cannot use non-negative axis on a nested list structure "
                      "of variable depth (negative axis counts from the leaves "
                      "of the tree; non-negative from the root)")
          + FILENAME(__LINE__));
      }
      if (negaxis > depth) {
        throw std::invalid_argument(
          std::string("cannot use axis=") + std::to_string(axis)
          + std::string(" on a nested list structure that splits into different "
                        "depths, the minimum of which is depth=")
          + std::to_string(depth) + std::string(" from the leaves")
          + FILENAME(__LINE__));
      }
    }
    else {
      if (negaxis <= 0) {
        negaxis += depth;
      }
      if (!(0 < negaxis  &&  negaxis <= depth)) {
        throw std::invalid_argument(
          std::string("axis=") +
        std::to_string(axis) + std::string(" exceeds the depth of the nested "
                                           "list structure (which is ")
          + std::to_string(depth) + std::string(")") + FILENAME(__LINE__));
      }
    }

    Index64 starts(1);
    starts.setitem_at_nowrap(0, 0);
    Index64 parents(length());
    struct Error err = kernel::content_reduce_zeroparents_64(
      kernel::lib::cpu,   // DERIVE
      parents.data(),
      length());
    util::handle_error(err, classname(), identities_.get());
    ContentPtr next = argsort_next(negaxis,
                                   starts,
                                   parents,
                                   1,
                                   ascending,
                                   stable,
                                   true);

    return next.get()->getitem_at_nowrap(0);
  }

  const ContentPtr
  Content::sort(int64_t axis,
                bool ascending,
                bool stable) const {
    int64_t negaxis = -axis;
    std::pair<bool, int64_t> branchdepth = branch_depth();
    bool branch = branchdepth.first;
    int64_t depth = branchdepth.second;

    if (branch) {
      if (negaxis <= 0) {
        throw std::invalid_argument(
          std::string("cannot use non-negative axis on a nested list structure "
                      "of variable depth (negative axis counts from the leaves "
                      "of the tree; non-negative from the root)")
          + FILENAME(__LINE__));
      }
      if (negaxis > depth) {
        throw std::invalid_argument(
          std::string("cannot use axis=") + std::to_string(axis)
          + std::string(" on a nested list structure that splits into "
                        "different depths, the minimum of which is depth=")
          + std::to_string(depth) + std::string(" from the leaves")
          + FILENAME(__LINE__));
      }
    }
    else {
      if (negaxis <= 0) {
        negaxis += depth;
      }
      if (!(0 < negaxis  &&  negaxis <= depth)) {
        throw std::invalid_argument(
          std::string("axis=") + std::to_string(axis)
          + std::string(" exceeds the depth of the nested list structure "
                        "(which is ")
          + std::to_string(depth) + std::string(")")
          + FILENAME(__LINE__));
      }
    }

    Index64 starts(1);
    starts.setitem_at_nowrap(0, 0);

    Index64 parents(length());
    struct Error err = kernel::content_reduce_zeroparents_64(
      kernel::lib::cpu,   // DERIVE
      parents.data(),
      length());
    util::handle_error(err, classname(), identities_.get());

    ContentPtr next = sort_next(negaxis,
                                starts,
                                parents,
                                1,
                                ascending,
                                stable,
                                true);

    return next.get()->getitem_at_nowrap(0);
  }

  const util::Parameters
  Content::parameters() const {
    return parameters_;
  }

  void
  Content::setparameters(const util::Parameters& parameters) {
    parameters_ = parameters;
  }

  const std::string
  Content::parameter(const std::string& key) const {
    auto item = parameters_.find(key);
    if (item == parameters_.end()) {
      return "null";
    }
    return item->second;
  }

  void
  Content::setparameter(const std::string& key, const std::string& value) {
    if (value == std::string("null")) {
      parameters_.erase(key);
    }
    else {
      parameters_[key] = value;
    }
  }

  bool
  Content::parameter_equals(const std::string& key,
                            const std::string& value) const {
    return util::parameter_equals(parameters_, key, value);
  }

  bool
  Content::parameters_equal(const util::Parameters& other) const {
    return util::parameters_equal(parameters_, other);
  }

  bool
  Content::parameter_isstring(const std::string& key) const {
    return util::parameter_isstring(parameters_, key);
  }

  bool
  Content::parameter_isname(const std::string& key) const {
    return util::parameter_isname(parameters_, key);
  }

  const std::string
  Content::parameter_asstring(const std::string& key) const {
    return util::parameter_asstring(parameters_, key);
  }

  const ContentPtr
  Content::merge_as_union(const ContentPtr& other) const {
    int64_t mylength = length();
    int64_t theirlength = other.get()->length();
    Index8 tags(mylength + theirlength);
    Index64 index(mylength + theirlength);

    ContentPtrVec contents({ shallow_copy(), other });

    struct Error err1 = kernel::UnionArray_filltags_to8_const(
      kernel::lib::cpu,   // DERIVE
      tags.data(),
      0,
      mylength,
      0);
    util::handle_error(err1, classname(), identities_.get());
    struct Error err2 = kernel::UnionArray_fillindex_count_64(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      0,
      mylength);
    util::handle_error(err2, classname(), identities_.get());

    struct Error err3 = kernel::UnionArray_filltags_to8_const(
      kernel::lib::cpu,   // DERIVE
      tags.data(),
      mylength,
      theirlength,
      1);
    util::handle_error(err3, classname(), identities_.get());
    struct Error err4 = kernel::UnionArray_fillindex_count_64(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      mylength,
      theirlength);
    util::handle_error(err4, classname(), identities_.get());

    return std::make_shared<UnionArray8_64>(Identities::none(),
                                            util::Parameters(),
                                            tags,
                                            index,
                                            contents);
  }

  const ContentPtr
  Content::rpad_axis0(int64_t target, bool clip) const {
    if (!clip  &&  target < length()) {
      return shallow_copy();
    }
    Index64 index(target);
    struct Error err = kernel::index_rpad_and_clip_axis0_64(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      target,
      length());
    util::handle_error(err, classname(), identities_.get());
    std::shared_ptr<IndexedOptionArray64> next =
      std::make_shared<IndexedOptionArray64>(Identities::none(),
                                             util::Parameters(),
                                             index,
                                             shallow_copy());
    return next.get()->simplify_optiontype();
  }

  const ContentPtr
  Content::localindex_axis0() const {
    Index64 localindex(length());
    struct Error err = kernel::localindex_64(
      kernel::lib::cpu,   // DERIVE
      localindex.data(),
      length());
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<NumpyArray>(localindex);
  }

  const ContentPtr
  Content::combinations_axis0(int64_t n,
                              bool replacement,
                              const util::RecordLookupPtr& recordlookup,
                              const util::Parameters& parameters) const {
    int64_t size = length();
    if (replacement) {
      size += (n - 1);
    }
    int64_t thisn = n;
    int64_t combinationslen;
    if (thisn > size) {
      combinationslen = 0;
    }
    else if (thisn == size) {
      combinationslen = 1;
    }
    else {
      if (thisn * 2 > size) {
        thisn = size - thisn;
      }
      combinationslen = size;
      for (int64_t j = 2;  j <= thisn;  j++) {
        combinationslen *= (size - j + 1);
        combinationslen /= j;
      }
    }

    std::vector<std::shared_ptr<int64_t>> tocarry;
    std::vector<int64_t*> tocarryraw;
    for (int64_t j = 0;  j < n;  j++) {
      std::shared_ptr<int64_t> ptr(new int64_t[(size_t)combinationslen],
                                   kernel::array_deleter<int64_t>());
      tocarry.push_back(ptr);
      tocarryraw.push_back(ptr.get());
    }
    IndexOf<int64_t> toindex(n);
    IndexOf<int64_t> fromindex(n);
    struct Error err = kernel::RegularArray_combinations_64(
      kernel::lib::cpu,   // DERIVE
      tocarryraw.data(),
      toindex.data(),
      fromindex.data(),
      n,
      replacement,
      length(),
      1);
    util::handle_error(err, classname(), identities_.get());

    ContentPtrVec contents;
    for (auto ptr : tocarry) {
      contents.push_back(std::make_shared<IndexedArray64>(
        Identities::none(),
        util::Parameters(),
        Index64(ptr, 0, combinationslen, kernel::lib::cpu),   // DERIVE
        shallow_copy()));
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         parameters,
                                         contents,
                                         recordlookup);
  }

  const ContentPtr
  Content::getitem(const Slice& where) const {
    ContentPtr next = std::make_shared<RegularArray>(Identities::none(),
                                                     util::Parameters(),
                                                     shallow_copy(),
                                                     length());
    SliceItemPtr nexthead = where.head();
    Slice nexttail = where.tail();
    Index64 nextadvanced(0);
    ContentPtr out = next.get()->getitem_next(nexthead,
                                              nexttail,
                                              nextadvanced);

    if (out.get()->length() == 0) {
      return out.get()->getitem_nothing();
    }
    else {
      return out.get()->getitem_at_nowrap(0);
    }
  }

  const ContentPtr
  Content::getitem_next(const SliceItemPtr& head,
                        const Slice& tail,
                        const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (SliceAt* at =
             dynamic_cast<SliceAt*>(head.get())) {
      return getitem_next(*at, tail, advanced);
    }
    else if (SliceRange* range =
             dynamic_cast<SliceRange*>(head.get())) {
      return getitem_next(*range, tail, advanced);
    }
    else if (SliceEllipsis* ellipsis =
             dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_next(*ellipsis, tail, advanced);
    }
    else if (SliceNewAxis* newaxis =
             dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_next(*newaxis, tail, advanced);
    }
    else if (SliceArray64* array =
             dynamic_cast<SliceArray64*>(head.get())) {
      return getitem_next(*array, tail, advanced);
    }
    else if (SliceField* field =
             dynamic_cast<SliceField*>(head.get())) {
      return getitem_next(*field, tail, advanced);
    }
    else if (SliceFields* fields =
             dynamic_cast<SliceFields*>(head.get())) {
      return getitem_next(*fields, tail, advanced);
    }
    else if (SliceMissing64* missing =
             dynamic_cast<SliceMissing64*>(head.get())) {
      return getitem_next(*missing, tail, advanced);
    }
    else if (SliceJagged64* jagged =
             dynamic_cast<SliceJagged64*>(head.get())) {
      return getitem_next(*jagged, tail, advanced);
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized slice type") + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  Content::getitem_next_jagged(const Index64& slicestarts,
                               const Index64& slicestops,
                               const SliceItemPtr& slicecontent,
                               const Slice& tail) const {
    if (SliceArray64* array =
        dynamic_cast<SliceArray64*>(slicecontent.get())) {
      return getitem_next_jagged(slicestarts, slicestops, *array, tail);
    }
    else if (SliceMissing64* missing =
             dynamic_cast<SliceMissing64*>(slicecontent.get())) {
      return getitem_next_jagged(slicestarts, slicestops, *missing, tail);
    }
    else if (SliceJagged64* jagged =
             dynamic_cast<SliceJagged64*>(slicecontent.get())) {
      return getitem_next_jagged(slicestarts, slicestops, *jagged, tail);
    }
    else {
      throw std::runtime_error(
        std::string("unexpected slice type for getitem_next_jagged")
        + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  Content::getitem_next(const SliceEllipsis& ellipsis,
                        const Slice& tail,
                        const Index64& advanced) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    int64_t mindepth = minmax.first;
    int64_t maxdepth = minmax.second;

    if (tail.length() == 0  ||
        (mindepth - 1 == tail.dimlength()  &&
         maxdepth - 1 == tail.dimlength())) {
      SliceItemPtr nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_next(nexthead, nexttail, advanced);
    }
    else if (mindepth - 1 == tail.dimlength()  ||
             maxdepth - 1 == tail.dimlength()) {
      throw std::invalid_argument(
        std::string("ellipsis (...) can't be used on a data structure of "
                    "different depths") + FILENAME(__LINE__));
    }
    else {
      std::vector<SliceItemPtr> tailitems = tail.items();
      std::vector<SliceItemPtr> items = { std::make_shared<SliceEllipsis>() };
      items.insert(items.end(), tailitems.begin(), tailitems.end());
      SliceItemPtr nexthead = std::make_shared<SliceRange>(Slice::none(),
                                                           Slice::none(),
                                                           1);
      Slice nexttail(items);
      return getitem_next(nexthead, nexttail, advanced);
    }
  }

  const ContentPtr
  Content::getitem_next(const SliceNewAxis& newaxis,
                        const Slice& tail,
                        const Index64& advanced) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    return std::make_shared<RegularArray>(
      Identities::none(),
      util::Parameters(),
      getitem_next(nexthead, nexttail, advanced),
      1);
  }

  const ContentPtr
  Content::getitem_next(const SliceField& field,
                        const Slice& tail,
                        const Index64& advanced) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_field(field.key()).get()->getitem_next(nexthead,
                                                          nexttail,
                                                          advanced);
  }

  const ContentPtr
  Content::getitem_next(const SliceFields& fields,
                        const Slice& tail,
                        const Index64& advanced) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_fields(fields.keys()).get()->getitem_next(nexthead,
                                                             nexttail,
                                                             advanced);
  }

  const ContentPtr getitem_next_regular_missing(const SliceMissing64& missing,
                                                const Slice& tail,
                                                const Index64& advanced,
                                                const RegularArray* raw,
                                                int64_t length,
                                                const std::string& classname) {
    Index64 index(missing.index());
    Index64 outindex(index.length()*length);

    struct Error err = kernel::missing_repeat_64(
      kernel::lib::cpu,   // DERIVE
      outindex.data(),
      index.data(),
      index.length(),
      length,
      raw->size());
    util::handle_error(err, classname, nullptr);

    IndexedOptionArray64 out(Identities::none(),
                             util::Parameters(),
                             outindex,
                             raw->content());
    return std::make_shared<RegularArray>(Identities::none(),
                                          util::Parameters(),
                                          out.simplify_optiontype(),
                                          index.length());
  }

  const ContentPtr getitem_next_missing_jagged(const SliceMissing64& missing,
                                               const Slice& tail,
                                               const Index64& advanced,
                                               const ContentPtr& that) {
    const SliceJagged64* jagged =
        dynamic_cast<SliceJagged64*>(missing.content().get());
    if (jagged == nullptr) {
      throw std::runtime_error(
        std::string("Logic error: calling getitem_next_missing_jagged with bad "
                    "slice type") + FILENAME(__LINE__));
    }
    const Index64 index = missing.index();
    ContentPtr content = that.get()->getitem_at_nowrap(0);
    if (content.get()->length() < index.length()) {
      throw std::invalid_argument(
        std::string("cannot fit masked jagged slice with length ") +
        std::to_string(index.length()) + std::string(" into ") +
        that.get()->classname() + std::string(" of size ") +
        std::to_string(content.get()->length()) + FILENAME(__LINE__));
    }
    Index64 outputmask(index.length());
    Index64 starts(index.length());
    Index64 stops(index.length());
    struct Error err = kernel::Content_getitem_next_missing_jagged_getmaskstartstop(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      jagged->offsets().data(),
      outputmask.data(),
      starts.data(),
      stops.data(),
      index.length());
    util::handle_error(err, that.get()->classname(), nullptr);
    ContentPtr tmp = content.get()->getitem_next_jagged(
        starts, stops, jagged->content(), tail);
    IndexedOptionArray64 out(Identities::none(), util::Parameters(), outputmask, tmp);
    return std::make_shared<RegularArray>(
        Identities::none(), util::Parameters(), out.simplify_optiontype(),
        index.length());
  }

  const ContentPtr
  Content::getitem_next(const SliceMissing64& missing,
                        const Slice& tail,
                        const Index64& advanced) const {
    if (advanced.length() != 0) {
      throw std::invalid_argument(
        std::string("cannot mix missing values in slice with NumPy-style "
                    "advanced indexing") + FILENAME(__LINE__));
    }

    if (dynamic_cast<SliceJagged64*>(missing.content().get())) {
      if (length() != 1) {
        throw std::runtime_error(
          std::string("Reached a not-well-considered code path")
          + FILENAME(__LINE__));
      }
      return getitem_next_missing_jagged(missing, tail, advanced,
                                         shallow_copy());
    }

    ContentPtr next = getitem_next(missing.content(), tail, advanced);

    if (RegularArray* raw = dynamic_cast<RegularArray*>(next.get())) {
      return getitem_next_regular_missing(missing,
                                          tail,
                                          advanced,
                                          raw,
                                          length(),
                                          classname());
    }

    else if (RecordArray* rec = dynamic_cast<RecordArray*>(next.get())) {
      if (rec->numfields() == 0) {
        return next;
      }
      ContentPtrVec contents;
      for (auto content : rec->contents()) {
        if (RegularArray* raw = dynamic_cast<RegularArray*>(content.get())) {
          contents.push_back(getitem_next_regular_missing(missing,
                                                          tail,
                                                          advanced,
                                                          raw,
                                                          length(),
                                                          classname()));
        }
        else {
          throw std::runtime_error(
            std::string("FIXME: unhandled case of SliceMissing with ")
            + std::string("RecordArray containing\n")
            + content.get()->tostring() + FILENAME(__LINE__));
        }
      }
      return std::make_shared<RecordArray>(Identities::none(),
                                           util::Parameters(),
                                           contents,
                                           rec->recordlookup());
    }

    else {
      throw std::runtime_error(
        std::string("FIXME: unhandled case of SliceMissing with\n")
        + next.get()->tostring() + FILENAME(__LINE__));
    }
  }

  const int64_t
  Content::axis_wrap_if_negative(int64_t axis) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    int64_t mindepth = minmax.first;
    int64_t maxdepth = minmax.second;
    int64_t depth = purelist_depth();
    if (axis < 0  &&  mindepth == depth  &&  maxdepth == depth) {
      int64_t posaxis = depth + axis;
      if (posaxis < 0) {
        throw std::invalid_argument(
          std::string("axis == ") + std::to_string(axis)
                      + std::string(" exceeds the depth == ") + std::to_string(depth)
                      + std::string(" of this array") + FILENAME(__LINE__));
      }
      return posaxis;
    } else if (axis < 0  &&  mindepth + axis == 0) {
      throw std::invalid_argument(
        std::string("axis == ") + std::to_string(axis)
                    + std::string(" exceeds the min depth == ") + std::to_string(mindepth)
                    + std::string(" of this array") + FILENAME(__LINE__));
    }
    return axis;
  }

  const ContentPtr
  Content::getitem_next_array_wrap(const ContentPtr& outcontent,
                                   const std::vector<int64_t>& shape) const {
    ContentPtr out =
      std::make_shared<RegularArray>(Identities::none(),
                                     util::Parameters(),
                                     outcontent,
                                     (int64_t)shape[shape.size() - 1]);
    for (int64_t i = (int64_t)shape.size() - 2;  i >= 0;  i--) {
      out = std::make_shared<RegularArray>(Identities::none(),
                                           util::Parameters(),
                                           out,
                                           (int64_t)shape[(size_t)i]);
    }
    return out;
  }

  const std::string
  Content::parameters_tostring(const std::string& indent,
                               const std::string& pre,
                               const std::string& post) const {
    if (parameters_.empty()) {
      return "";
    }
    else {
      std::stringstream out;
      out << indent << pre << "<parameters>\n";
      for (auto pair : parameters_) {
        out << indent << "    <param key=" << util::quote(pair.first)
            << ">" << pair.second << "</param>\n";
      }
      out << indent << "</parameters>" << post;
      return out.str();
    }
  }
}
