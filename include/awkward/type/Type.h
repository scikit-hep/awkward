// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_TYPE_H_
#define AWKWARD_TYPE_H_

#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/util.h"

namespace awkward {
  class Content;
  using ContentPtr = std::shared_ptr<Content>;

  class Type;
  using TypePtr = std::shared_ptr<Type>;

  /// @class Type
  ///
  /// @brief Abstract superclass of all high level types (flat hierarchy).
  class LIBAWKWARD_EXPORT_SYMBOL Type {
  public:
    /// @brief Called by all subclass constructors; assigns #parameters and
    /// #typestr upon construction.
    Type(const util::Parameters& parameters, const std::string& typestr);

    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~Type();

    /// @brief Internal function to build an output string for #tostring.
    ///
    /// @param indent Indentation depth as a string of spaces.
    /// @param pre Prefix string, usually an opening XML tag.
    /// @param post Postfix string, usually a closing XML tag and carriage
    /// return.
    virtual std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const = 0;

    /// @brief Copies this Type without copying any hierarchically nested
    /// types.
    virtual const TypePtr
      shallow_copy() const = 0;

    /// @brief Returns `true` if this type is equal to `other`; `false`
    /// otherwise.
    ///
    /// @param other The other Type.
    /// @param check_parameters If `true`, types must have the same
    /// #parameters to be considered equal; if `false`, types do not check
    /// #parameters.
    virtual bool
      equal(const TypePtr& other, bool check_parameters) const = 0;

    /// @brief The number of fields in the first nested tuple or
    /// records or `-1` if this array does not contain a RecordType.
    virtual int64_t
      numfields() const = 0;

    /// @brief The position of a tuple or record key name if this array
    /// contains a RecordType.
    virtual int64_t
      fieldindex(const std::string& key) const = 0;

    /// @brief The record name associated with a given field index or
    /// the tuple index as a string (e.g. `"0"`, `"1"`, `"2"`) if a tuple.
    ///
    /// Raises an error if the array does not contain a RecordType.
    virtual const std::string
      key(int64_t fieldindex) const = 0;

    /// @brief Returns `true` if the type contains a RecordType with the
    /// specified `key`; `false` otherwise.
    virtual bool
      haskey(const std::string& key) const = 0;

    /// @brief A list of RecordType keys or an empty list if this
    /// type does not contain a RecordType.
    virtual const std::vector<std::string>
      keys() const = 0;

    /// @brief Returns an empty array (Content) with this type.
    virtual const ContentPtr
      empty() const = 0;

    /// @brief Get one parameter from this type.
    ///
    /// If the `key` does not exist, this function returns `"null"`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    const util::Parameters
      parameters() const;

    /// @brief Assign one parameter for this type (in-place).
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// @note This mutability is temporary:
    /// [scikit-hep/awkward-1.0#117](https://github.com/scikit-hep/awkward-1.0/issues/177)
    /// Eventually, this interface will be deprecated and all Content
    /// instances will be immutable.
    void
      setparameters(const util::Parameters& parameters);

    /// @brief Custom parameters inherited from the Content that
    /// this type describes.
    ///
    /// If the `key` does not exist, this function returns `"null"`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    const std::string
      parameter(const std::string& key) const;

    /// @brief Assign one parameter to this type (in-place).
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// @note This mutability is temporary:
    /// [scikit-hep/awkward-1.0#117](https://github.com/scikit-hep/awkward-1.0/issues/177)
    /// Eventually, this interface will be deprecated and all Content
    /// instances will be immutable.
    void
      setparameter(const std::string& key, const std::string& value);

    /// @brief Returns `true` if the parameter associated with `key` exists
    /// and is equal to `value`; `false` otherwise.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// Equality is checked at the level of JSON DOMs. The `value` does not
    /// need to be exactly the same string; it needs to have equivalent JSON
    /// value.
    bool
      parameter_equals(const std::string& key, const std::string& value) const;

    /// @brief Returns `true` if all parameters of this type are equal
    /// to the `other` parameters.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// Equality is checked at the level of JSON DOMs. The `value` does not
    /// need to be exactly the same string; it needs to have equivalent JSON
    /// value.
    bool
      parameters_equal(const util::Parameters& other) const;

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string; `false` otherwise.
    bool
      parameter_isstring(const std::string& key) const;

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string that matches `[A-Za-z_][A-Za-z_0-9]*`; `false` otherwise.
    bool
      parameter_isname(const std::string& key) const;

    /// @brief Returns the parameter associated with `key` as a string if
    /// #parameter_isstring; raises an error otherwise.
    const std::string
      parameter_asstring(const std::string& key) const;

    /// @brief Returns a string representation of the type as a
    /// [Datashape](https://datashape.readthedocs.io/en/latest/) or its
    /// #typestr overload (if non-empty).
    const std::string
      tostring() const;

    /// @brief Returns a string showing a side-by-side comparison of two types,
    /// highlighting differences.
    ///
    /// @note This function does not align types side-by-side.
    const std::string
      compare(TypePtr supertype);

    /// @brief Optional string that overrides the default string
    /// representation (missing if empty).
    const std::string
      typestr() const;

  protected:
    /// @brief Internal function that replaces `output` in-place with the
    /// #typestr and returns `true` if the #typestr is not missing (i.e.
    /// empty); otherwise, it leaves `output` untouched and returns `false`.
    bool
      get_typestr(std::string& output) const;

    /// @brief Internal function to determine if there are no parameters
    /// *except* `__categorical__`.
    bool
      parameters_empty() const;

    /// @brief Internal function that wraps `output` with `categorical[type=`
    /// and `]` if `__categorical__` is `true`; passes through otherwise.
    std::string
      wrap_categorical(const std::string& output) const;

    /// @brief Internal function to format parameters as part of the #tostring
    /// string.
    const std::string
      string_parameters() const;

    /// @brief See #parameters.
    util::Parameters parameters_;
    /// @brief See #typestr.
    const std::string typestr_;
  };
}

#endif // AWKWARD_TYPE_H_
