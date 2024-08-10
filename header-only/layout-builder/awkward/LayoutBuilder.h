// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_LAYOUTBUILDER_H_
#define AWKWARD_LAYOUTBUILDER_H_

#include "awkward/BuilderOptions.h"
#include "awkward/GrowableBuffer.h"
#include "awkward/utils.h"

#include <map>
#include <algorithm>
#include <tuple>
#include <string>
#include <functional>

/// @brief Object of {@link BuilderOptions BuilderOptions} which sets the
/// values of the default options.
#define AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS awkward::BuilderOptions(1024, 1)

namespace awkward {

  namespace LayoutBuilder {

    /// @class Field
    ///
    /// @brief Helper class for sending a pair of field names (as enum) and field
    /// type as template parameters in {@link Record Record}.
    ///
    /// @tparam ENUM Enumerated type which assigns index values to field names.
    /// @tparam BUILDER The type of field.
    template <std::size_t ENUM, typename BUILDER>
    class Field {
    public:
      using Builder = BUILDER;

      /// @brief Converts index as field string.
      std::string
      index_as_field() const {
        return std::to_string(index);
      }

      /// @brief The index of a Record field.
      const std::size_t index = ENUM;
      /// @brief The content type of field in a Record.
      Builder builder;
    };

    /// @class Numpy
    ///
    /// @brief Builds a NumpyArray which describes multi-dimensional data
    /// of `PRIMITIVE` type.
    ///
    /// @tparam PRIMITIVE The type of data in NumpyArray.
    template <typename PRIMITIVE>
    class Numpy {
    public:
      /// @brief Creates a new Numpy layout builder by allocating a new buffer,
      /// using `AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS` for initializing the buffer.
      Numpy()
          : data_(
                awkward::GrowableBuffer<PRIMITIVE>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Creates a new Numpy layout builder by allocating a new buffer,
      /// taking `options` from {@link BuilderOptions BuilderOptions}
      /// for initializing the buffer.
      ///
      /// @param options Initial size configuration of a buffer.
      Numpy(const awkward::BuilderOptions& options)
          : data_(awkward::GrowableBuffer<PRIMITIVE>(options)) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Inserts a `PRIMITIVE` type data.
      void
      append(PRIMITIVE x) noexcept {
        data_.append(x);
      }

      /// @brief Inserts an entire array of `PRIMITIVE` type data.
      ///
      /// Just an interface; not actually faster than calling append many times.
      void
      extend(PRIMITIVE* ptr, size_t size) noexcept {
        data_.extend(ptr, size);
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
      }

      /// @brief Discards the accumulated data in the builder.
      void
      clear() noexcept {
        data_.clear();
      }

      /// @brief Current length of the data.
      size_t
      length() const noexcept {
        return data_.length();
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& /* error */) const noexcept {
        return true;
      }

      /// @brief Retrieves the name and size (in bytes) of the buffer.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        names_nbytes["node" + std::to_string(id_) + "-data"] = data_.nbytes();
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a user-defined pointer.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same name and size (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        data_.concatenate(reinterpret_cast<PRIMITIVE*>(
            buffers["node" + std::to_string(id_) + "-data"]));
      }

      /// @brief Copies and concatenates the accumulated data in the builder buffer to
      /// a user-defined pointer if the given node name matches with the node associated
      /// with the builder.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        if (std::string(name) == std::string("node" + std::to_string(id_) + "-data")) {
          data_.concatenate(reinterpret_cast<PRIMITIVE*>(buffer));
        }
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        data_.concatenate(reinterpret_cast<PRIMITIVE*>(
          buffers["node" + std::to_string(id_) + "-data"]));
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const {
        std::stringstream form_key;
        form_key << "node" << id_;

        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }

        if (std::is_arithmetic<PRIMITIVE>::value) {
          return "{ \"class\": \"NumpyArray\", \"primitive\": \"" +
                 type_to_name<PRIMITIVE>() + "\"" + params +
                 ", \"form_key\": \"" + form_key.str() + "\" }";
        } else if (is_specialization<PRIMITIVE, std::complex>::value) {
          return "{ \"class\": \"NumpyArray\", \"primitive\": \"" +
                 type_to_name<PRIMITIVE>() + "\"" + params +
                 ", \"form_key\": \"" + form_key.str() + "\" }";
        } else {
          throw std::runtime_error("type " +
                                   std::string(typeid(PRIMITIVE).name()) +
                                   "is not supported");
        }
      }

    private:
      /// @brief Buffer of `PRIMITIVE` type.
      awkward::GrowableBuffer<PRIMITIVE> data_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;
    };

    /// @class ListOffset
    ///
    /// @brief Builds a ListOffsetArray which describes unequal-length lists
    /// (often called a "jagged" or "ragged" array). The underlying data for
    /// all lists are in a BUILDER content. It is subdivided into lists according
    /// to an offsets array, which specifies the starting and stopping index of each list.
    ///
    /// The offsets must have at least length `1` (corresponding to an empty array).
    ///
    /// The offsets values can be 64-bit signed integers `int64`, 32-bit signed integers
    /// `int32` or 32-bit unsigned integers `uint32`.
    ///
    /// @tparam PRIMITIVE The type of `offsets` buffer.
    /// @tparam BUILDER The type of builder content.
    template <typename PRIMITIVE, typename BUILDER>
    class ListOffset {
    public:
      /// @brief Creates a new ListOffset layout builder by allocating a new `offset`
      ///  buffer, using `AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS` for initializing the buffer.
      ListOffset()
          : offsets_(
                awkward::GrowableBuffer<PRIMITIVE>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)) {
        offsets_.append(0);
        size_t id = 0;
        set_id(id);
      }

      /// @brief Creates a new ListOffset layout builder by allocating a new `offset`
      /// buffer, taking `options` from {@link BuilderOptions BuilderOptions} for
      /// initializing the buffer.
      ///
      /// @param options Initial size configuration of a buffer.
      ListOffset(const awkward::BuilderOptions& options)
          : offsets_(awkward::GrowableBuffer<PRIMITIVE>(options)) {
        offsets_.append(0);
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns the reference to the builder content.
      BUILDER&
      content() noexcept {
        return content_;
      }

      /// @brief Begins a list and returns the reference to the builder content.
      BUILDER&
      begin_list() noexcept {
        return content_;
      }

      /// @brief Ends a list and appends the current length of the list
      /// contents in the `offsets` buffer.
      void
      end_list() noexcept {
        offsets_.append(content_.length());
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        content_.set_id(id);
      }

      /// @brief Discards the accumulated offsets and clears the builder content.
      void
      clear() noexcept {
        offsets_.clear();
        offsets_.append(0);
        content_.clear();
      }

      /// @brief Current length of the content.
      size_t
      length() const noexcept {
        return offsets_.length() - 1;
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        if ((int64_t)content_.length() != (int64_t)offsets_.last()) {
          std::stringstream out;
          out << "ListOffset node" << id_ << "has content length "
              << content_.length() << "but last offset " << offsets_.last()
              << "\n";
          error.append(out.str());

          return false;
        } else {
          return content_.is_valid(error);
        }
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        names_nbytes["node" + std::to_string(id_) + "-offsets"] =
            offsets_.nbytes();
        content_.buffer_nbytes(names_nbytes);
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        offsets_.concatenate(reinterpret_cast<PRIMITIVE*>(
            buffers["node" + std::to_string(id_) + "-offsets"]));
        content_.to_buffers(buffers);
      }

      /// @brief Copies and concatenates the accumulated data in the builder buffer to
      /// a user-defined pointer if the given node name matches with the node associated
      /// with the builder; otherwise, it searches the builder contents to locate a
      /// matching node.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        if (std::string(name) == std::string("node" + std::to_string(id_) + "-offsets")) {
          offsets_.concatenate(reinterpret_cast<PRIMITIVE*>(buffer));
        }
        content_.to_buffer(buffer, name);
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        offsets_.concatenate(reinterpret_cast<PRIMITIVE*>(
            buffers["node" + std::to_string(id_) + "-offsets"]));
        content_.to_char_buffers(buffers);
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        return "{ \"class\": \"ListOffsetArray\", \"offsets\": \"" +
               type_to_numpy_like<PRIMITIVE>() +
               "\", \"content\": " + content_.form() + params +
               ", \"form_key\": \"" + form_key.str() + "\" }";
      }

    private:
      /// @brief Buffer of `PRIMITIVE` type.
      ///
      /// It specifies the starting and stopping index of each list.
      GrowableBuffer<PRIMITIVE> offsets_;

      /// @brief The content of the ListOffsetArray.
      BUILDER content_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;
    };

    /// @class String
    ///
    /// @brief Helper for building an array of strings with a similar API as a
    /// Numpy builder.
    template<class PRIMITIVE>
    class String : public ListOffset<PRIMITIVE, Numpy<uint8_t>> {
      public:
          String() : ListOffset<PRIMITIVE, Numpy<uint8_t>>() {
      this->set_parameters(R"""("__array__": "string")""");
      this->content().set_parameters(R"""("__array__": "char")""");
    }

      void append(const std::string& value) {
          this->begin_list();
          for (const auto& c: value) {
              this->content().append(c);
          }
          this->end_list();
      }
    };

    /// @class Empty
    ///
    /// @brief Builds an EmptyArray which has no content in it.
    /// It is used whenever an array's type is not known because it is empty.
    class Empty {
    public:
      /// @brief Creates a new Empty layout builder.
      Empty() {
        size_t id = 0;
        set_id(id);
      }

      void
      set_id(size_t& /* id */) noexcept {}

      void
      clear() noexcept {}

      /// @brief Current length of the content.
      size_t
      length() const noexcept {
        return 0;
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& /* error */) const noexcept {
        return true;
      }

      void
      buffer_nbytes(std::map<std::string, size_t>& /* names_nbytes */) const
          noexcept {}

      void
      to_buffers(std::map<std::string, void*>& /* buffers */) const noexcept {}

      void
      to_buffer(void* /* buffer */, const char* /* name */) const noexcept {}

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& /* buffers */) const noexcept {}

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        return "{ \"class\": \"EmptyArray\" }";
      }

    private:
      /// @brief Unique form ID.
      size_t id_;
    };


    /// @class Record
    ///
    /// @brief Builds a RecordArray which represents an array of records, which
    /// can be of same or different types. Its contents is an ordered list of arrays
    /// with the same length as the length of its shortest content; all are aligned
    /// element-by-element, associating a field name to every content.
    ///
    /// @tparam MAP Map of index keys and field name.
    /// @tparam BUILDERS The types of builder contents.
    template <class MAP = std::map<std::size_t, std::string>,
              typename... BUILDERS>
    class Record {
    public:
      using RecordContents = typename std::tuple<BUILDERS...>;
      using UserDefinedMap = MAP;

      template <std::size_t INDEX>
      using RecordFieldType = std::tuple_element_t<INDEX, RecordContents>;

      /// @brief Creates a new Record layout builder.
      Record() {
        size_t id = 0;
        set_id(id);
        map_fields(std::index_sequence_for<BUILDERS...>());
      }

      /// @brief Creates a new Record layout builder, taking a user-defined
      /// map with enumerated type field ID as keys and field names as value
      /// which sets the field names.
      ///
      /// @param user_defined_field_id_to_name_map A user-defined field ID
      /// field name map.
      Record(UserDefinedMap user_defined_field_id_to_name_map)
          : content_names_(user_defined_field_id_to_name_map) {
        assert(content_names_.size() == fields_count_);
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns a vector of strings sontaining all the field names.
      const std::vector<std::string>
      fields() const noexcept {
        if (content_names_.empty()) {
          return fields_;
        } else {
          std::vector<std::string> result;
          for (auto it : content_names_) {
            result.emplace_back(it.second);
          }
          return result;
        }
      }

      /// @brief Sets the field names.
      ///
      /// Alternative method to set the field names besides passing the
      /// user-defined map as constructor parameter.
      void
      set_fields(MAP user_defined_field_id_to_name_map) noexcept {
        content_names_ = user_defined_field_id_to_name_map;
      }

      /// @brief Returns the reference to the builder contents at `INDEX`.
      template <std::size_t INDEX>
      typename RecordFieldType<INDEX>::Builder&
      content() noexcept {
        return std::get<INDEX>(contents).builder;
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&id](auto& content) {
            content.builder.set_id(id);
          });
        }
      }

      /// @brief Clears the builder contents.
      ///
      /// Discards the accumulated data and the contents in each
      /// field of the record.
      void
      clear() noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [](auto& content) {
            content.builder.clear();
          });
        }
      }

      /// @brief Current number of records in first field.
      size_t
      length() const noexcept {
        return (std::get<0>(contents).builder.length());
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        auto index_sequence((std::index_sequence_for<BUILDERS...>()));

        int64_t length = -1;
        std::vector<size_t> lengths = field_lengths(index_sequence);
        for (size_t i = 0; i < lengths.size(); i++) {
          if (length == -1) {
            length = lengths[i];
          }
          else if (length != (int64_t)lengths[i]) {
            std::stringstream out;
            out << "Record node" << id_ << " has field \""
                << fields().at(i) << "\" length " << lengths[i]
                << " that differs from the first length " << length << "\n";
            error.append(out.str());

            return false;
          }
        }

        std::vector<bool> valid_fields = field_is_valid(index_sequence, error);
        return std::none_of(std::cbegin(valid_fields),
                            std::cend(valid_fields),
                            std::logical_not<bool>());
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&names_nbytes](auto& content) {
            content.builder.buffer_nbytes(names_nbytes);
          });
        }
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&buffers](auto& content) {
            content.builder.to_buffers(buffers);
          });
        }
      }

      /// @brief Copies and concatenates the accumulated data in the buffers of the
      /// builder contents to user-defined pointers if the given node name matches
      /// with the node associated with that builder.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&buffer, &name](auto& content) {
            content.builder.to_buffer(buffer, name);
          });
        }
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&buffers](auto& content) {
            content.builder.to_char_buffers(buffers);
          });
        }
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string("\"parameters\": { " + parameters_ + " }, ");
        }
        std::stringstream out;
        out << "{ \"class\": \"RecordArray\", \"contents\": { ";
        for (size_t i = 0; i < fields_count_; i++) {
          if (i != 0) {
            out << ", ";
          }
          auto contents_form = [&](auto& content) {
            out << "\""
                << (!content_names_.empty() ? content_names_.at(content.index)
                                            : content.index_as_field())
                << +"\": ";
            out << content.builder.form();
          };
          visit_at(contents, i, contents_form);
        }
        out << " }, ";
        out << params << "\"form_key\": \"" << form_key.str() << "\" }";
        return out.str();
      }

      /// @brief The contents of the RecordArray.
      RecordContents contents;

    private:
      /// @brief Inserts the field names of all the record fields in a vector
      /// of strings.
      template <std::size_t... S>
      void
      map_fields(std::index_sequence<S...>) noexcept {
        fields_ = std::vector<std::string>(
            {std::string(std::get<S>(contents).index_as_field())...});
      }

      /// @brief Inserts the lengths of all the record fields in a vector and
      /// returns the vector.
      template <std::size_t... S>
      std::vector<size_t>
      field_lengths(std::index_sequence<S...>) const noexcept {
        return std::vector<size_t>({std::get<S>(contents).builder.length()...});
      }

      /// @brief Helper function of #is_valid for checking the validity.
      template <std::size_t... S>
      std::vector<bool>
      field_is_valid(std::index_sequence<S...>, std::string& error) const
          noexcept {
        return std::vector<bool>(
            {std::get<S>(contents).builder.is_valid(error)...});
      }

      /// @brief Vector of strings of field names.
      std::vector<std::string> fields_;

      /// @brief User-defined map of record field names.
      UserDefinedMap content_names_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief The count of the fields in the builder.
      static constexpr size_t fields_count_ = sizeof...(BUILDERS);
    };

    /// @class Tuple
    ///
    /// @brief Builds a RecordArray which represents an array of tuples which can be
    /// of same or different types without field names, indexed only by their order.
    ///
    /// @tparam BUILDERS The types of builder contents.
    template <typename... BUILDERS>
    class Tuple {
      using TupleContents = typename std::tuple<BUILDERS...>;

      template <std::size_t INDEX>
      using TupleContentType = std::tuple_element_t<INDEX, TupleContents>;

    public:
      /// @brief Creates a new Tuple layout builder.
      Tuple() {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns the reference to the builder contents at `INDEX`.
      template <std::size_t INDEX>
      TupleContentType<INDEX>&
      content() noexcept {
        return std::get<INDEX>(contents);
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&id](auto& content) {
            content.set_id(id);
          });
        }
      }

      /// @brief Clears the builder contents.
      ///
      /// Discards the accumulated data and the contents at each tuple index.
      void
      clear() noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [](auto& content) {
            content.clear();
          });
        }
      }

      /// @brief Current number of records in the first index of the tuple.
      size_t
      length() const noexcept {
        return (std::get<0>(contents).length());
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        auto index_sequence((std::index_sequence_for<BUILDERS...>()));

        int64_t length = -1;
        std::vector<size_t> lengths = content_lengths(index_sequence);
        for (size_t i = 0; i < lengths.size(); i++) {
          if (length == -1) {
            length = (int64_t)lengths[i];
          }
          else if (length != (int64_t)lengths[i]) {
            std::stringstream out;
            out << "Record node" << id_ << " has index \"" << i << "\" length "
                << lengths[i] << " that differs from the first length "
                << length << "\n";
            error.append(out.str());

            return false;
          }
        }

        std::vector<bool> valid_fields =
            content_is_valid(index_sequence, error);
        return std::none_of(std::cbegin(valid_fields),
                            std::cend(valid_fields),
                            std::logical_not<bool>());
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&names_nbytes](auto& content) {
            content.buffer_nbytes(names_nbytes);
          });
        }
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&buffers](auto& content) {
            content.to_buffers(buffers);
          });
        }
      }

      /// @brief Copies and concatenates the accumulated data in the buffers of the
      /// builder contents to user-defined pointers if the given node name matches
      /// with the node associated with that builder.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&buffer, &name](auto& content) {
            content.to_buffer(buffer, name);
          });
        }
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        for (size_t i = 0; i < fields_count_; i++) {
          visit_at(contents, i, [&buffers](auto& content) {
            content.to_char_buffers(buffers);
          });
        }
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string("\"parameters\": { " + parameters_ + " }, ");
        }
        std::stringstream out;
        out << "{ \"class\": \"RecordArray\", \"contents\": [";
        for (size_t i = 0; i < fields_count_; i++) {
          if (i != 0) {
            out << ", ";
          }
          auto contents_form = [&out](auto& content) {
            out << content.form();
          };
          visit_at(contents, i, contents_form);
        }
        out << "], ";
        out << params << "\"form_key\": \"" << form_key.str() << "\" }";
        return out.str();
      }

      /// @brief The contents of the RecordArray without fields.
      TupleContents contents;

    private:
      /// @brief Inserts the lengths of all the builder contents in a vector
      /// and returns the vector.
      template <std::size_t... S>
      std::vector<size_t>
      content_lengths(std::index_sequence<S...>) const noexcept {
        return std::vector<size_t>({std::get<S>(contents).length()...});
      }

      /// @brief Helper function of #is_valid for checking the validity.
      template <std::size_t... S>
      std::vector<bool>
      content_is_valid(std::index_sequence<S...>, std::string& error) const
          noexcept {
        return std::vector<bool>({std::get<S>(contents).is_valid(error)...});
      }

      /// @brief Vector of integers of field index.
      std::vector<int64_t> field_index_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief The count of the fields in the builder.
      static constexpr size_t fields_count_ = sizeof...(BUILDERS);
    };

    /// @class Regular
    ///
    /// @brief Builds a RegularArray that describes lists that have the same
    /// length, a single integer size. Its underlying content is a flattened
    /// view of the data; that is, each list is not stored separately in memory,
    /// but is inferred as a subinterval of the underlying data.
    ///
    /// A multidimensional {@link Numpy NumpyArray} is equivalent to a one-dimensional
    /// {@link Numpy NumpyArray} nested within several RegularArrays, one for each
    /// dimension. However, RegularArrays can be used to make lists of any other type.
    ///
    /// @tparam SIZE
    /// @tparam BUILDER The type of builder content.
    template <unsigned SIZE, typename BUILDER>
    class Regular {
    public:
      /// @brief Creates a new Regular layout builder.
      Regular() : length_(0) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns the reference to the builder content.
      BUILDER&
      content() noexcept {
        return content_;
      }

      /// @brief Begins a list and returns the reference to the content
      /// of the builder.
      BUILDER&
      begin_list() noexcept {
        return content_;
      }

      /// @brief Ends a list and increments the number of lists.
      void
      end_list() noexcept {
        length_++;
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        content_.set_id(id);
      }

      /// @brief Clears the builder content.
      void
      clear() noexcept {
        length_ = 0;
        content_.clear();
      }

      /// @brief Current number of lists of length `SIZE`.
      size_t
      length() const noexcept {
        return length_;
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        if (content_.length() != length_ * size_) {
          std::stringstream out;
          out << "Regular node" << id_ << "has content length "
              << content_.length() << ", but length " << length_ << " and size "
              << size_ << "\n";
          error.append(out.str());

          return false;
        } else {
          return content_.is_valid(error);
        }
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        content_.buffer_nbytes(names_nbytes);
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        content_.to_buffers(buffers);
      }

      /// @brief Copies and concatenates the accumulated data in the buffers of the
      /// builder content to user-defined pointers if the given node name matches
      /// with the node associated with that builder.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        content_.to_buffer(buffer, name);
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        content_.to_char_buffers(buffers);
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        return "{ \"class\": \"RegularArray\", \"content\": " +
               content_.form() + ", \"size\": " + std::to_string(size_) +
               params + ", \"form_key\": \"" + form_key.str() + "\" }";
      }

    private:
      /// @brief The content of the RegularArray.
      BUILDER content_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief Current number of lists of length `SIZE`.
      size_t length_;

      /// @brief Length of each list.
      size_t size_ = SIZE;
    };


    /// @class Indexed
    ///
    /// @brief Builds an IndexedArray which consists of an `index` buffer.
    ///
    /// The index values can be 64-bit signed integers `int64`, 32-bit signed
    /// integers `int32` or 32-bit unsigned integers `uint32`.
    ///
    /// @tparam PRIMITIVE The type of `index` buffer.
    /// @tparam BUILDER The type of builder content.
    template <typename PRIMITIVE, typename BUILDER>
    class Indexed {
    public:
      /// @brief Creates a new Indexed layout builder by allocating a new `index`
      /// buffer, using `AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS` for initializing the buffer.
      Indexed()
          : index_(
                awkward::GrowableBuffer<PRIMITIVE>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)),
            max_index_(0) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Creates a new Indexed layout builder by allocating a new `index`
      /// buffer, taking `options` from {@link BuilderOptions BuilderOptions}
      /// for initializing the buffer.
      ///
      /// @param options Initial size configuration of a buffer.
      Indexed(const awkward::BuilderOptions& options)
          : index_(awkward::GrowableBuffer<PRIMITIVE>(options)),
            max_index_(0) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns the reference to the builder content.
      BUILDER&
      content() noexcept {
        return content_;
      }

      /// @brief Inserts the last valid index in the `index` buffer and
      /// returns the reference to the builder content.
      BUILDER&
      append_index() noexcept {
        return append_index(content_.length());
      }

      /// @brief Inserts an explicit value in the `index` buffer and
      /// returns the reference to the builder content.
      BUILDER&
      append_index(size_t i) noexcept {
        index_.append(i);
        if (i > max_index_) {
          max_index_ = i;
        } else if (i < 0) {
          max_index_ = UINTMAX_MAX;
        }
        return content_;
      }

      /// @brief Inserts `size` number indices in the `index` buffer
      /// and returns the reference to the builder content.
      ///
      /// Just an interface; not actually faster than calling append many times.
      BUILDER&
      extend_index(size_t size) noexcept {
        size_t start = content_.length();
        size_t stop = start + size;
        if (stop - 1 > max_index_) {
          max_index_ = stop - 1;
        }
        for (size_t i = start; i < stop; i++) {
          index_.append(i);
        }
        return content_;
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        content_.set_id(id);
      }

      /// @brief Discards the accumulated index and clears the content
      /// of the builder.
      void
      clear() noexcept {
        max_index_ = 0;
        index_.clear();
        content_.clear();
      }

      /// @brief Current length of the `index` buffer.
      size_t
      length() const noexcept {
        return index_.length();
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        names_nbytes["node" + std::to_string(id_) + "-index"] = index_.nbytes();
        content_.buffer_nbytes(names_nbytes);
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {

        if (max_index_ == UINTMAX_MAX) {
          std::stringstream out;
          out << "Indexed node" << id_ << " has a negative index\n";
          error.append(out.str());
          return false;
        } else if ((content_.length() == 0) && (max_index_ == 0)) { // catches empty object
        } else if (max_index_ >= content_.length()) {
          std::stringstream out;
          out << "Indexed node" << id_ << " has index " << max_index_
              << " but content has length "
              << content_.length() << "\n";
          error.append(out.str());
          return false;
        }
        return content_.is_valid(error);
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        index_.concatenate(reinterpret_cast<PRIMITIVE*>(
            buffers["node" + std::to_string(id_) + "-index"]));
        content_.to_buffers(buffers);
      }

      /// @brief Copies and concatenates the accumulated data in the builder buffer to
      /// a user-defined pointer if the given node name matches with the node associated
      /// with the builder; otherwise, it searches the builder contents to locate a
      /// matching node.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        if (std::string(name) == std::string("node" + std::to_string(id_) + "-index")) {
          index_.concatenate(reinterpret_cast<PRIMITIVE*>(buffer));
        }
        content_.to_buffer(buffer, name);
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        index_.concatenate(reinterpret_cast<PRIMITIVE*>(
            buffers["node" + std::to_string(id_) + "-index"]));
        content_.to_char_buffers(buffers);
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        return "{ \"class\": \"IndexedArray\", \"index\": \"" +
               type_to_numpy_like<PRIMITIVE>() +
               "\", \"content\": " + content_.form() + params +
               ", \"form_key\": \"" + form_key.str() + "\" }";
      }

    private:
      /// @brief Buffer of `PRIMITIVE` type.
      ///
      /// It specifies the index of each element.
      GrowableBuffer<PRIMITIVE> index_;

      /// @brief The content of the IndexedOptionArray.
      BUILDER content_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief Keep track of maximum index value.
      size_t max_index_;
    };

    /// @class IndexedOption
    ///
    /// @brief Builds an IndexedOptionArray which consists of an `index` buffer.
    /// The negative values in the index are interpreted as missing.
    ///
    /// The index values can be 64-bit signed integers `int64`, 32-bit signed
    /// integers `int32`.
    ///
    /// @tparam PRIMITIVE The type of `index` buffer.
    /// @tparam BUILDER The type of builder content.
    template <typename PRIMITIVE, typename BUILDER>
    class IndexedOption {
    public:
      /// @brief Creates a new IndexedOption layout builder by allocating a new `index`
      /// buffer, using `AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS` for initializing the buffer.
      IndexedOption()
          : index_(
                awkward::GrowableBuffer<PRIMITIVE>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)),
            last_valid_(-1),
            max_index_(0) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Creates a new IndexedOption layout builder by allocating a new `index`
      /// buffer, taking `options` from {@link BuilderOptions BuilderOptions}
      /// for initializing the buffer.
      ///
      /// @param options Initial size configuration of a buffer.
      IndexedOption(const awkward::BuilderOptions& options)
          : index_(awkward::GrowableBuffer<PRIMITIVE>(options)),
            last_valid_(-1),
            max_index_(0) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns the reference to the builder content.
      BUILDER&
      content() noexcept {
        return content_;
      }

      /// @brief Inserts the last valid index in the `index` buffer and
      /// returns the reference to the builder content.
      BUILDER&
      append_valid() noexcept {
        last_valid_ = content_.length();
        return append_valid(last_valid_);
      }

      /// @brief Inserts an explicit value in the `index` buffer and
      /// returns the reference to the builder content.
      BUILDER&
      append_valid(size_t i) noexcept {
        last_valid_ = content_.length();
        index_.append(i);
        if (i > max_index_) {
          max_index_ = i;
        }
        return content_;
      }

      /// @brief Inserts `size` number of valid index in the `index` buffer
      /// and returns the reference to the builder content.
      ///
      /// Just an interface; not actually faster than calling append many times.
      BUILDER&
      extend_valid(size_t size) noexcept {
        size_t start = content_.length();
        size_t stop = start + size;
        last_valid_ = stop - 1;
        if (last_valid_ > max_index_) {
          max_index_ = last_valid_;
        }
        for (size_t i = start; i < stop; i++) {
          index_.append(i);
        }
        return content_;
      }

      /// @brief Inserts `-1` in the `index` buffer.
      void
      append_invalid() noexcept {
        index_.append(-1);
      }

      /// @brief Inserts `-1` in the `index` buffer `size` number of times
      ///
      /// Just an interface; not actually faster than calling append many times.
      void
      extend_invalid(size_t size) noexcept {
        for (size_t i = 0; i < size; i++) {
          index_.append(-1);
        }
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        content_.set_id(id);
      }

      /// @brief Discards the accumulated index and clears the content
      /// of the builder. Also, last valid returns to `-1`.
      void
      clear() noexcept {
        last_valid_ = -1;
        index_.clear();
        content_.clear();
      }

      /// @brief Current length of the `index` buffer.
      size_t
      length() const noexcept {
        return index_.length();
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        if ((content_.length() == 0) && (max_index_ == 0)) { // catches empty object
        } else if (max_index_ >= content_.length()) {
          std::stringstream out;
          out << "IndexedOption node" << id_ << " has index " << max_index_
              << " but content has length "
              << content_.length() << "\n";
          error.append(out.str());

          return false;
        }
        return content_.is_valid(error);
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        names_nbytes["node" + std::to_string(id_) + "-index"] = index_.nbytes();
        content_.buffer_nbytes(names_nbytes);
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        index_.concatenate(reinterpret_cast<PRIMITIVE*>(
            buffers["node" + std::to_string(id_) + "-index"]));
        content_.to_buffers(buffers);
      }

      /// @brief Copies and concatenates the accumulated data in the builder buffer to
      /// a user-defined pointer if the given node name matches with the node associated
      /// with the builder; otherwise, it searches the builder contents to locate a
      /// matching node.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        if (std::string(name) == std::string("node" + std::to_string(id_) + "-index")) {
          index_.concatenate(reinterpret_cast<PRIMITIVE*>(buffer));
        }
        content_.to_buffer(buffer, name);
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        index_.concatenate(reinterpret_cast<PRIMITIVE*>(
            buffers["node" + std::to_string(id_) + "-index"]));
        content_.to_char_buffers(buffers);
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        return "{ \"class\": \"IndexedOptionArray\", \"index\": \"" +
               type_to_numpy_like<PRIMITIVE>() +
               "\", \"content\": " + content_.form() + params +
               ", \"form_key\": \"" + form_key.str() + "\" }";
      }

    private:
      /// @brief Buffer of `PRIMITIVE` type.
      ///
      /// It specifies the index of each element.
      GrowableBuffer<PRIMITIVE> index_;

      /// @brief The content of the IndexedOptionArray.
      BUILDER content_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief Last valid index.
      size_t last_valid_;

      /// @brief Keep track of maximum index value.
      size_t max_index_;
    };

    /// @class Unmasked
    ///
    /// @brief Builds an UnmaskedArray which the values are never, in fact, missing.
    /// It exists to satisfy systems that formally require this high-level type without
    /// the overhead of generating an array of all True or all False values.
    ///
    /// This is similar to NumPy's
    /// [masked arrays](https://numpy.org/doc/stable/reference/maskedarray.html)
    /// with mask=None.
    ///
    /// @tparam BUILDER The type of builder content.
    template <typename BUILDER>
    class Unmasked {
    public:
      /// @brief Creates a new Unmasked layout builder.
      Unmasked() {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns the reference to the builder content.
      BUILDER&
      content() noexcept {
        return content_;
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        content_.set_id(id);
      }

      /// @brief Clears the builder content.
      void
      clear() noexcept {
        content_.clear();
      }

      /// @brief Current length of the content.
      size_t
      length() const noexcept {
        return content_.length();
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        return content_.is_valid(error);
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        content_.buffer_nbytes(names_nbytes);
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        content_.to_buffers(buffers);
      }

      /// @brief Copies and concatenates the accumulated data in the buffers of the
      /// builder content to user-defined pointers if the given node name matches
      /// with the node associated with that builder.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        content_.to_buffer(buffer, name);
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        content_.to_char_buffers(buffers);
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        return "{ \"class\": \"UnmaskedArray\", \"content\": " +
               content_.form() + params + ", \"form_key\": \"" +
               form_key.str() + "\" }";
      }

    private:
      /// @brief The content of the UnmaskedArray.
      BUILDER content_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;
    };

    /// @class ByteMasked
    ///
    /// @brief Builds a ByteMaskedArray using a mask which is an array
    /// of booleans that determines whether the corresponding value in the
    /// contents array is valid or not.
    ///
    /// If an element of the mask is equal to #valid_when, the corresponding
    /// element of the builder content is valid and unmasked, else it is
    /// invalid (missing) and masked.
    ///
    /// This is similar to NumPy's
    /// [masked arrays](https://numpy.org/doc/stable/reference/maskedarray.html)
    /// if #valid_when = `false`.
    ///
    /// @tparam VALID_WHEN A boolean value which determines when the builder content are valid.
    /// @tparam BUILDER The type of builder content.
    template <bool VALID_WHEN, typename BUILDER>
    class ByteMasked {
    public:
      /// @brief Creates a new ByteMasked layout builder by allocating a new `mask`
      ///  buffer, using `AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS` for initializing the buffer.
      ByteMasked()
          : mask_(awkward::GrowableBuffer<int8_t>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Creates a new ByteMasked layout builder by allocating a new `mask`
      /// buffer, taking `options` from {@link BuilderOptions BuilderOptions}
      /// for initializing the buffer.
      ///
      /// @param options Initial size configuration of a buffer.
      ByteMasked(const awkward::BuilderOptions& options)
          : mask_(awkward::GrowableBuffer<int8_t>(options)) {
        size_t id = 0;
        set_id(id);
      }

      /// @brief Returns the reference to the builder content.
      BUILDER&
      content() noexcept {
        return content_;
      }

      /// @brief Determines when the builder content are valid.
      bool
      valid_when() const noexcept {
        return valid_when_;
      }

      /// @brief Inserts #valid_when in the mask.
      ///
      /// After this, a valid element is inserted in the builder content.
      BUILDER&
      append_valid() noexcept {
        mask_.append(valid_when_);
        return content_;
      }

      /// @brief Inserts `size` number of #valid_when in the mask.
      ///
      /// After this, `size` number of valid elements are inserted in the builder content.
      ///
      /// Just an interface; not actually faster than calling append many times.
      BUILDER&
      extend_valid(size_t size) noexcept {
        for (size_t i = 0; i < size; i++) {
          mask_.append(valid_when_);
        }
        return content_;
      }

      /// @brief Inserts !valid_when in the mask.
      ///
      /// After this, a dummy (invalid) value is inserted in the builder content.
      BUILDER&
      append_invalid() noexcept {
        mask_.append(!valid_when_);
        return content_;
      }

      /// @brief Inserts `size` number of !valid_when in the mask.
      ///
      /// After this, `size` number of dummy (invalid) values are inserted in the builder content.
      ///
      /// Just an interface; not actually faster than calling append many times.
      BUILDER&
      extend_invalid(size_t size) noexcept {
        for (size_t i = 0; i < size; i++) {
          mask_.append(!valid_when_);
        }
        return content_;
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        content_.set_id(id);
      }

      /// @brief Discards the accumulated mask and clears the content
      /// of the builder.
      void
      clear() noexcept {
        mask_.clear();
        content_.clear();
      }

      /// @brief Current length of the `mask` buffer.
      size_t
      length() const noexcept {
        return mask_.length();
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        if (content_.length() != mask_.length()) {
          std::stringstream out;
          out << "ByteMasked node" << id_ << "has content length "
              << content_.length() << "but mask length " << mask_.length()
              << "\n";
          error.append(out.str());

          return false;
        } else {
          return content_.is_valid(error);
        }
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        names_nbytes["node" + std::to_string(id_) + "-mask"] = mask_.nbytes();
        content_.buffer_nbytes(names_nbytes);
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        mask_.concatenate(reinterpret_cast<int8_t*>(
            buffers["node" + std::to_string(id_) + "-mask"]));
        content_.to_buffers(buffers);
      }

      /// @brief Copies and concatenates the accumulated data in the builder buffer to
      /// a user-defined pointer if the given node name matches with the node associated
      /// with the builder; otherwise, it searches the builder contents to locate a
      /// matching node.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        if (std::string(name) == std::string("node" + std::to_string(id_) + "-mask")) {
          mask_.concatenate(reinterpret_cast<int8_t*>(buffer));
        }
        content_.to_buffer(buffer, name);
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        mask_.concatenate(reinterpret_cast<int8_t*>(
            buffers["node" + std::to_string(id_) + "-mask"]));
        content_.to_char_buffers(buffers);
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key, form_valid_when;
        form_key << "node" << id_;
        form_valid_when << std::boolalpha << valid_when_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        return "{ \"class\": \"ByteMaskedArray\", \"mask\": \"i8\", "
               "\"content\": " +
               content_.form() + ", \"valid_when\": " + form_valid_when.str() +
               params + ", \"form_key\": \"" + form_key.str() + "\" }";
      }

    private:
      /// @brief Buffer of `int8` type.
      ///
      /// It specifies the mask value of each element.
      GrowableBuffer<int8_t> mask_;

      /// @brief The content of the ByteMaskedArray.
      BUILDER content_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief Determines when the builder content are valid.
      bool valid_when_ = VALID_WHEN;
    };

    /// @class BitMasked
    ///
    /// @brief Builds a BitMaskedArray in which mask values are packed into a bitmap.
    ///
    /// It has an additional parameter, #lsb_order; If `true`, the position of each bit is in
    /// `Least-Significant Bit order (LSB)` and if it is `false`, then in `Most-Significant Bit
    /// order (MSB)`.
    ///
    /// This is similar to NumPy's
    /// [unpackbits](https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html)
    /// with `bitorder="little"` for LSB, `bitorder="big"` for MSB.
    ///
    /// @tparam VALID_WHEN A boolean value which determines when the builder content are valid.
    /// @tparam LSB_ORDER A boolean value which determines whether the position of each bit is
    /// in LSB order or not.
    /// @tparam BUILDER The type of builder content.
    template <bool VALID_WHEN, bool LSB_ORDER, typename BUILDER>
    class BitMasked {
    public:
      /// @brief Creates a new BitMasked layout builder by allocating a new `mask`
      /// buffer, using `AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS` for initializing the buffer.
      BitMasked()
          : mask_(awkward::GrowableBuffer<uint8_t>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)),
            current_byte_(uint8_t(0)),
            current_byte_ref_(mask_.append_and_get_ref(current_byte_)),
            current_index_(0) {
        size_t id = 0;
        set_id(id);
        if (lsb_order_) {
          for (size_t i = 0; i < 8; i++) {
            cast_[i] = 1 << i;
          }
        } else {
          for (size_t i = 0; i < 8; i++) {
            cast_[i] = 128 >> i;
          }
        }
      }

      /// @brief Creates a new BitMasked layout builder by allocating a new `mask`
      /// buffer, taking `options` from {@link BuilderOptions BuilderOptions}
      /// for initializing the buffer.
      ///
      /// @param options Initial size configuration of a buffer.
      BitMasked(const awkward::BuilderOptions& options)
          : mask_(awkward::GrowableBuffer<uint8_t>(options)),
            current_byte_(uint8_t(0)),
            current_byte_ref_(mask_.append_and_get_ref(current_byte_)),
            current_index_(0) {
        size_t id = 0;
        set_id(id);
        if (lsb_order_) {
          for (size_t i = 0; i < 8; i++) {
            cast_[i] = 1 << i;
          }
        } else {
          for (size_t i = 0; i < 8; i++) {
            cast_[i] = 128 >> i;
          }
        }
      }

      /// @brief Returns the reference to the builder content.
      BUILDER&
      content() noexcept {
        return content_;
      }

      /// @brief Determines when the builder content are valid.
      bool
      valid_when() const noexcept {
        return valid_when_;
      }

      /// @brief Determines whether the position of each bit is in
      /// Least-Significant Bit order (LSB) or not.
      bool
      lsb_order() const noexcept {
        return lsb_order_;
      }

      /// @brief Sets a bit in the mask.
      /// If current_byte_ and cast_: `0` indicates `null`, `1` indicates `valid` and vice versa.
      ///
      /// After this, a valid element is inserted in the builder content.
      BUILDER&
      append_valid() noexcept {
        append_begin();
        current_byte_ |= cast_[current_index_];
        append_end();
        return content_;
      }

      /// @brief Sets `size` number of bits in the mask.
      /// If current_byte_ and cast_: `0` indicates `null`, `1` indicates `valid` and vice versa.
      ///
      /// After this, `size` number of valid elements are inserted in the builder content.
      ///
      /// Just an interface; not actually faster than calling append many times.
      BUILDER&
      extend_valid(size_t size) noexcept {
        for (size_t i = 0; i < size; i++) {
          append_valid();
        }
        return content_;
      }

      /// @brief Sets current_byte_ and cast_ default to null, no change in current_byte_.
      ///
      /// After this, a dummy (invalid) value is inserted in the builder content.
      BUILDER&
      append_invalid() noexcept {
        append_begin();
        append_end();
        return content_;
      }

      /// @brief Sets current_byte_ and cast_ default to null, no change in current_byte_.
      ///
      /// After this, `size` number of dummy (invalid) values are inserted in the builder content.
      ///
      /// Just an interface; not actually faster than calling append many times.
      BUILDER&
      extend_invalid(size_t size) noexcept {
        for (size_t i = 0; i < size; i++) {
          append_invalid();
        }
        return content_;
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        content_.set_id(id);
      }

      /// @brief Discards the accumulated mask and clears the content
      /// of the builder.
      void
      clear() noexcept {
        mask_.clear();
        content_.clear();
        current_byte_ = 0;
        current_byte_ref_ = mask_.append_and_get_ref(current_byte_);
        current_index_ = 0;
      }

      /// @brief Current length of the `mask` buffer.
      size_t
      length() const noexcept {
        return mask_.length() > 0 ?
          (mask_.length() - 1) * 8 + current_index_ : current_index_;
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        if (content_.length() != length()) {
          std::stringstream out;
          out << "BitMasked node" << id_ << "has content length "
              << content_.length() << "but bit mask length " << mask_.length()
              << "\n";
          error.append(out.str());

          return false;
        } else {
          return content_.is_valid(error);
        }
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        names_nbytes["node" + std::to_string(id_) + "-mask"] = mask_.nbytes();
        content_.buffer_nbytes(names_nbytes);
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        mask_.concatenate_from(reinterpret_cast<uint8_t*>(
            buffers["node" + std::to_string(id_) + "-mask"]), 0, 1);
        mask_.append(reinterpret_cast<uint8_t*>(
            buffers["node" + std::to_string(id_) + "-mask"]), mask_.length() - 1, 0, 1);
        content_.to_buffers(buffers);
      }

      /// @brief Copies and concatenates the accumulated data in the builder buffer to
      /// a user-defined pointer if the given node name matches with the node associated
      /// with the builder; otherwise, it searches the builder contents to locate a
      /// matching node.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        if (std::string(name) == std::string("node" + std::to_string(id_) + "-mask")) {
          mask_.concatenate_from(reinterpret_cast<uint8_t*>(buffer), 0, 1);
          mask_.append(reinterpret_cast<uint8_t*>(buffer), mask_.length() - 1, 0, 1);
        }
        content_.to_buffer(buffer, name);
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        mask_.concatenate_from(reinterpret_cast<uint8_t*>(
            buffers["node" + std::to_string(id_) + "-mask"]), 0, 1);
        mask_.append(reinterpret_cast<uint8_t*>(
            buffers["node" + std::to_string(id_) + "-mask"]), mask_.length() - 1, 0, 1);
        content_.to_char_buffers(buffers);
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key, form_valid_when, form_lsb_order;
        form_key << "node" << id_;
        form_valid_when << std::boolalpha << valid_when_;
        form_lsb_order << std::boolalpha << lsb_order_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        return "{ \"class\": \"BitMaskedArray\", \"mask\": \"u8\", "
               "\"content\": " +
               content_.form() + ", \"valid_when\": " + form_valid_when.str() +
               ", \"lsb_order\": " + form_lsb_order.str() + params +
               ", \"form_key\": \"" + form_key.str() + "\" }";
      }

    private:
      /// @brief Inserts a byte in the mask buffer when `current_index_` equals `8`,
      /// returns it reference to the `current_byte_ref_` and resets `current_byte_`
      /// and `current_index_`.
      void
      append_begin() {
        if (current_index_ == 8) {
          current_byte_ref_ = mask_.append_and_get_ref(current_byte_);
          current_byte_ = uint8_t(0);
          current_index_ = 0;
        }
      }

      /// @brief Updates the `current_index_` and `current_byte_ref_` according to
      /// the value of #valid_when.
      ///
      /// If #valid_when equals `true`: `0` indicates `null`, `1` indicates `valid`.
      /// If #valid_when equals `false`: `0` indicates `valid`, `1` indicates `null`.
      void
      append_end() {
        current_index_ += 1;
        if (valid_when_) {
          current_byte_ref_ = current_byte_;
        } else {
          current_byte_ref_ = ~current_byte_;
        }
      }

      /// @brief Buffer of `uint8` type.
      ///
      /// It specifies the mask value of each element.
      GrowableBuffer<uint8_t> mask_;

      /// @brief The content of the BitMaskedArray.
      BUILDER content_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief The current byte.
      uint8_t current_byte_;

      /// @brief Stores the reference to the current byte.
      uint8_t& current_byte_ref_;

      /// @brief The current index.
      size_t current_index_;

      /// @brief Array of bytes used to set the bit.
      uint8_t cast_[8];

      /// @brief Determines when the builder content are valid.
      bool valid_when_ = VALID_WHEN;

      /// @brief Determines whether the position of each bit is in
      /// Least-Significant Bit order (LSB) or not.
      bool lsb_order_ = LSB_ORDER;
    };

    /// @class Union
    ///
    /// @brief Builds a UnionArray which represents data drawn from an ordered
    /// list of contents, which can have different types, using tags, which is an
    /// array of integers indicating which content each array element draws from and
    /// index, which is an array of integers indicating which element from the content
    /// to draw from.
    ///
    /// The index values can be 64-bit signed integers `int64`, 32-bit signed integers
    /// `int32` or 32-bit unsigned integers `uint32` and the tags values can be 8-bit
    /// signed integers.
    ///
    /// @tparam TAGS The type of `tags` buffer.
    /// @tparam INDEX The type of `index` buffer.
    /// @tparam BUILDERS The types of builder contents.
    template <typename TAGS, typename INDEX, typename... BUILDERS>
    class Union {
    public:
      using Contents = typename std::tuple<BUILDERS...>;

      template <std::size_t I>
      using ContentType = std::tuple_element_t<I, Contents>;

      /// @brief Creates a new Union layout builder by allocating new tags and
      /// index buffers, using `AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS` for initializing the buffer.
      Union()
          : tags_(awkward::GrowableBuffer<TAGS>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)),
            index_(awkward::GrowableBuffer<INDEX>(AWKWARD_LAYOUTBUILDER_DEFAULT_OPTIONS)) {
        size_t id = 0;
        set_id(id);
        for (size_t i = 0; i < contents_count_; i++) {
          last_valid_index_[i] = -1;
        }
      }

      /// @brief Creates a new Union layout builder by allocating new tags and index
      /// buffers, taking `options` from {@link BuilderOptions BuilderOptions}
      /// for initializing the buffer.
      ///
      /// @param options Initial size configuration of a buffer.
      Union(const awkward::BuilderOptions& options)
          : tags_(awkward::GrowableBuffer<TAGS>(options)),
            index_(awkward::GrowableBuffer<INDEX>(options)) {
        size_t id = 0;
        set_id(id);
        for (size_t i = 0; i < contents_count_; i++) {
          last_valid_index_[i] = -1;
        }
      }

      template <std::size_t I>
      ContentType<I>&
      content() noexcept {
        return std::get<I>(contents_);
      }

      /// @brief Inserts the current tag in the `tags` buffer and the next index in the
      /// `index` buffer and returns the reference to the content of the current builder.
      template <std::size_t TAG>
      ContentType<TAG>&
      append_content() noexcept {
        auto& which_content = std::get<TAG>(contents_);
        INDEX next_index = which_content.length();

        TAGS tag = (TAGS)TAG;
        last_valid_index_[tag] = next_index;
        tags_.append(tag);
        index_.append(next_index);

        return which_content;
      }

      /// @brief Parameters for the builder form.
      const std::string&
      parameters() const noexcept {
        return parameters_;
      }

      /// @brief Sets the form parameters.
      void
      set_parameters(std::string parameter) noexcept {
        parameters_ = parameter;
      }

      /// @brief Assigns a unique ID to each node.
      void
      set_id(size_t& id) noexcept {
        id_ = id;
        id++;
        auto contents_id = [&id](auto& content) {
          content.set_id(id);
        };
        for (size_t i = 0; i < contents_count_; i++) {
          visit_at(contents_, i, contents_id);
        }
      }

      /// @brief Discards the accumulated tags and index, and clears
      /// the builder contents.
      ///
      /// Also, resets the last valid index array to `-1`.
      void
      clear() noexcept {
        for (size_t i = 0; i < contents_count_; i++) {
          last_valid_index_[i] = -1;
        }
        tags_.clear();
        index_.clear();
        auto clear_contents = [](auto& content) {
          content.clear();
        };
        for (size_t i = 0; i < contents_count_; i++) {
          visit_at(contents_, i, clear_contents);
        }
      }

      /// @brief Current length of the `tags` buffer.
      size_t
      length() const noexcept {
        return tags_.length();
      }

      /// @brief Checks for validity and consistency.
      bool
      is_valid(std::string& error) const noexcept {
        auto index_sequence((std::index_sequence_for<BUILDERS...>()));

        std::vector<size_t> lengths = content_lengths(index_sequence);
        std::unique_ptr<INDEX[]> index_ptr(new INDEX[index_.length()]);
        index_.concatenate(index_ptr.get());
        std::unique_ptr<TAGS[]> tags_ptr(new TAGS[tags_.length()]);
        tags_.concatenate(tags_ptr.get());
        for (size_t i = 0; i < index_.length(); i++) {
          if (index_ptr.get()[i] < 0 || index_ptr.get()[i] >= lengths[tags_ptr.get()[i]]) {
            std::stringstream out;
            out << "Union node" << id_ << " has index " << index_ptr.get()[i]
                << " at position " << i << " but content has length "
                << lengths[tags_ptr.get()[i]] << "\n";
            error.append(out.str());

            return false;
          }
        }

        std::vector<bool> valid_contents =
            content_is_valid(index_sequence, error);
        return std::none_of(std::cbegin(valid_contents),
                            std::cend(valid_contents),
                            std::logical_not<bool>());
      }

      /// @brief Retrieves the names and sizes (in bytes) of the buffers used
      /// in the builder and its contents.
      void
      buffer_nbytes(std::map<std::string, size_t>& names_nbytes) const
          noexcept {
        auto index_sequence((std::index_sequence_for<BUILDERS...>()));

        names_nbytes["node" + std::to_string(id_) + "-tags"] = tags_.nbytes();
        names_nbytes["node" + std::to_string(id_) + "-index"] = index_.nbytes();

        for (size_t i = 0; i < contents_count_; i++) {
          visit_at(contents_, i, [&names_nbytes](auto& content) {
            content.buffer_nbytes(names_nbytes);
          });
        }
      }

      /// @brief Copies and concatenates all the accumulated data in each of the
      /// buffers of the builder and its contents to user-defined pointers.
      ///
      /// Used to fill the buffers map by allocating it with user-defined pointers
      /// using the same names and sizes (in bytes) obtained from #buffer_nbytes.
      void
      to_buffers(std::map<std::string, void*>& buffers) const noexcept {
        auto index_sequence((std::index_sequence_for<BUILDERS...>()));

        tags_.concatenate(reinterpret_cast<TAGS*>(
            buffers["node" + std::to_string(id_) + "-tags"]));
        index_.concatenate(reinterpret_cast<INDEX*>(
            buffers["node" + std::to_string(id_) + "-index"]));

        for (size_t i = 0; i < contents_count_; i++) {
          visit_at(contents_, i, [&buffers](auto& content) {
            content.to_buffers(buffers);
          });
        }
      }

      /// @brief Copies and concatenates the accumulated data in the builder buffers to
      /// user-defined pointers if the given node name matches with any one of the nodes
      /// associated with the builder; otherwise, it searches the builder contents to
      /// locate a matching node.
      void
      to_buffer(void* buffer, const char* name) const noexcept {
        auto index_sequence((std::index_sequence_for<BUILDERS...>()));

        if (std::string(name) == std::string("node" + std::to_string(id_) + "-tags")) {
          tags_.concatenate(reinterpret_cast<TAGS*>(buffer));
        }
        else if (std::string(name) == std::string("node" + std::to_string(id_) + "-index")) {
          index_.concatenate(reinterpret_cast<INDEX*>(buffer));
        }

        for (size_t i = 0; i < contents_count_; i++) {
          visit_at(contents_, i, [&buffer, &name](auto& content) {
            content.to_buffer(buffer, name);
          });
        }
      }

      /// @brief Copies and concatenates all the accumulated data in the builder
      /// to a map of user-allocated buffers.
      ///
      /// The map keys and the buffer sizes are obtained from #buffer_nbytes
      void
      to_char_buffers(std::map<std::string, uint8_t*>& buffers) const noexcept {
        auto index_sequence((std::index_sequence_for<BUILDERS...>()));

        tags_.concatenate(reinterpret_cast<TAGS*>(
            buffers["node" + std::to_string(id_) + "-tags"]));
        index_.concatenate(reinterpret_cast<INDEX*>(
            buffers["node" + std::to_string(id_) + "-index"]));

        for (size_t i = 0; i < contents_count_; i++) {
          visit_at(contents_, i, [&buffers](auto& content) {
            content.to_char_buffers(buffers);
          });
        }
      }

      /// @brief Generates a unique description of the builder and its
      /// contents in the form of a JSON-like string.
      std::string
      form() const noexcept {
        std::stringstream form_key;
        form_key << "node" << id_;
        std::string params("");
        if (parameters_ == "") {
        } else {
          params = std::string(", \"parameters\": { " + parameters_ + " }");
        }
        std::stringstream out;
        out << "{ \"class\": \"UnionArray\", \"tags\": \"" +
                   type_to_numpy_like<TAGS>() + "\", \"index\": \"" +
                   type_to_numpy_like<INDEX>() + "\", \"contents\": [";
        for (size_t i = 0; i < contents_count_; i++) {
          if (i != 0) {
            out << ", ";
          }
          auto contents_form = [&](auto& content) {
            out << content.form();
          };
          visit_at(contents_, i, contents_form);
        }
        out << "], ";
        out << params << "\"form_key\": \"" << form_key.str() << "\" }";
        return out.str();
      }

    private:
      /// @brief Inserts the lengths of all the builder contents in a vector
      /// and returns the vector.
      template <std::size_t... S>
      std::vector<size_t>
      content_lengths(std::index_sequence<S...>) const {
        return std::vector<size_t>({std::get<S>(contents_).length()...});
      }

      /// @brief Helper function of #is_valid for checking the validity.
      template <std::size_t... S>
      std::vector<bool>
      content_is_valid(std::index_sequence<S...>, std::string& error) const {
        return std::vector<bool>({std::get<S>(contents_).is_valid(error)...});
      }

      /// @brief Buffer of `TAGS` type.
      ///
      /// It specifies the index of each Tuple content.
      GrowableBuffer<TAGS> tags_;

      /// @brief Buffer of `INDEX` type.
      ///
      /// It specifies the index of each element.
      GrowableBuffer<INDEX> index_;

      /// @brief The contents of the UnionArray.
      Contents contents_;

      /// @brief Form parameters.
      std::string parameters_;

      /// @brief Unique form ID.
      size_t id_;

      /// @brief Last valid index array.
      size_t last_valid_index_[sizeof...(BUILDERS)];

      /// @brief The count of the Tuple contents in the builder.
      static constexpr size_t contents_count_ = sizeof...(BUILDERS);
    };

  }  // namespace LayoutBuilder
}  // namespace awkward

#endif  // AWKWARD_LAYOUTBUILDER_H_
