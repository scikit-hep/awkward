// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_TUPLEBUILDER_H_
#define AWKWARD_TUPLEBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/BuilderOptions.h"
#include "awkward/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {

  /// @class TupleBuilder
  ///
  /// @brief Builder node for accumulated tuples.
  class EXPORT_SYMBOL TupleBuilder: public Builder {
  public:
    /// @brief Create an empty TupleBuilder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const BuilderOptions& options);

    /// @brief Create a TupleBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param contents A Builder for each tuple field.
    /// @param length Length of accumulated array (same as #length).
    /// @param begun If `true`, the TupleBuilder is in an active state;
    /// `false` otherwise.
    /// @param nextindex The next field index to fill with data.
    TupleBuilder(const BuilderOptions& options,
                 const std::vector<BuilderPtr>& contents,
                 int64_t length,
                 bool begun,
                 size_t nextindex);

    /// @brief Current number of fields.
    int64_t
      numfields() const;

    /// @brief User-friendly name of this class: `"TupleBuilder"`.
    const std::string
      classname() const override;

    const std::string
      to_buffers(BuffersContainer& container, int64_t& form_key_id) const override;

    int64_t
      length() const override;

    void
      clear() override;

    /// @copydoc Builder::active()
    ///
    /// Calling #begintuple makes a TupleBuilder active; #endtuple makes it
    /// inactive.
    bool
      active() const override;

    const BuilderPtr
      null() override;

    const BuilderPtr
      boolean(bool x) override;

    const BuilderPtr
      integer(int64_t x) override;

    const BuilderPtr
      real(double x) override;

    const BuilderPtr
      complex(std::complex<double> x) override;

    const BuilderPtr
      datetime(int64_t x, const std::string& unit) override;

    const BuilderPtr
      timedelta(int64_t x, const std::string& unit) override;

    const BuilderPtr
      string(const char* x, int64_t length, const char* encoding) override;

    const BuilderPtr
      beginlist() override;

    const BuilderPtr
      endlist() override;

    const BuilderPtr
      begintuple(int64_t numfields) override;

    const BuilderPtr
      index(int64_t index) override;

    const BuilderPtr
      endtuple() override;

    const BuilderPtr
      beginrecord(const char* name, bool check) override;

    void
      field(const char* key, bool check) override;

    const BuilderPtr
      endrecord() override;

    const BuilderOptions&
      options() const { return options_; }

    const std::vector<BuilderPtr>& contents() const { return contents_; }

    bool begun() { return begun_; }

    int64_t nextindex() { return nextindex_; }

    void
      maybeupdate(int64_t i, const BuilderPtr builder);

  private:
    const BuilderOptions options_;
    std::vector<BuilderPtr> contents_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;
  };
}

#endif // AWKWARD_TUPLEBUILDER_H_
