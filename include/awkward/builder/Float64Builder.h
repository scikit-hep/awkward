// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FLOAT64BUILDER_H_
#define AWKWARD_FLOAT64BUILDER_H_

#include "awkward/common.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  /// @class Float64Builder
  ///
  /// @brief Builder node that accumulates real numbers (`double`).
  class LIBAWKWARD_EXPORT_SYMBOL Float64Builder: public Builder {
  public:
    /// @brief Create an empty Float64Builder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const ArrayBuilderOptions& options);

    /// @brief Create a Float64Builder from an existing Int64Builder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param old The Int64Builder's buffer.
    static const BuilderPtr
      fromint64(const ArrayBuilderOptions& options,
                const GrowableBuffer<int64_t>& old);

    /// @brief Create a Float64Builder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param buffer Contains the accumulated real numbers.
    Float64Builder(const ArrayBuilderOptions& options,
                   const GrowableBuffer<double>& buffer);

    /// @brief User-friendly name of this class: `"Float64Builder"`.
    const std::string
      classname() const override;

    int64_t
      length() const override;

    void
      clear() override;

    const ContentPtr
      snapshot() const override;

    /// @copydoc Builder::active()
    ///
    /// A Float64Builder is never active.
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

    const BuilderPtr
      field(const char* key, bool check) override;

    const BuilderPtr
      endrecord() override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;

  private:
    const ArrayBuilderOptions options_;
    GrowableBuffer<double> buffer_;
  };

}

#endif // AWKWARD_FLOAT64BUILDER_H_
