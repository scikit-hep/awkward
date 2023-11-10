// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_COMPLEX128BUILDER_H_
#define AWKWARD_COMPLEX128BUILDER_H_

#include "awkward/common.h"
#include "awkward/BuilderOptions.h"
#include "awkward/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

#include <complex>

namespace awkward {
  /// @class Complex128Builder
  ///
  /// @brief Builder node that accumulates real numbers (`double`).
  class EXPORT_SYMBOL Complex128Builder: public Builder {
  public:
    /// @brief Create an empty Complex128Builder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const BuilderOptions& options);

    /// @brief Create a Complex128Builder from an existing Int64Builder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param old The Int64Builder's buffer.
    static const BuilderPtr
      fromint64(const BuilderOptions& options,
                const GrowableBuffer<int64_t>& old);

    /// @brief Create a Complex128Builder from an existing Float64Builder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param old The Float64Builder's buffer.
    static const BuilderPtr
      fromfloat64(const BuilderOptions& options,
                  const GrowableBuffer<double>& old);

    /// @brief Create a Complex128Builder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param buffer Contains the accumulated real numbers.
    Complex128Builder(const BuilderOptions& options,
                      GrowableBuffer<std::complex<double>> buffer);

    /// @brief User-friendly name of this class: `"Complex128Builder"`.
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
    /// A Complex128Builder is never active.
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

    const GrowableBuffer<std::complex<double>>& buffer() const { return buffer_; }

  private:
    const BuilderOptions options_;
    GrowableBuffer<std::complex<double>> buffer_;
  };

}

#endif // AWKWARD_COMPLEX128BUILDER_H_
