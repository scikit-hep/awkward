// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_DATETIMEBUILDER_H_
#define AWKWARD_DATETIMEBUILDER_H_

#include "awkward/common.h"
#include "awkward/BuilderOptions.h"
#include "awkward/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {

  /// @class DatetimeBuilder
  ///
  /// @brief Builder node that accumulates integers (`int64_t`).
  class EXPORT_SYMBOL DatetimeBuilder: public Builder {
  public:
    /// @brief Create an empty DatetimeBuilder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const BuilderOptions& options, const std::string& units);

    /// @brief Create an DatetimeBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param buffer Contains the accumulated integers.
    DatetimeBuilder(const BuilderOptions& options,
                    GrowableBuffer<int64_t> content,
                    const std::string& units);

    /// @brief User-friendly name of this class: `"DatetimeBuilder"`.
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
    /// A DatetimeBuilder is never active.
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

    const std::string&
      units() const;

    const GrowableBuffer<int64_t>& buffer() const { return content_; }

    const std::string& unit() const { return units_; }

  private:
    const BuilderOptions options_;
    GrowableBuffer<int64_t> content_;
    const std::string units_;
  };
}

#endif // AWKWARD_DATETIMEBUILDER_H_
