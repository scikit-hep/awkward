// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_OPTIONBUILDER_H_
#define AWKWARD_OPTIONBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/BuilderOptions.h"
#include "awkward/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {

  /// @class OptionBuilder
  ///
  /// @brief Builder node that accumulates data with missing values (`None`).
  class EXPORT_SYMBOL OptionBuilder: public Builder {
  public:
    /// @brief Create an OptionBuilder from a number of nulls (all missing).
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param nullcount Length of the purely missing data to create.
    /// @param content Builder for the non-missing data.
    static const BuilderPtr
      fromnulls(const BuilderOptions& options,
                int64_t nullcount,
                const BuilderPtr& content);

    /// @brief Create an OptionBuilder from an existing builder (all
    /// non-missing).
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param content Builder for the non-missing data.
    static const BuilderPtr
      fromvalids(const BuilderOptions& options,
                 const BuilderPtr& content);

    /// @brief Create a OptionBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param index Contains the accumulated index (like
    /// {@link IndexedArrayOf#index IndexedOptionArray::index}).
    /// @param content Builder for the non-missing data.
    OptionBuilder(const BuilderOptions& options,
                  GrowableBuffer<int64_t> index,
                  const BuilderPtr content);

    /// @brief User-friendly name of this class: `"OptionBuilder"`.
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
    /// An OptionBuilder is active if and only if its `content` is active.
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

    const GrowableBuffer<int64_t>& buffer() const { return index_; }

    const GrowableBuffer<int64_t>& index() { return index_; }

    const BuilderPtr builder() const { return content_; }

    void
      maybeupdate(const BuilderPtr builder);

  private:
    GrowableBuffer<int64_t> index_;
    BuilderPtr content_;
  };

}

#endif // AWKWARD_OPTIONBUILDER_H_
