// export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
// g++ `root-config --cflags` -Iawkward-1.0/include -Iawkward-1.0/rapidjson/include chep2019-studies-3.cpp -L. -lawkward `root-config --libs` -lROOTNTuple -o chep2019-studies-3 && ./chep2019-studies-3

#include <cstring>
#include <chrono>
#include <iostream>

#include "awkward/Identity.h"
#include "awkward/array/RawArray.h"
#include "awkward/array/ListOffsetArray.h"

#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RNTupleMetrics.hxx"
#include "ROOT/RNTupleOptions.hxx"
#include "ROOT/RNTupleUtil.hxx"
#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleView.hxx"
#include "ROOT/RNTupleDS.hxx"
#include "ROOT/RNTupleDescriptor.hxx"

#define LENJAGGED0 1073741824
#define LENJAGGED1 134217728
#define LENJAGGED2 16777216
#define LENJAGGED3 2097152

#define kDefaultClusterSizeEntries 64000

namespace ak = awkward;

template <typename V, typename T>
void fillpages(T* array, V& view, int64_t& offset, int64_t length, int64_t shift) {
  int64_t current = 0;
  while (current < length) {
    T* data = (T*)view.fField.Map(offset + current);
    int32_t num = view.fField.fPrincipalColumn->fCurrentPage.GetNElements();
    int32_t skipped = (offset + current) - view.fField.fPrincipalColumn->fCurrentPage.GetGlobalRangeFirst();
    int32_t remaining = num - skipped;
    if (current + remaining > length) {
      remaining = length - current;
    }
    if (remaining > 0) {
      memcpy(&array[current + shift], data, remaining*sizeof(T));
    }
    current += remaining;
  }
  offset += current;
}

double jagged0() {
  double total_length = 0.0;

  auto model = ROOT::Experimental::RNTupleModel::Create();
  ROOT::Experimental::RNTupleReadOptions options;
  auto ntuple = ROOT::Experimental::RNTupleReader::Open(std::move(model), "jagged0", "data/sample-jagged0.ntuple", options);
  auto view0 = ntuple->GetViewCollection("field");

  int64_t offset0 = 0;
  for (uint64_t entry = 0;  entry < LENJAGGED0;  entry += kDefaultClusterSizeEntries) {
    int64_t length = kDefaultClusterSizeEntries;
    if (entry + length > LENJAGGED0) {
      length = LENJAGGED0 - entry;
    }
    ak::RawArrayOf<float> content(ak::Identity::none(), length);
    float* rawcontent = content.ptr().get();
    fillpages(rawcontent, view0, offset0, length, 0);
    total_length += length;
  }
  std::cout << total_length << std::endl;
  return total_length;
}

double jagged1() {
  double total_length = 0.0;

  auto model = ROOT::Experimental::RNTupleModel::Create();
  ROOT::Experimental::RNTupleReadOptions options;
  auto ntuple = ROOT::Experimental::RNTupleReader::Open(std::move(model), "jagged1", "data/sample-jagged1.ntuple", options);
  auto view1 = ntuple->GetViewCollection("field");
  auto view0 = view1.GetView<float>("float");

  int64_t offset1 = 0;
  int64_t offset0 = 0;
  for (int64_t entry = 0;  entry < LENJAGGED1;  entry += kDefaultClusterSizeEntries) {
    int64_t length = kDefaultClusterSizeEntries;
    if (entry + length > LENJAGGED1) {
      length = LENJAGGED1 - entry;
    }
    ak::Index32 offsets1(length + 1);
    int32_t* rawoffsets1 = offsets1.ptr().get();
    rawoffsets1[0] = 0;
    fillpages(rawoffsets1, view1, offset1, length, 1);

    length = rawoffsets1[length];
    ak::RawArrayOf<float> content(ak::Identity::none(), length);
    float* rawcontent = content.borrow(0);
    fillpages(rawcontent, view0, offset0, length, 0);

    ak::ListOffsetArray32 done(ak::Identity::none(), offsets1, content.shallow_copy());
    total_length += length;
  }

  std::cout << total_length << std::endl;
  return total_length;
}

double jagged2() {
  double total_length = 0.0;

  auto model = ROOT::Experimental::RNTupleModel::Create();
  ROOT::Experimental::RNTupleReadOptions options;
  auto ntuple = ROOT::Experimental::RNTupleReader::Open(std::move(model), "jagged2", "data/sample-jagged2.ntuple", options);
  auto view2 = ntuple->GetViewCollection("field");
  auto view1 = view2.GetViewCollection("std::vector<float>");
  auto view0 = view1.GetView<float>("float");

  int64_t offset2 = 0;
  int64_t offset1 = 0;
  int64_t offset0 = 0;
  for (int64_t entry = 0;  entry < LENJAGGED2;  entry += kDefaultClusterSizeEntries) {
    int64_t length = kDefaultClusterSizeEntries;
    if (entry + length > LENJAGGED2) {
      length = LENJAGGED2 - entry;
    }
    ak::Index32 offsets2(length + 1);
    int32_t* rawoffsets2 = offsets2.ptr().get();
    rawoffsets2[0] = 0;
    fillpages(rawoffsets2, view2, offset2, length, 1);

    length = rawoffsets2[length];
    ak::Index32 offsets1(length + 1);
    int32_t* rawoffsets1 = offsets1.ptr().get();
    rawoffsets1[0] = 0;
    fillpages(rawoffsets1, view1, offset1, length, 1);

    length = rawoffsets1[length];
    ak::RawArrayOf<float> content(ak::Identity::none(), length);
    float* rawcontent = content.borrow(0);
    fillpages(rawcontent, view0, offset0, length, 0);

    ak::ListOffsetArray32 tmp(ak::Identity::none(), offsets1, content.shallow_copy());
    ak::ListOffsetArray32 done(ak::Identity::none(), offsets2, tmp.shallow_copy());
    total_length += length;
  }

  std::cout << total_length << std::endl;
  return total_length;
}

double jagged3() {
  double total_length = 0.0;

  auto model = ROOT::Experimental::RNTupleModel::Create();
  ROOT::Experimental::RNTupleReadOptions options;
  auto ntuple = ROOT::Experimental::RNTupleReader::Open(std::move(model), "jagged3", "data/sample-jagged3.ntuple", options);
  auto view3 = ntuple->GetViewCollection("field");
  auto view2 = view3.GetViewCollection("std::vector<std::vector<float>>");
  auto view1 = view2.GetViewCollection("std::vector<float>");
  auto view0 = view1.GetView<float>("float");

  int64_t offset3 = 0;
  int64_t offset2 = 0;
  int64_t offset1 = 0;
  int64_t offset0 = 0;
  for (int64_t entry = 0;  entry < LENJAGGED3;  entry += kDefaultClusterSizeEntries) {
    int64_t length = kDefaultClusterSizeEntries;
    if (entry + length > LENJAGGED3) {
      length = LENJAGGED3 - entry;
    }
    ak::Index32 offsets3(length + 1);
    int32_t* rawoffsets3 = offsets3.ptr().get();
    rawoffsets3[0] = 0;
    fillpages(rawoffsets3, view3, offset3, length, 1);

    length = rawoffsets3[length];
    ak::Index32 offsets2(length + 1);
    int32_t* rawoffsets2 = offsets2.ptr().get();
    rawoffsets2[0] = 0;
    fillpages(rawoffsets2, view2, offset2, length, 1);

    length = rawoffsets2[length];
    ak::Index32 offsets1(length + 1);
    int32_t* rawoffsets1 = offsets1.ptr().get();
    rawoffsets1[0] = 0;
    fillpages(rawoffsets1, view1, offset1, length, 1);

    length = rawoffsets1[length];
    ak::RawArrayOf<float> content(ak::Identity::none(), length);
    float* rawcontent = content.borrow(0);
    fillpages(rawcontent, view0, offset0, length, 0);

    ak::ListOffsetArray32 tmp1(ak::Identity::none(), offsets1, content.shallow_copy());
    ak::ListOffsetArray32 tmp2(ak::Identity::none(), offsets2, tmp1.shallow_copy());
    ak::ListOffsetArray32 done(ak::Identity::none(), offsets3, tmp2.shallow_copy());
    total_length += length;
  }

  std::cout << total_length << std::endl;
  return total_length;
}

int main() {
  {
    auto start0 = std::chrono::high_resolution_clock::now();
    double num0 = jagged0();
    auto stop0 = std::chrono::high_resolution_clock::now();
    double walltime0 = std::chrono::duration<double>(stop0 - start0).count();
    std::cout << "jagged0 " << walltime0 << "sec;\t" << num0/walltime0/1e6 << " million floats/sec" << std::endl;
  }
  {
    auto start1 = std::chrono::high_resolution_clock::now();
    double num1 = jagged1();
    auto stop1 = std::chrono::high_resolution_clock::now();
    double walltime1 = std::chrono::duration<double>(stop1 - start1).count();
    std::cout << "jagged1 " << walltime1 << "sec;\t" << num1/walltime1/1e6 << " million floats/sec" << std::endl;
  }
  {
    auto start2 = std::chrono::high_resolution_clock::now();
    double num2 = jagged2();
    auto stop2 = std::chrono::high_resolution_clock::now();
    double walltime2 = std::chrono::duration<double>(stop2 - start2).count();
    std::cout << "jagged2 " << walltime2 << "sec;\t" << num2/walltime2/1e6 << " million floats/sec" << std::endl;
  }
  {
    auto start3 = std::chrono::high_resolution_clock::now();
    double num3 = jagged3();
    auto stop3 = std::chrono::high_resolution_clock::now();
    double walltime3 = std::chrono::duration<double>(stop3 - start3).count();
    std::cout << "jagged3 " << walltime3 << "sec;\t" << num3/walltime3/1e6 << " million floats/sec" << std::endl;
  }
  return 0;
}
