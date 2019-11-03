// c++ --std=c++14 -I../../jblomer-root-build/include -L../../jblomer-root-build/lib -L/home/pivarski/miniconda3/lib -L/usr/lib/x86_64-linux-gnu -lc -lpcre -llz4 -lzstd -lsqlite3 -lssl -ltbb -lcairo -lrt -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -pthread -lm -ldl -rdynamic -lROOTNTuple write_rntuple.cpp -o write_rntuple && ./write_rntuple

// g++ `../../jblomer-root-build/bin/root-config --cflags --libdir --incdir --libs` write_rntuple.cpp -o write_rntuple && ./write_rntuple

// g++ -pthread -std=c++14 -m64 -I../../jblomer-root-build/include -L../../jblomer-root-build/lib -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lROOTNTuple -pthread -lm -ldl -rdynamic write_rntuple.cpp -o write_rntuple && ./write_rntuple

#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RNTupleMetrics.hxx"
#include "ROOT/RNTupleOptions.hxx"
#include "ROOT/RNTupleUtil.hxx"
#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleView.hxx"
#include "ROOT/RNTupleDS.hxx"
#include "ROOT/RNTupleDescriptor.hxx"

#include <iostream>

int main() {
  std::cout << "ONE" << std::endl;

  auto model = ROOT::Experimental::RNTupleModel::Create();

  std::cout << "TWO" << std::endl;

  auto fldData = model->MakeField<float>("data");

  std::cout << "THREE" << std::endl;

  ROOT::Experimental::RNTupleWriteOptions options;
  options.SetCompression(0);

  std::cout << "FOUR" << std::endl;

  auto ntuple = ROOT::Experimental::RNTupleWriter::Recreate(std::move(model), "ntuple", "sample-jagged0.ntuple", options);

  std::cout << "FIVE" << std::endl;

  *fldData = 3.14;

  std::cout << "SIX" << std::endl;

  ntuple->Fill();

  std::cout << "SEVEN" << std::endl;

  return 0;
}
