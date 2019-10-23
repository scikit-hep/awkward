// c++ --std=c++14 -I../../jblomer-root-build/include -L../../jblomer-root-build/lib -L/home/pivarski/miniconda3/lib -L/usr/lib/x86_64-linux-gnu -lc -lpcre -llz4 -lzstd -lCore -lRIO -lThread -pthread -lROOTNTuple write_rntuple.cpp -o write_rntuple && ./write_rntuple

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
  auto model = ROOT::Experimental::RNTupleModel::Create();

  std::cout << "yay" << std::endl;

  return 0;
}
