#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TInterpreter.h"

void read_jagged3_root() {
  gInterpreter->GenerateDictionary("vector<vector<vector<float> > >", "vector");

  std::vector<std::vector<std::vector<float>>>* data3;
  data3 = nullptr;

  TFile* f = new TFile("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-jagged3.root");
  TTree* t;
  f->GetObject("tree", t);

  TBranch* b;
  b = nullptr;
  t->SetBranchAddress("branch", &data3, &b);

  std::vector<int32_t> offsets0;
  std::vector<int32_t> offsets1;
  std::vector<int32_t> offsets2;
  std::vector<float> content;

  offsets0.reserve(1024);
  offsets1.reserve(1024);
  offsets2.reserve(1024);
  content.reserve(1024);

  offsets0.push_back(0);
  offsets1.push_back(0);
  offsets2.push_back(0);

  t->GetEntries();
  int64_t num_entries = 85522;  // first 3 baskets

  auto begin_time = std::chrono::high_resolution_clock::now();

  for (int64_t i = 0;  i < num_entries;  i++) {
    b->GetEntry(i);

    offsets0.push_back(offsets0.back() + data3->size());
    for (auto x : *data3) {
      offsets1.push_back(offsets1.back() + x.size());
      for (auto y : x) {
        offsets2.push_back(offsets2.back() + y.size());
        std::copy(y.begin(), y.end(), std::back_inserter(content));
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  int64_t count_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - begin_time
  ).count();

  std::cout << "time: " << count_nanoseconds << " nanoseconds" << std::endl;

  // std::cout << offsets0[0] << " " << offsets0[1] << " " << offsets0[2] << " ... " << offsets0[offsets0.size() - 1] << std::endl;
  // std::cout << offsets1[0] << " " << offsets1[1] << " " << offsets1[2] << " ... " << offsets1[offsets1.size() - 1] << std::endl;
  // std::cout << offsets2[0] << " " << offsets2[1] << " " << offsets2[2] << " ... " << offsets2[offsets2.size() - 1] << std::endl;
  // std::cout << content[0] << " " << content[1] << " " << content[2] << " ... " << content[content.size() - 1] << std::endl;

}
