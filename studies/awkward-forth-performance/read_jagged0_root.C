#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TInterpreter.h"

void read_jagged0_root() {
  float data;
  data = 3.14;

  TFile* f = new TFile("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-jagged0.root");
  TTree* t;
  f->GetObject("tree", t);

  TBranch* b;
  b = nullptr;
  t->SetBranchAddress("branch", &data, &b);

  auto begin_time = std::chrono::high_resolution_clock::now();

  int64_t num_entries = b->GetEntries();
  int64_t num_baskets = b->GetWriteBasket();
  Long64_t* basket_starts = b->GetBasketEntry();
  for (int64_t basketid = 0;  basketid < num_baskets;  basketid++) {
    int64_t start = basket_starts[basketid];
    int64_t stop = num_entries;
    if (basketid + 1 < num_baskets) {
      stop = basket_starts[basketid + 1];
    }

    std::vector<float> content;

    content.reserve(1024);

    for (int64_t i = start;  i < stop;  i++) {
      b->GetEntry(i);

      content.push_back(data);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  int64_t count_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - begin_time
  ).count();

  std::cout << "ROOT zlib0-jagged0.root " << num_entries << " entries " << (count_nanoseconds / 1e9) << " seconds" << std::endl;
}
