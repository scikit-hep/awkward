#include <stdio.h>
#include <iostream>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

void make_jagged0_root(int64_t level) {
  int64_t events_per_basket = 64 * 1024 * 1024 / sizeof(float);

  std::string name = std::string("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib") + std::to_string(level) + "-jagged0.root";
  auto f = new TFile(name.c_str(), "RECREATE");
  f->SetCompressionAlgorithm(1);
  f->SetCompressionLevel(level);

  auto t = new TTree("tree", "");

  float data0;
  t->Branch("branch", &data0, 64*1024*1024);
  t->SetAutoFlush(0);
  t->SetAutoSave(0);

  FILE* content = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", "r");

  int64_t count = 0;
  while (fread(&data0, sizeof(float), 1, content) != 0) {
    t->Fill();
    count++;
    // if (count % events_per_basket == 0) {
    //   t->Write();
    // }
    // if (count % 10000 == 0) {
    //   std::cout << ((double)count / (3.0 * (double)events_per_basket)) << std::endl;
    // }
    // if (count == 3 * events_per_basket) {
    //   break;
    // }
  }

  t->Write();
  f->Close();
}
