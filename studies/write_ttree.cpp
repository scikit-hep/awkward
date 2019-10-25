#include "TInterpreter.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include <iostream>
#include <vector>

#define LENCONTENT 1073741824
#define LENOFFSETS1 134217729
#define LENOFFSETS2 16777217
#define LENOFFSETS3 2097153

void getdata(float* content, int64_t* offsets1, int64_t* offsets2, int64_t* offsets3) {
  FILE* f;

  f = fopen("sample-content.float32", "rb");
  fread(content, sizeof(float), LENCONTENT, f);
  fclose(f);

  std::cout << "content  " << content[0] << " ... " << content[LENCONTENT - 1] << std::endl;

  f = fopen("sample-offsets1.int64", "rb");
  fread(offsets1, sizeof(int64_t), LENOFFSETS1, f);
  fclose(f);

  std::cout << "offsets1 " << offsets1[0] << " ... " << offsets1[LENOFFSETS1 - 1] << std::endl;

  f = fopen("sample-offsets2.int64", "rb");
  fread(offsets2, sizeof(int64_t), LENOFFSETS2, f);
  fclose(f);

  std::cout << "offsets2 " << offsets2[0] << " ... " << offsets2[LENOFFSETS2 - 1] << std::endl;

  f = fopen("sample-offsets3.int64", "rb");
  fread(offsets3, sizeof(int64_t), LENOFFSETS3, f);
  fclose(f);

  std::cout << "offsets3 " << offsets3[0] << " ... " << offsets3[LENOFFSETS3 - 1] << std::endl;
}

void ttree0(float* content, int64_t* offsets1, int64_t* offsets2, int64_t* offsets3) {
  std::cout << "starting tree0" << std::endl;
  float jagged0;
  TFile* file0 = new TFile("sample-jagged0.root", "RECREATE");
  file0->SetCompressionLevel(0);
  TTree* tree0 = new TTree("jagged0", "");
  tree0->Branch("branch", &jagged0, 10485760);
  for (int64_t i0 = 0;  i0 < LENCONTENT;  i0++) {
    jagged0 = content[i0];
    tree0->Fill();
  }
  tree0->Write();
  file0->Close();
}

void ttree1(float* content, int64_t* offsets1, int64_t* offsets2, int64_t* offsets3) {
  std::cout << "starting tree1" << std::endl;
  gInterpreter->GenerateDictionary("vector<float>", "vector");
  std::vector<float> jagged1;
  TFile* file1 = new TFile("sample-jagged1.root", "RECREATE");
  file1->SetCompressionLevel(0);
  TTree* tree1 = new TTree("jagged1", "");
  tree1->Branch("branch", &jagged1, 10485760);
  for (int64_t i1 = 0;  i1 < LENOFFSETS1 - 1;  i1++) {
    jagged1.clear();
    int64_t start1 = offsets1[i1];
    int64_t stop1 = offsets1[i1 + 1];
    for (int64_t i0 = start1;  i0 < stop1;  i0++) {
      jagged1.push_back(content[i0]);
    }
    tree1->Fill();
  }
  tree1->Write();
  file1->Close();
}

void ttree2(float* content, int64_t* offsets1, int64_t* offsets2, int64_t* offsets3) {
  std::cout << "starting tree2" << std::endl;
  gInterpreter->GenerateDictionary("vector<vector<float> >", "vector");
  std::vector<std::vector<float>> jagged2;
  TFile* file2 = new TFile("sample-jagged2.root", "RECREATE");
  file2->SetCompressionLevel(0);
  TTree* tree2 = new TTree("jagged2", "");
  tree2->Branch("branch", &jagged2, 10485760);
  for (int64_t i2 = 0;  i2 < LENOFFSETS2 - 1;  i2++) {
    jagged2.clear();
    int64_t start2 = offsets2[i2];
    int64_t stop2 = offsets2[i2 + 1];
    for (int64_t i1 = start2;  i1 < stop2;  i1++) {
      std::vector<float> tmp1;
      int64_t start1 = offsets1[i1];
      int64_t stop1 = offsets1[i1 + 1];
      for (int64_t i0 = start1;  i0 < stop1;  i0++) {
        tmp1.push_back(content[i0]);
      }
      jagged2.push_back(tmp1);
    }
    tree2->Fill();
  }
  tree2->Write();
  file2->Close();
}

void ttree3(float* content, int64_t* offsets1, int64_t* offsets2, int64_t* offsets3) {
  std::cout << "starting tree3" << std::endl;
  gInterpreter->GenerateDictionary("vector<vector<vector<float> > >", "vector");
  std::vector<std::vector<std::vector<float>>> jagged3;
  TFile* file3 = new TFile("sample-jagged3.root", "RECREATE");
  TTree* tree3 = new TTree("jagged3", "");
  tree3->Branch("branch", &jagged3, 10485760);
  for (int64_t i3 = 0;  i3 < LENOFFSETS3 - 1;  i3++) {
    jagged3.clear();
    int64_t start3 = offsets3[i3];
    int64_t stop3 = offsets3[i3 + 1];
    for (int64_t i2 = start3;  i2 < stop3;  i2++) {
      std::vector<std::vector<float>> tmp2;
      int64_t start2 = offsets2[i2];
      int64_t stop2 = offsets2[i2 + 1];
      for (int64_t i1 = start2;  i1 < stop2;  i1++) {
        std::vector<float> tmp1;
        int64_t start1 = offsets1[i1];
        int64_t stop1 = offsets1[i1 + 1];
        for (int64_t i0 = start1;  i0 < stop1;  i0++) {
          tmp1.push_back(content[i0]);
        }
        tmp2.push_back(tmp1);
      }
      jagged3.push_back(tmp2);
    }
    tree3->Fill();
  }
  tree3->Write();
  file3->Close();
}


int main() {
  float* content = new float[LENCONTENT];
  int64_t* offsets1 = new int64_t[LENOFFSETS1];
  int64_t* offsets2 = new int64_t[LENOFFSETS2];
  int64_t* offsets3 = new int64_t[LENOFFSETS3];

  getdata(content, offsets1, offsets2, offsets3);

  ttree0(content, offsets1, offsets2, offsets3);
  ttree1(content, offsets1, offsets2, offsets3);
  ttree2(content, offsets1, offsets2, offsets3);
  ttree3(content, offsets1, offsets2, offsets3);

  return 0;
}

// python3 -i -c 'import uproot; from uproot import asgenobj, asdtype, STLVector; t0 = uproot.open("data/sample-jagged0.root")["jagged0"]; t1 = uproot.open("data/sample-jagged1.root")["jagged1"]; t2 = uproot.open("data/sample-jagged2.root")["jagged2"]; t3 = uproot.open("data/sample-jagged3.root")["jagged3"]'
// t0["branch"].array(entrystart=-100)
// t1["branch"].array(entrystart=-100)
// t2["branch"].array(entrystart=-100)
// t3["branch"].array(asgenobj(STLVector(STLVector(STLVector(asdtype(">f4")))), t3["branch"]._context, 6), entrystart=-100)
