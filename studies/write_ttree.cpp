#include "TInterpreter.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include <iostream>
#include <vector>

void write_ttree() {
  gInterpreter->GenerateDictionary("vector<vector<vector<float> > >", "vector");

  std::vector<std::vector<std::vector<float>>> data;

  TFile* f = new TFile("sample-jagged3.root", "RECREATE");
  TTree* t = new TTree("tree", "");
  t->Branch("branch", &data);

  data = {{{1.1, 2.2}, {3.3}}, {}, {{4.4, 5.5, 6.6}, {}, {7.7, 8.8}}};
  t->Fill();

  data = {{{100}, {200, 300}}, {}, {{400, 500, 600}, {}, {700, 800}}};
  t->Fill();

  t->Write();
  f->Close();
}

// python -i -c 'import uproot; from uproot import *; t = uproot.open("sample-jagged3.root")["tree"]; branch = t["branch"]'
// branch.array(asgenobj(STLVector(STLVector(STLVector(asdtype(">f4")))), branch._context, 6))
