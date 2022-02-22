# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import ROOT

import awkward as ak

from awkward._v2._connect.cling import *
from ctypes import *

ROOT.RDF.RInterface(
    "ROOT::Detail::RDF::RLoopManager", "void"
).AsAwkward = lambda self: _as_awkward(self)

print_me_cpp = """
void print_me(ROOT::RDF::RResultPtr<ULong64_t> myResultProxy) {
    ULong64_t* ptr = myResultProxy.GetPtr();
    // FIXME: only objects that support iteration
    // for (auto& myItem : myResultProxy) {
    for (int64_t i = 0; i < myResultProxy.GetValue(); i++ ) {
        std::cout << (double)ptr[i] << ", ";
    };
}
"""

progress_cpp = """
void progress(const ROOT::RDataFrame& tdf, int64_t nEvents) {
    // any action would do, but `Count` is the most lightweight
    auto c = tdf.Count();
    std::string progress;
    std::mutex bar_mutex;
    c.OnPartialResultSlot(nEvents / 100, [&progress, &bar_mutex](unsigned int, ULong64_t &) {
        std::lock_guard<std::mutex> lg(bar_mutex);
        progress.push_back('#');
        std::cout << "[";
        std::cout << std::left;
        std::cout << std::setw(100);
        std::cout << progress;
        std::cout << "]";
        std::cout << std::flush;
    });
    std::cout << "Analysis running..." << std::endl;

    // trigger the event loop by accessing an action's result
    *c;
    std::cout << std::endl << "Done!" << std::endl;
}
"""


def _as_awkward(self):
    print(self)
    ncol = self.GetColumnNames()
    if len(ncol) == 0:
        return ak._v2.contents.EmptyArray()

    else:
        for key in range(len(ncol)):
            print(key, "->", ncol[key])

        # FIXME: handle just one column for now
        ncol_x = self.GetColumnNames()
        ncol_x_type = self.GetColumnType(ncol_x[0])

        entries_x = self.Count()
        num = entries_x.GetValue()
        ptr = entries_x.GetPtr()
        print(f"{entries_x.GetValue()} entries passed all filters")

        ROOT.gInterpreter.ProcessLine(print_me_cpp)
        ROOT.print_me(entries_x)

        # ROOT.gInterpreter.ProcessLine(progress_cpp)
        # ROOT.progress(self, 5)
        #
        return ak._v2.contents.EmptyArray()

        raise NotImplementedError(f"FIXME: cannot handle {ncol} columns just yet.")
