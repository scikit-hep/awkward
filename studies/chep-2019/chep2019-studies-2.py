import time

import numpy

import ROOT
import root_numpy
import awkward
import awkward1
import uproot
from uproot import asgenobj, asdtype, STLVector

num = 0
starttime = time.time()
branch = uproot.open("data/sample-jagged0.root")["jagged0"]["branch"]
for i in range(branch.numbaskets):
    result = branch.basket(i)
    num += len(result)
walltime = time.time() - starttime
print("TTree uproot jagged0\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

file = ROOT.TFile("data/sample-jagged0.root")
tree = file.Get("jagged0")
starttime = time.time()
for i in range(branch.numbaskets):
    result = root_numpy.tree2array(tree, "branch", start=branch.basket_entrystart(i), stop=branch.basket_entrystop(i))
walltime = time.time() - starttime
print("TTree root_numpy jagged0\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

num = 0
starttime = time.time()
branch = uproot.open("data/sample-jagged1.root")["jagged1"]["branch"]
for i in range(branch.numbaskets):
    result = branch.basket(i)
    num += len(result.content)
walltime = time.time() - starttime
print("TTree OLD jagged1\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

num = 0
starttime = time.time()
branch = uproot.open("data/sample-jagged1.root")["jagged1"]["branch"]
for i in range(branch.numbaskets):
    jagged = branch.basket(i, uproot.asdebug)
    byteoffsets = awkward1.layout.Index64(jagged.offsets)
    rawdata = awkward1.layout.NumpyArray(jagged.content[6:])
    result = awkward1.layout.fromroot_nestedvector(byteoffsets, rawdata, 1, numpy.dtype(">f").itemsize, ">f")
    q = numpy.asarray(result.content).astype("<f")
    num += len(result.content)
walltime = time.time() - starttime
print("TTree NEW jagged1\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

file = ROOT.TFile("data/sample-jagged1.root")
tree = file.Get("jagged1")
starttime = time.time()
for i in range(branch.numbaskets):
    result = root_numpy.tree2array(tree, "branch", start=branch.basket_entrystart(i), stop=branch.basket_entrystop(i))
walltime = time.time() - starttime
print("TTree root_numpy jagged1\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

num = 0
starttime = time.time()
branch = uproot.open("data/sample-jagged2.root")["jagged2"]["branch"]
for i in range(5):
    jagged = branch.basket(i)
    q = awkward.fromiter(jagged)
    num += len(q.content.content)
walltime = time.time() - starttime
print("TTree OLD jagged2\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

num = 0
starttime = time.time()
branch = uproot.open("data/sample-jagged2.root")["jagged2"]["branch"]
for i in range(5):
    jagged = branch.basket(i, uproot.asdebug)
    byteoffsets = awkward1.layout.Index64(jagged.offsets)
    rawdata = awkward1.layout.NumpyArray(jagged.content[6:])
    result = awkward1.layout.fromroot_nestedvector(byteoffsets, rawdata, 2, numpy.dtype(">f").itemsize, ">f")
    q = numpy.asarray(result.content.content).astype("<f")
    num += len(result.content.content)
walltime = time.time() - starttime
print("TTree NEW jagged2\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

file = ROOT.TFile("data/sample-jagged2.root")
tree = file.Get("jagged2")
starttime = time.time()
for i in range(5):
    result = root_numpy.tree2array(tree, "branch", start=branch.basket_entrystart(i), stop=branch.basket_entrystop(i))
walltime = time.time() - starttime
print("TTree root_numpy jagged2\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

num = 0
starttime = time.time()
branch = uproot.open("data/sample-jagged3.root")["jagged3"]["branch"]
for i in range(5):
    jagged = branch.basket(i, asgenobj(STLVector(STLVector(STLVector(asdtype(">f4")))), branch._context, 6))
    q = awkward.fromiter(jagged)
    num += len(q.content.content.content)
walltime = time.time() - starttime
print("TTree OLD jagged3\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")

num = 0
starttime = time.time()
branch = uproot.open("data/sample-jagged3.root")["jagged3"]["branch"]
for i in range(5):
    jagged = branch.basket(i, uproot.asdebug)
    byteoffsets = awkward1.layout.Index64(jagged.offsets)
    rawdata = awkward1.layout.NumpyArray(jagged.content[6:])
    result = awkward1.layout.fromroot_nestedvector(byteoffsets, rawdata, 3, numpy.dtype(">f").itemsize, ">f")
    q = numpy.asarray(result.content.content.content).astype("<f")
    num += len(result.content.content.content)
walltime = time.time() - starttime
print("TTree NEW jagged3\t", walltime, "sec;\t", num/walltime/1e6, "million floats/sec")
