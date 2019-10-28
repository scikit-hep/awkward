set logscale y

plot [-0.5:3.5][0.1:2000] "read_rntuple.dat" t "RNTuple + new awkward" with lines, "read_ttree_old.dat" t "TTree in uproot + old awkward" with lines,  "read_ttree_new.dat" t "TTree in uproot + new awkward" with lines,  "read_ttree_rootnumpy.dat" t "TTree in root-numpy" with lines, "read_pyobj_old.dat" t "Python objects in old awkward" with lines, "read_pyobj_new.dat" t "Python objects in new awkward" with lines, "read_json_old.dat" t "JSON in old awkward" with lines, "read_json_new.dat" t "JSON in new awkward" with lines
