# generate ROOT dictionary
rootcling -f mydict.cxx -rmf libmydict.rootmap -rml libmydict.so  LinkDef.h
clang++ -mmacosx-version-min=11.6 -shared -o libmydict.so mydict.cxx `root-config --cflags --libs`

