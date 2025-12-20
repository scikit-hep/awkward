import cupy as cp
import awkward as ak

# A small script to randomly generate a ListOffsetArray
print("Generating values...")
array_size = 1000000
# inner arrays lengths 
lengths = cp.random.randint(0, 6, array_size)
# a flat array of all the data
layout = ak.contents.NumpyArray(cp.random.random(int(sum(lengths))))
# calculate offsets from the lenghts
offsets = ak.index.Index(cp.concatenate((cp.array([0]), cp.cumsum(lengths))))

print("Constructing an array...")
rand_arr = ak.Array(ak.contents.ListOffsetArray(offsets, layout))
print("Generated array:", rand_arr)

print("Saving and array to a parquet file...")
ak.to_parquet(ak.concatenate([rand_arr] * 10), "random_listoffset_small.parquet")