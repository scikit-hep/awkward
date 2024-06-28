import cupy as cp

def arg_helper(counts, absolute=False):
    counts_comb = counts * (counts - 1) // 2
    offsets_comb = cp.cumsum(cp.concatenate((cp.array([0]), counts_comb)))
    parents_comb = cp.zeros(int(offsets_comb[-1]), dtype=int)

    for i in range(1, len(offsets_comb)):
        parents_comb[offsets_comb[i-1]:offsets_comb[i]] = i - 1

    local_indices = cp.arange(offsets_comb[-1]) - offsets_comb[parents_comb]

    n = counts[parents_comb]
    b = 2 * n - 1
    i = cp.floor((b - cp.sqrt(b * b - 8 * local_indices)) / 2).astype(counts_comb.dtype)
    j = local_indices + i * (i - b + 2) // 2 + 1

    if absolute:
        starts_parents = cp.cumsum(cp.concatenate((cp.array([0]), counts)))[:-1][parents_comb]
        i += starts_parents
        j += starts_parents
        
    return i, j

def argdistincts(starts, stops):
    counts = stops - starts
    i, j = arg_helper(counts, absolute=False)
    out = cp.vstack((i, j)).T

    return out

def distincts(starts, stops, content):
    counts = stops - starts
    i, j = arg_helper(counts, absolute=True)

    if max(i.max(), j.max()) >= len(content):
        raise IndexError("index exceeds the bounds of the content array.")

    out = cp.vstack((content[i], content[j])).T

    return out

content = cp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

counts = cp.array([4, 0, 3, 1, 5])
starts = cp.array([0, 4, 4, 7, 8])
stops = cp.array([4, 4, 7, 8, 13])

result = argdistincts(starts, stops)
print("argcombinations:\n", result)

result = distincts(starts, stops, content)
print("\ncombinations:\n", result)