import numpy as np

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

def argdistincts(counts):
    i, j = arg_helper(counts, absolute=False)
    content = cp.vstack((i, j)).T

    return content

def distincts(content, counts):
    i, j = arg_helper(counts, absolute=True)

    if max(i.max(), j.max()) >= len(content):
        raise IndexError("index exceeds the bounds of the content array.")

    out_content = cp.vstack((content[i], content[j])).T

    return out_content

content = cp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
counts = cp.array([2, 0, 4, 2])

result = argdistincts(counts)
print("argcombinations:\n", result)
result = distincts(content, counts)
print("combinations:\n", result)