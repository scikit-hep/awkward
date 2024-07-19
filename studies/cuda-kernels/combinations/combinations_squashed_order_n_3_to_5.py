import cupy as cp

def root2(a):
    return cp.floor((1+cp.sqrt(8*a+1))/2)


def root3(a):
    out = 2*cp.ones(a.shape)
    mask = a > 0
    rad = cp.power(cp.sqrt(3)*cp.sqrt(243*a[mask]**2 - 1) + 27*a[mask], 1./3)
    # 1e-12 to correct rounding error (good to 1000 choose 3)
    out[mask] = cp.floor(cp.power(3, -2./3)*rad + cp.power(3, -1./3)/rad + 1 + 1e-12)
    return out


def root4(a):
    # good to (at least) 100 choose 4
    return cp.floor((cp.sqrt(4*cp.sqrt(24*a + 1) + 5) + 3)/2)


def repeat(x, repeats):
    all_stops = cp.cumsum(repeats)
    parents = cp.zeros(all_stops[-1].item(), dtype=int)
    stops, stop_counts = cp.unique(all_stops[:-1], return_counts=True)
    parents[stops] = stop_counts
    cp.cumsum(parents, out=parents)
    return x[parents]


def argchoose(starts, stops, n, absolute=False, replacement=False):
    counts = stops - starts
    if n > 5:
        raise NotImplementedError
    elif n == 5:
        counts_comb = counts*(counts - 1)*(counts - 2)*(counts - 3)*(counts - 4)//120
    elif n == 4:
        counts_comb = counts*(counts - 1)*(counts - 2)*(counts - 3)//24
    elif n == 3:
        counts_comb = counts*(counts - 1)*(counts - 2)//6
    elif n == 2:
        counts_comb = counts*(counts - 1)//2
    elif n <= 1:
        raise ValueError("Choosing 0 or 1 items is trivial")

    offsets = cp.cumsum(cp.concatenate((cp.array([0]), counts_comb)))
    offsets2 = cp.cumsum(cp.concatenate((cp.array([0]), counts)))
    parents = cp.zeros(int(offsets[-1]), dtype=int)
    parents2 = cp.zeros(int(offsets2[-1]), dtype=int)
    for i in range(1, len(offsets)):
        parents[offsets[i-1]:offsets[i]] = i - 1
    for i in range(1, len(offsets2)):
        parents2[offsets2[i-1]:offsets2[i]] = i - 1
    local = cp.arange(offsets2[-1]) - offsets2[parents2]
    indices = cp.arange(offsets[-1])

    if n == 5:
        k5 = indices - offsets[parents]
        i5 = repeat(local, local*(local - 1)*(local - 2)*(local - 3)//24)
        k4 = k5 - i5*(i5 - 1)*(i5 - 2)*(i5 - 3)*(i5 - 4)//120
        i4 = root4(k4)
        k3 = k4 - i4*(i4 - 1)*(i4 - 2)*(i4 - 3)//24
        i3 = root3(k3)
        k2 = k3 - i3*(i3 - 1)*(i3 - 2)//6
        i2 = root2(k2)
        k1 = k2 - i2*(i2 - 1)//2
        i1 = k1
        if absolute:
            starts_parents = starts[parents]
            for idx in [i1, i2, i3, i4, i5]:
                idx += starts_parents
        out = cp.vstack((i1, i2, i3, i4, i5)).T

    elif n == 4:
        k4 = indices - offsets[parents]
        i4 = repeat(local, local*(local - 1)*(local - 2)//6)
        k3 = k4 - i4*(i4 - 1)*(i4 - 2)*(i4 - 3)//24
        i3 = root3(k3)
        k2 = k3 - i3*(i3 - 1)*(i3 - 2)//6
        i2 = root2(k2)
        k1 = k2 - i2*(i2 - 1)//2
        i1 = k1
        if absolute:
            starts_parents = starts[parents]
            for idx in [i1, i2, i3, i4]:
                idx += starts_parents
        out = cp.vstack((i1, i2, i3, i4)).T

    elif n == 3:
        k3 = indices - offsets[parents]
        i3 = repeat(local, local*(local - 1)//2)
        k2 = k3 - i3*(i3 - 1)*(i3 - 2)//6
        i2 = root2(k2)
        k1 = k2 - i2*(i2 - 1)//2
        i1 = k1
        if absolute:
            starts_parents = starts[parents]
            for idx in [i1, i2, i3]:
                idx += starts_parents
        out = cp.vstack((i1, i2, i3)).T

    return out


content = cp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

counts = cp.array([4, 0, 3, 1, 5])
starts = cp.array([0, 4, 4, 7, 8])
stops = cp.array([4, 4, 7, 8, 13])

result = argchoose(starts, stops, 3)
print("argcombinations (n = 3):\n", result)

result = argchoose(starts, stops, 4)
print("argcombinations (n = 4):\n", result)

result = argchoose(starts, stops, 5)
print("argcombinations (n = 5):\n", result)