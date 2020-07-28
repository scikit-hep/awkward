

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

def awkward_new_Identities(toptr, length):
    for i in range(length):
        toptr[i] = i

awkward_new_Identities32 = awkward_new_Identities
awkward_new_Identities64 = awkward_new_Identities


def awkward_Identities32_to_Identities64(toptr, fromptr, length, width):
    for i in range(length * width):
        toptr[i] = int(fromptr[i])

def awkward_Identities_from_ListOffsetArray(
    toptr,
    fromptr,
    fromoffsets,
    fromptroffset,
    offsetsoffset,
    tolength,
    fromlength,
    fromwidth,
):
    globalstart = fromoffsets[offsetsoffset]
    globalstop = fromoffsets[offsetsoffset + fromlength]
    for k in range(globalstart * (fromwidth + 1)):
        toptr[k] = -1
    k = globalstop * (fromwidth + 1)
    while k < (tolength * (fromwidth + 1)):
        toptr[k] = -1
        k = k + 1
    for i in range(fromlength):
        start = fromoffsets[offsetsoffset + i]
        stop = fromoffsets[(offsetsoffset + i) + 1]
        if (start != stop) and (stop > tolength):
            raise ValueError("max(stop) > len(content)")
        for j in range(start, stop):
            for k in range(fromwidth):
                toptr[(j * (fromwidth + 1)) + k] = fromptr[
                    (fromptroffset + (i * fromwidth)) + k
                ]
            toptr[(j * (fromwidth + 1)) + fromwidth] = float(j - start)

awkward_Identities64_from_ListOffsetArray64 = awkward_Identities_from_ListOffsetArray
awkward_Identities32_from_ListOffsetArray64 = awkward_Identities_from_ListOffsetArray
awkward_Identities32_from_ListOffsetArray32 = awkward_Identities_from_ListOffsetArray
awkward_Identities32_from_ListOffsetArrayU32 = awkward_Identities_from_ListOffsetArray
awkward_Identities64_from_ListOffsetArrayU32 = awkward_Identities_from_ListOffsetArray
awkward_Identities64_from_ListOffsetArray32 = awkward_Identities_from_ListOffsetArray


def awkward_Identities_from_ListArray(
    uniquecontents,
    toptr,
    fromptr,
    fromstarts,
    fromstops,
    fromptroffset,
    startsoffset,
    stopsoffset,
    tolength,
    fromlength,
    fromwidth,
):
    for k in range(tolength * (fromwidth + 1)):
        toptr[k] = -1
    for i in range(fromlength):
        start = fromstarts[startsoffset + i]
        stop = fromstops[stopsoffset + i]
        if (start != stop) and (stop > tolength):
            raise ValueError("max(stop) > len(content)")
        for j in range(start, stop):
            if toptr[(j * (fromwidth + 1)) + fromwidth] != -1:
                uniquecontents[0] = False
                return
            for k in range(fromwidth):
                toptr[(j * (fromwidth + 1)) + k] = fromptr[
                    (fromptroffset + (i * fromwidth)) + k
                ]
            toptr[(j * (fromwidth + 1)) + fromwidth] = float(j - start)
    uniquecontents[0] = True

awkward_Identities64_from_ListArrayU32 = awkward_Identities_from_ListArray
awkward_Identities32_from_ListArrayU32 = awkward_Identities_from_ListArray
awkward_Identities32_from_ListArray64 = awkward_Identities_from_ListArray
awkward_Identities64_from_ListArray32 = awkward_Identities_from_ListArray
awkward_Identities32_from_ListArray32 = awkward_Identities_from_ListArray
awkward_Identities64_from_ListArray64 = awkward_Identities_from_ListArray


def awkward_Identities_from_RegularArray(
    toptr, fromptr, fromptroffset, size, tolength, fromlength, fromwidth
):
    for i in range(fromlength):
        for j in range(size):
            for k in range(fromwidth):
                toptr[(((i * size) + j) * (fromwidth + 1)) + k] = fromptr[
                    (fromptroffset + (i * fromwidth)) + k
                ]
            toptr[(((i * size) + j) * (fromwidth + 1)) + fromwidth] = float(j)
    k = ((fromlength + 1) * size) * (fromwidth + 1)
    while k < (tolength * (fromwidth + 1)):
        toptr[k] = -1
        k = k + 1

awkward_Identities64_from_RegularArray = awkward_Identities_from_RegularArray
awkward_Identities32_from_RegularArray = awkward_Identities_from_RegularArray


def awkward_Identities_from_IndexedArray(
    uniquecontents,
    toptr,
    fromptr,
    fromindex,
    fromptroffset,
    indexoffset,
    tolength,
    fromlength,
    fromwidth,
):
    for k in range(tolength * fromwidth):
        toptr[k] = -1
    for i in range(fromlength):
        j = fromindex[indexoffset + i]
        if j >= tolength:
            raise ValueError("max(index) > len(content)")
        else:
            if j >= 0:
                if toptr[j * fromwidth] != -1:
                    uniquecontents[0] = False
                    return
                for k in range(fromwidth):
                    toptr[(j * fromwidth) + k] = fromptr[
                        (fromptroffset + (i * fromwidth)) + k
                    ]
    uniquecontents[0] = True

awkward_Identities64_from_IndexedArray32 = awkward_Identities_from_IndexedArray
awkward_Identities64_from_IndexedArrayU32 = awkward_Identities_from_IndexedArray
awkward_Identities64_from_IndexedArray64 = awkward_Identities_from_IndexedArray
awkward_Identities32_from_IndexedArray32 = awkward_Identities_from_IndexedArray
awkward_Identities32_from_IndexedArrayU32 = awkward_Identities_from_IndexedArray
awkward_Identities32_from_IndexedArray64 = awkward_Identities_from_IndexedArray


def awkward_Identities_from_UnionArray(
    uniquecontents,
    toptr,
    fromptr,
    fromtags,
    fromindex,
    fromptroffset,
    tagsoffset,
    indexoffset,
    tolength,
    fromlength,
    fromwidth,
    which,
):
    for k in range(tolength * fromwidth):
        toptr[k] = -1
    for i in range(fromlength):
        if fromtags[tagsoffset + i] == which:
            j = fromindex[indexoffset + i]
            if j >= tolength:
                raise ValueError("max(index) > len(content)")
            else:
                if j < 0:
                    raise ValueError("min(index) < 0")
                else:
                    if toptr[j * fromwidth] != -1:
                        uniquecontents[0] = False
                        return
                    for k in range(fromwidth):
                        toptr[(j * fromwidth) + k] = fromptr[
                            (fromptroffset + (i * fromwidth)) + k
                        ]
    uniquecontents[0] = True

awkward_Identities64_from_UnionArray8_32 = awkward_Identities_from_UnionArray
awkward_Identities32_from_UnionArray8_64 = awkward_Identities_from_UnionArray
awkward_Identities32_from_UnionArray8_32 = awkward_Identities_from_UnionArray
awkward_Identities64_from_UnionArray8_64 = awkward_Identities_from_UnionArray
awkward_Identities64_from_UnionArray8_U32 = awkward_Identities_from_UnionArray
awkward_Identities32_from_UnionArray8_U32 = awkward_Identities_from_UnionArray


def awkward_Identities_extend(toptr, fromptr, fromoffset, fromlength, tolength):
    i = 0
    while i < fromlength:
        toptr[i] = fromptr[fromoffset + i]
        i = i + 1
    while i < tolength:
        toptr[i] = -1
        i = i + 1

awkward_Identities32_extend = awkward_Identities_extend
awkward_Identities64_extend = awkward_Identities_extend


def awkward_ListArray_num(
    tonum, fromstarts, startsoffset, fromstops, stopsoffset, length
):
    for i in range(length):
        start = fromstarts[startsoffset + i]
        stop = fromstops[stopsoffset + i]
        tonum[i] = float(stop - start)

awkward_ListArrayU32_num_64 = awkward_ListArray_num
awkward_ListArray32_num_64 = awkward_ListArray_num
awkward_ListArray64_num_64 = awkward_ListArray_num


def awkward_RegularArray_num(tonum, size, length):
    for i in range(length):
        tonum[i] = size

awkward_RegularArray_num_64 = awkward_RegularArray_num


def awkward_ListOffsetArray_flatten_offsets(
    tooffsets,
    outeroffsets,
    outeroffsetsoffset,
    outeroffsetslen,
    inneroffsets,
    inneroffsetsoffset,
    inneroffsetslen,
):
    for i in range(outeroffsetslen):
        tooffsets[i] = inneroffsets[
            inneroffsetsoffset + outeroffsets[outeroffsetsoffset + i]
        ]

awkward_ListOffsetArray64_flatten_offsets_64 = awkward_ListOffsetArray_flatten_offsets
awkward_ListOffsetArrayU32_flatten_offsets_64 = awkward_ListOffsetArray_flatten_offsets
awkward_ListOffsetArray32_flatten_offsets_64 = awkward_ListOffsetArray_flatten_offsets


def awkward_IndexedArray_flatten_none2empty(
    outoffsets,
    outindex,
    outindexoffset,
    outindexlength,
    offsets,
    offsetsoffset,
    offsetslength,
):
    outoffsets[0] = offsets[offsetsoffset + 0]
    k = 1
    for i in range(outindexlength):
        idx = outindex[outindexoffset + i]
        if idx < 0:
            outoffsets[k] = outoffsets[k - 1]
            k = k + 1
        else:
            if ((offsetsoffset + idx) + 1) >= offsetslength:
                raise ValueError("flattening offset out of range")
            else:
                count = (
                    offsets[(offsetsoffset + idx) + 1] - offsets[offsetsoffset + idx]
                )
                outoffsets[k] = outoffsets[k - 1] + count
                k = k + 1

awkward_IndexedArray64_flatten_none2empty_64 = awkward_IndexedArray_flatten_none2empty
awkward_IndexedArray32_flatten_none2empty_64 = awkward_IndexedArray_flatten_none2empty
awkward_IndexedArrayU32_flatten_none2empty_64 = awkward_IndexedArray_flatten_none2empty


def awkward_UnionArray_flatten_length(
    total_length,
    fromtags,
    fromtagsoffset,
    fromindex,
    fromindexoffset,
    length,
    offsetsraws,
    offsetsoffsets,
):
    total_length[0] = 0
    for i in range(length):
        tag = fromtags[fromtagsoffset + i]
        idx = fromindex[fromindexoffset + i]
        start = offsetsraws[tag][offsetsoffsets[tag] + idx]
        stop = offsetsraws[tag][(offsetsoffsets[tag] + idx) + 1]
        total_length[0] = total_length[0] + (stop - start)

awkward_UnionArray64_flatten_length_64 = awkward_UnionArray_flatten_length
awkward_UnionArrayU32_flatten_length_64 = awkward_UnionArray_flatten_length
awkward_UnionArray32_flatten_length_64 = awkward_UnionArray_flatten_length


def awkward_UnionArray_flatten_combine(
    totags,
    toindex,
    tooffsets,
    fromtags,
    fromtagsoffset,
    fromindex,
    fromindexoffset,
    length,
    offsetsraws,
    offsetsoffsets,
):
    tooffsets[0] = 0
    k = 0
    for i in range(length):
        tag = fromtags[fromtagsoffset + i]
        idx = fromindex[fromindexoffset + i]
        start = offsetsraws[tag][offsetsoffsets[tag] + idx]
        stop = offsetsraws[tag][(offsetsoffsets[tag] + idx) + 1]
        tooffsets[i + 1] = tooffsets[i] + (stop - start)
        for j in range(start, stop):
            totags[k] = tag
            toindex[k] = j
            k = k + 1

awkward_UnionArray32_flatten_combine_64 = awkward_UnionArray_flatten_combine
awkward_UnionArrayU32_flatten_combine_64 = awkward_UnionArray_flatten_combine
awkward_UnionArray64_flatten_combine_64 = awkward_UnionArray_flatten_combine


def awkward_IndexedArray_flatten_nextcarry(
    tocarry, fromindex, indexoffset, lenindex, lencontent
):
    k = 0
    for i in range(lenindex):
        j = fromindex[indexoffset + i]
        if j >= lencontent:
            raise ValueError("index out of range")
        else:
            if j >= 0:
                tocarry[k] = j
                k = k + 1

awkward_IndexedArray32_flatten_nextcarry_64 = awkward_IndexedArray_flatten_nextcarry
awkward_IndexedArray64_flatten_nextcarry_64 = awkward_IndexedArray_flatten_nextcarry
awkward_IndexedArrayU32_flatten_nextcarry_64 = awkward_IndexedArray_flatten_nextcarry


def awkward_IndexedArray_overlay_mask(
    toindex, mask, maskoffset, fromindex, indexoffset, length
):
    for i in range(length):
        m = mask[maskoffset + i]
        toindex[i] = -1 if m else fromindex[indexoffset + i]

awkward_IndexedArrayU32_overlay_mask8_to64 = awkward_IndexedArray_overlay_mask
awkward_IndexedArray64_overlay_mask8_to64 = awkward_IndexedArray_overlay_mask
awkward_IndexedArray32_overlay_mask8_to64 = awkward_IndexedArray_overlay_mask


def awkward_IndexedArray_mask(tomask, fromindex, indexoffset, length):
    for i in range(length):
        tomask[i] = fromindex[indexoffset + i] < 0

awkward_IndexedArray32_mask8 = awkward_IndexedArray_mask
awkward_IndexedArray64_mask8 = awkward_IndexedArray_mask
awkward_IndexedArrayU32_mask8 = awkward_IndexedArray_mask


def awkward_ByteMaskedArray_mask(tomask, frommask, maskoffset, length, validwhen):
    for i in range(length):
        tomask[i] = (frommask[maskoffset + i] != 0) != validwhen

awkward_ByteMaskedArray_mask8 = awkward_ByteMaskedArray_mask


def awkward_zero_mask(tomask, length):
    for i in range(length):
        tomask[i] = 0

awkward_zero_mask8 = awkward_zero_mask


def awkward_IndexedArray_simplify(
    toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength
):
    for i in range(outerlength):
        j = outerindex[outeroffset + i]
        if j < 0:
            toindex[i] = -1
        else:
            if j >= innerlength:
                raise ValueError("index out of range")
            else:
                toindex[i] = innerindex[inneroffset + j]

awkward_IndexedArray32_simplifyU32_to64 = awkward_IndexedArray_simplify
awkward_IndexedArrayU32_simplify64_to64 = awkward_IndexedArray_simplify
awkward_IndexedArray64_simplify64_to64 = awkward_IndexedArray_simplify
awkward_IndexedArrayU32_simplifyU32_to64 = awkward_IndexedArray_simplify
awkward_IndexedArray32_simplify64_to64 = awkward_IndexedArray_simplify
awkward_IndexedArray64_simplify32_to64 = awkward_IndexedArray_simplify
awkward_IndexedArray32_simplify32_to64 = awkward_IndexedArray_simplify
awkward_IndexedArrayU32_simplify32_to64 = awkward_IndexedArray_simplify
awkward_IndexedArray64_simplifyU32_to64 = awkward_IndexedArray_simplify


def awkward_RegularArray_compact_offsets(tooffsets, length, size):
    tooffsets[0] = 0
    for i in range(length):
        tooffsets[i + 1] = (i + 1) * size

awkward_RegularArray_compact_offsets64 = awkward_RegularArray_compact_offsets


def awkward_ListArray_compact_offsets(
    tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length
):
    tooffsets[0] = 0
    for i in range(length):
        start = fromstarts[startsoffset + i]
        stop = fromstops[stopsoffset + i]
        if stop < start:
            raise ValueError("stops[i] < starts[i]")
        tooffsets[i + 1] = tooffsets[i] + (stop - start)

awkward_ListArray32_compact_offsets_64 = awkward_ListArray_compact_offsets
awkward_ListArrayU32_compact_offsets_64 = awkward_ListArray_compact_offsets
awkward_ListArray64_compact_offsets_64 = awkward_ListArray_compact_offsets


def awkward_ListOffsetArray_compact_offsets(
    tooffsets, fromoffsets, offsetsoffset, length
):
    diff = int(fromoffsets[offsetsoffset + 0])
    tooffsets[0] = 0
    for i in range(length):
        tooffsets[i + 1] = fromoffsets[(offsetsoffset + i) + 1] - diff

awkward_ListOffsetArray64_compact_offsets_64 = awkward_ListOffsetArray_compact_offsets
awkward_ListOffsetArray32_compact_offsets_64 = awkward_ListOffsetArray_compact_offsets
awkward_ListOffsetArrayU32_compact_offsets_64 = awkward_ListOffsetArray_compact_offsets


def awkward_ListArray_broadcast_tooffsets(
    tocarry,
    fromoffsets,
    offsetsoffset,
    offsetslength,
    fromstarts,
    startsoffset,
    fromstops,
    stopsoffset,
    lencontent,
):
    k = 0
    for i in range(offsetslength - 1):
        start = int(fromstarts[startsoffset + i])
        stop = int(fromstops[stopsoffset + i])
        if (start != stop) and (stop > lencontent):
            raise ValueError("stops[i] > len(content)")
        count = int(
            fromoffsets[(offsetsoffset + i) + 1] - fromoffsets[offsetsoffset + i]
        )
        if count < 0:
            raise ValueError("broadcast's offsets must be monotonically increasing")
        if (stop - start) != count:
            raise ValueError("cannot broadcast nested list")
        for j in range(start, stop):
            tocarry[k] = float(j)
            k = k + 1

awkward_ListArrayU32_broadcast_tooffsets_64 = awkward_ListArray_broadcast_tooffsets
awkward_ListArray64_broadcast_tooffsets_64 = awkward_ListArray_broadcast_tooffsets
awkward_ListArray32_broadcast_tooffsets_64 = awkward_ListArray_broadcast_tooffsets


def awkward_RegularArray_broadcast_tooffsets(
    fromoffsets, offsetsoffset, offsetslength, size
):
    for i in range(offsetslength - 1):
        count = int(
            fromoffsets[(offsetsoffset + i) + 1] - fromoffsets[offsetsoffset + i]
        )
        if count < 0:
            raise ValueError("broadcast's offsets must be monotonically increasing")
        if size != count:
            raise ValueError("cannot broadcast nested list")

awkward_RegularArray_broadcast_tooffsets_64 = awkward_RegularArray_broadcast_tooffsets


def awkward_RegularArray_broadcast_tooffsets_size1(
    tocarry, fromoffsets, offsetsoffset, offsetslength
):
    k = 0
    for i in range(offsetslength - 1):
        count = int(
            fromoffsets[(offsetsoffset + i) + 1] - fromoffsets[offsetsoffset + i]
        )
        if count < 0:
            raise ValueError("broadcast's offsets must be monotonically increasing")
        for j in range(count):
            tocarry[k] = float(i)
            k = k + 1

awkward_RegularArray_broadcast_tooffsets_size1_64 = awkward_RegularArray_broadcast_tooffsets_size1


def awkward_ListOffsetArray_toRegularArray(
    size, fromoffsets, offsetsoffset, offsetslength
):
    size[0] = -1
    for i in range(offsetslength - 1):
        count = int(
            fromoffsets[(offsetsoffset + i) + 1] - fromoffsets[offsetsoffset + i]
        )
        if count < 0:
            raise ValueError("offsets must be monotonically increasing")
        if size[0] == -1:
            size[0] = count
        else:
            if size[0] != count:
                raise ValueError(
                    "cannot convert to RegularArray because subarray lengths are not regular"
                )
    if size[0] == -1:
        size[0] = 0

awkward_ListOffsetArray64_toRegularArray = awkward_ListOffsetArray_toRegularArray
awkward_ListOffsetArrayU32_toRegularArray = awkward_ListOffsetArray_toRegularArray
awkward_ListOffsetArray32_toRegularArray = awkward_ListOffsetArray_toRegularArray


def awkward_NumpyArray_fill_frombool(toptr, tooffset, fromptr, fromoffset, length):
    for i in range(length):
        toptr[tooffset + i] = float(fromptr[fromoffset + i] != 0)

awkward_NumpyArray_fill_toint64_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_toint32_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_toint16_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_toint8_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_tofloat64_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_tobool_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_touint16_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_touint64_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_tofloat32_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_touint8_frombool = awkward_NumpyArray_fill_frombool
awkward_NumpyArray_fill_touint32_frombool = awkward_NumpyArray_fill_frombool


def awkward_NumpyArray_fill(toptr, tooffset, fromptr, fromoffset, length):
    for i in range(length):
        toptr[tooffset + i] = float(fromptr[fromoffset + i])

awkward_NumpyArray_fill_touint32_fromuint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint8_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromuint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint64_fromuint64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint32_fromint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromint64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint64_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromuint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint32_fromuint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint16_fromint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromuint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint8_fromint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint64_fromuint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromint64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint16_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint32_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromfloat32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromuint64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint16_fromint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint32_fromint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromuint64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromuint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromuint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromfloat32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromfloat64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint32_fromuint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint32_fromint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint16_fromuint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat32_fromint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint32_fromuint8 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromuint16 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_toint64_fromint64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_tofloat64_fromuint64 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint64_fromuint32 = awkward_NumpyArray_fill
awkward_NumpyArray_fill_touint16_fromuint8 = awkward_NumpyArray_fill


def awkward_ListArray_fill(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstartsoffset,
    fromstops,
    fromstopsoffset,
    length,
    base,
):
    for i in range(length):
        tostarts[tostartsoffset + i] = float(fromstarts[fromstartsoffset + i] + base)
        tostops[tostopsoffset + i] = float(fromstops[fromstopsoffset + i] + base)

awkward_ListArray_fill_to64_fromU32 = awkward_ListArray_fill
awkward_ListArray_fill_to64_from32 = awkward_ListArray_fill
awkward_ListArray_fill_to64_from64 = awkward_ListArray_fill


def awkward_IndexedArray_fill(
    toindex, toindexoffset, fromindex, fromindexoffset, length, base
):
    for i in range(length):
        fromval = fromindex[fromindexoffset + i]
        toindex[toindexoffset + i] = -1 if fromval < 0 else float(fromval + base)

awkward_IndexedArray_fill_to64_from32 = awkward_IndexedArray_fill
awkward_IndexedArray_fill_to64_fromU32 = awkward_IndexedArray_fill
awkward_IndexedArray_fill_to64_from64 = awkward_IndexedArray_fill


def awkward_IndexedArray_fill_count(toindex, toindexoffset, length, base):
    for i in range(length):
        toindex[toindexoffset + i] = i + base

awkward_IndexedArray_fill_to64_count = awkward_IndexedArray_fill_count


def awkward_UnionArray_filltags(
    totags, totagsoffset, fromtags, fromtagsoffset, length, base
):
    for i in range(length):
        totags[totagsoffset + i] = float(fromtags[fromtagsoffset + i] + base)

awkward_UnionArray_filltags_to8_from8 = awkward_UnionArray_filltags


def awkward_UnionArray_fillindex(
    toindex, toindexoffset, fromindex, fromindexoffset, length
):
    for i in range(length):
        toindex[toindexoffset + i] = float(fromindex[fromindexoffset + i])

awkward_UnionArray_fillindex_to64_from64 = awkward_UnionArray_fillindex
awkward_UnionArray_fillindex_to64_from32 = awkward_UnionArray_fillindex
awkward_UnionArray_fillindex_to64_fromU32 = awkward_UnionArray_fillindex


def awkward_UnionArray_filltags_const(totags, totagsoffset, length, base):
    for i in range(length):
        totags[totagsoffset + i] = float(base)

awkward_UnionArray_filltags_to8_const = awkward_UnionArray_filltags_const


def awkward_UnionArray_fillindex_count(toindex, toindexoffset, length):
    for i in range(length):
        toindex[toindexoffset + i] = float(i)

awkward_UnionArray_fillindex_to64_count = awkward_UnionArray_fillindex_count


def awkward_UnionArray_simplify(
    totags,
    toindex,
    outertags,
    outertagsoffset,
    outerindex,
    outerindexoffset,
    innertags,
    innertagsoffset,
    innerindex,
    innerindexoffset,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base,
):
    for i in range(length):
        if outertags[outertagsoffset + i] == outerwhich:
            j = outerindex[outerindexoffset + i]
            if innertags[innertagsoffset + j] == innerwhich:
                totags[i] = float(towhich)
                toindex[i] = float(innerindex[innerindexoffset + j] + base)

awkward_UnionArray8_U32_simplify8_32_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_64_simplify8_32_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_64_simplify8_64_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_32_simplify8_U32_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_U32_simplify8_U32_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_32_simplify8_32_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_64_simplify8_U32_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_U32_simplify8_64_to8_64 = awkward_UnionArray_simplify
awkward_UnionArray8_32_simplify8_64_to8_64 = awkward_UnionArray_simplify


def awkward_UnionArray_simplify_one(
    totags,
    toindex,
    fromtags,
    fromtagsoffset,
    fromindex,
    fromindexoffset,
    towhich,
    fromwhich,
    length,
    base,
):
    for i in range(length):
        if fromtags[fromtagsoffset + i] == fromwhich:
            totags[i] = float(towhich)
            toindex[i] = float(fromindex[fromindexoffset + i] + base)

awkward_UnionArray8_32_simplify_one_to8_64 = awkward_UnionArray_simplify_one
awkward_UnionArray8_U32_simplify_one_to8_64 = awkward_UnionArray_simplify_one
awkward_UnionArray8_64_simplify_one_to8_64 = awkward_UnionArray_simplify_one


def awkward_ListArray_validity(
    starts, startsoffset, stops, stopsoffset, length, lencontent
):
    for i in range(length):
        start = starts[startsoffset + i]
        stop = stops[stopsoffset + i]
        if start != stop:
            if start > stop:
                raise ValueError("start[i] > stop[i]")
            if start < 0:
                raise ValueError("start[i] < 0")
            if stop > lencontent:
                raise ValueError("stop[i] > len(content)")

awkward_ListArrayU32_validity = awkward_ListArray_validity
awkward_ListArray32_validity = awkward_ListArray_validity
awkward_ListArray64_validity = awkward_ListArray_validity


def awkward_IndexedArray_validity(index, indexoffset, length, lencontent, isoption):
    for i in range(length):
        idx = index[indexoffset + i]
        if not (isoption):
            if idx < 0:
                raise ValueError("index[i] < 0")
        if idx >= lencontent:
            raise ValueError("index[i] >= len(content)")

awkward_IndexedArray64_validity = awkward_IndexedArray_validity
awkward_IndexedArrayU32_validity = awkward_IndexedArray_validity
awkward_IndexedArray32_validity = awkward_IndexedArray_validity


def awkward_UnionArray_validity(
    tags, tagsoffset, index, indexoffset, length, numcontents, lencontents
):
    for i in range(length):
        tag = tags[tagsoffset + i]
        idx = index[indexoffset + i]
        if tag < 0:
            raise ValueError("tags[i] < 0")
        if idx < 0:
            raise ValueError("index[i] < 0")
        if tag >= numcontents:
            raise ValueError("tags[i] >= len(contents)")
        lencontent = lencontents[tag]
        if idx >= lencontent:
            raise ValueError("index[i] >= len(content[tags[i]])")

awkward_UnionArray8_32_validity = awkward_UnionArray_validity
awkward_UnionArray8_U32_validity = awkward_UnionArray_validity
awkward_UnionArray8_64_validity = awkward_UnionArray_validity


def awkward_UnionArray_fillna(toindex, fromindex, offset, length):
    for i in range(length):
        toindex[i] = fromindex[offset + i] if fromindex[offset + i] >= 0 else 0

awkward_UnionArray_fillna_from64_to64 = awkward_UnionArray_fillna
awkward_UnionArray_fillna_fromU32_to64 = awkward_UnionArray_fillna
awkward_UnionArray_fillna_from32_to64 = awkward_UnionArray_fillna


def awkward_IndexedOptionArray_rpad_and_clip_mask_axis1(toindex, frommask, length):
    count = 0
    for i in range(length):
        if frommask[i]:
            toindex[i] = -1
        else:
            toindex[i] = count
            count = count + 1

awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64 = awkward_IndexedOptionArray_rpad_and_clip_mask_axis1


def awkward_index_rpad_and_clip_axis0(toindex, target, length):
    shorter = target if target < length else length
    for i in range(shorter):
        toindex[i] = i
    for i in range(shorter, target):
        toindex[i] = -1

awkward_index_rpad_and_clip_axis0_64 = awkward_index_rpad_and_clip_axis0


def awkward_index_rpad_and_clip_axis1(tostarts, tostops, target, length):
    offset = 0
    for i in range(length):
        tostarts[i] = offset
        offset = offset + target
        tostops[i] = offset

awkward_index_rpad_and_clip_axis1_64 = awkward_index_rpad_and_clip_axis1


def awkward_RegularArray_rpad_and_clip_axis1(toindex, target, size, length):
    shorter = target if target < size else size
    for i in range(length):
        for j in range(shorter):
            toindex[(i * target) + j] = (i * size) + j
        for j in range(shorter, target):
            toindex[(i * target) + j] = -1

awkward_RegularArray_rpad_and_clip_axis1_64 = awkward_RegularArray_rpad_and_clip_axis1


def awkward_ListArray_min_range(
    tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset
):
    shorter = fromstops[stopsoffset + 0] - fromstarts[startsoffset + 0]
    for i in range(1, lenstarts):
        rangeval = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
        shorter = shorter if shorter < rangeval else rangeval
    tomin[0] = shorter

awkward_ListArray64_min_range = awkward_ListArray_min_range
awkward_ListArrayU32_min_range = awkward_ListArray_min_range
awkward_ListArray32_min_range = awkward_ListArray_min_range


def awkward_ListArray_rpad_and_clip_length_axis1(
    tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset
):
    length = 0
    for i in range(lenstarts):
        rangeval = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
        length += target if target > rangeval else rangeval
    tomin[0] = length

awkward_ListArrayU32_rpad_and_clip_length_axis1 = awkward_ListArray_rpad_and_clip_length_axis1
awkward_ListArray32_rpad_and_clip_length_axis1 = awkward_ListArray_rpad_and_clip_length_axis1
awkward_ListArray64_rpad_and_clip_length_axis1 = awkward_ListArray_rpad_and_clip_length_axis1


def awkward_ListArray_rpad_axis1(
    toindex,
    fromstarts,
    fromstops,
    tostarts,
    tostops,
    target,
    length,
    startsoffset,
    stopsoffset,
):
    offset = 0
    for i in range(length):
        tostarts[i] = offset
        rangeval = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
        for j in range(rangeval):
            toindex[offset + j] = fromstarts[startsoffset + i] + j
        for j in range(rangeval, target):
            toindex[offset + j] = -1
        offset = tostarts[i] + target if target > rangeval else tostarts[i] + rangeval
        tostops[i] = offset

awkward_ListArrayU32_rpad_axis1_64 = awkward_ListArray_rpad_axis1
awkward_ListArray64_rpad_axis1_64 = awkward_ListArray_rpad_axis1
awkward_ListArray32_rpad_axis1_64 = awkward_ListArray_rpad_axis1


def awkward_ListOffsetArray_rpad_and_clip_axis1(
    toindex, fromoffsets, offsetsoffset, length, target
):
    for i in range(length):
        rangeval = float(
            fromoffsets[(offsetsoffset + i) + 1] - fromoffsets[offsetsoffset + i]
        )
        shorter = target if target < rangeval else rangeval
        for j in range(shorter):
            toindex[(i * target) + j] = float(fromoffsets[offsetsoffset + i]) + j
        for j in range(shorter, target):
            toindex[(i * target) + j] = -1

awkward_ListOffsetArray64_rpad_and_clip_axis1_64 = awkward_ListOffsetArray_rpad_and_clip_axis1
awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64 = awkward_ListOffsetArray_rpad_and_clip_axis1
awkward_ListOffsetArray32_rpad_and_clip_axis1_64 = awkward_ListOffsetArray_rpad_and_clip_axis1


def awkward_ListOffsetArray_rpad_length_axis1(
    tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength
):
    length = 0
    tooffsets[0] = 0
    for i in range(fromlength):
        rangeval = fromoffsets[(offsetsoffset + i) + 1] - fromoffsets[offsetsoffset + i]
        longer = rangeval if target < rangeval else target
        length = length + longer
        tooffsets[i + 1] = tooffsets[i] + longer
    tolength[0] = length

awkward_ListOffsetArrayU32_rpad_length_axis1 = awkward_ListOffsetArray_rpad_length_axis1
awkward_ListOffsetArray64_rpad_length_axis1 = awkward_ListOffsetArray_rpad_length_axis1
awkward_ListOffsetArray32_rpad_length_axis1 = awkward_ListOffsetArray_rpad_length_axis1


def awkward_ListOffsetArray_rpad_axis1(
    toindex, fromoffsets, offsetsoffset, fromlength, target
):
    count = 0
    for i in range(fromlength):
        rangeval = float(
            fromoffsets[(offsetsoffset + i) + 1] - fromoffsets[offsetsoffset + i]
        )
        for j in range(rangeval):
            toindex[count] = float(fromoffsets[offsetsoffset + i]) + j
            count = count + 1
        for j in range(rangeval, target):
            toindex[count] = -1
            count = count + 1

awkward_ListOffsetArrayU32_rpad_axis1_64 = awkward_ListOffsetArray_rpad_axis1
awkward_ListOffsetArray64_rpad_axis1_64 = awkward_ListOffsetArray_rpad_axis1
awkward_ListOffsetArray32_rpad_axis1_64 = awkward_ListOffsetArray_rpad_axis1


def awkward_localindex(toindex, length):
    for i in range(length):
        toindex[i] = i

awkward_localindex_64 = awkward_localindex


def awkward_ListArray_localindex(toindex, offsets, offsetsoffset, length):
    for i in range(length):
        start = int(offsets[offsetsoffset + i])
        stop = int(offsets[(offsetsoffset + i) + 1])
        for j in range(start, stop):
            toindex[j] = j - start

awkward_ListArray64_localindex_64 = awkward_ListArray_localindex
awkward_ListArray32_localindex_64 = awkward_ListArray_localindex
awkward_ListArrayU32_localindex_64 = awkward_ListArray_localindex


def awkward_RegularArray_localindex(toindex, size, length):
    for i in range(length):
        for j in range(size):
            toindex[(i * size) + j] = j

awkward_RegularArray_localindex_64 = awkward_RegularArray_localindex


def awkward_combinations(toindex, n, replacement, singlelen):
    raise ValueError("FIXME: awkward_combinations")

awkward_combinations_64 = awkward_combinations


def awkward_ListArray_combinations_length(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    startsoffset,
    stops,
    stopsoffset,
    length,
):
    totallen[0] = 0
    tooffsets[0] = 0
    for i in range(length):
        size = int(stops[stopsoffset + i] - starts[startsoffset + i])
        if replacement:
            size += n - 1
        thisn = n

        if thisn > size:
            combinationslen = 0
        else:
            if thisn == size:
                combinationslen = 1
            else:
                if (thisn * 2) > size:
                    thisn = size - thisn
                combinationslen = size
                j = 2
                while j <= thisn:
                    combinationslen *= (size - j) + 1
                    combinationslen /= j
                    j = j + 1
        totallen[0] = totallen[0] + combinationslen
        tooffsets[i + 1] = tooffsets[i] + combinationslen

awkward_ListArray64_combinations_length_64 = awkward_ListArray_combinations_length
awkward_ListArrayU32_combinations_length_64 = awkward_ListArray_combinations_length
awkward_ListArray32_combinations_length_64 = awkward_ListArray_combinations_length


def awkward_ByteMaskedArray_overlay_mask(
    tomask, theirmask, theirmaskoffset, mymask, mymaskoffset, length, validwhen
):
    for i in range(length):
        theirs = theirmask[theirmaskoffset + i]
        mine = (mymask[mymaskoffset + i] != 0) != validwhen
        tomask[i] = 1 if theirs | mine else 0

awkward_ByteMaskedArray_overlay_mask8 = awkward_ByteMaskedArray_overlay_mask


def awkward_BitMaskedArray_to_ByteMaskedArray(
    tobytemask, frombitmask, bitmaskoffset, bitmasklength, validwhen, lsb_order
):
    if lsb_order:
        for i in range(bitmasklength):
            byte = frombitmask[bitmaskoffset + i]
            tobytemask[(i * 8) + 0] = (byte & int(1)) != validwhen
            byte >>= 1
            tobytemask[(i * 8) + 1] = (byte & int(1)) != validwhen
            byte >>= 1
            tobytemask[(i * 8) + 2] = (byte & int(1)) != validwhen
            byte >>= 1
            tobytemask[(i * 8) + 3] = (byte & int(1)) != validwhen
            byte >>= 1
            tobytemask[(i * 8) + 4] = (byte & int(1)) != validwhen
            byte >>= 1
            tobytemask[(i * 8) + 5] = (byte & int(1)) != validwhen
            byte >>= 1
            tobytemask[(i * 8) + 6] = (byte & int(1)) != validwhen
            byte >>= 1
            tobytemask[(i * 8) + 7] = (byte & int(1)) != validwhen
    else:
        for i in range(bitmasklength):
            byte = frombitmask[bitmaskoffset + i]
            tobytemask[(i * 8) + 0] = ((byte & int(128)) != 0) != validwhen
            byte <<= 1
            tobytemask[(i * 8) + 1] = ((byte & int(128)) != 0) != validwhen
            byte <<= 1
            tobytemask[(i * 8) + 2] = ((byte & int(128)) != 0) != validwhen
            byte <<= 1
            tobytemask[(i * 8) + 3] = ((byte & int(128)) != 0) != validwhen
            byte <<= 1
            tobytemask[(i * 8) + 4] = ((byte & int(128)) != 0) != validwhen
            byte <<= 1
            tobytemask[(i * 8) + 5] = ((byte & int(128)) != 0) != validwhen
            byte <<= 1
            tobytemask[(i * 8) + 6] = ((byte & int(128)) != 0) != validwhen
            byte <<= 1
            tobytemask[(i * 8) + 7] = ((byte & int(128)) != 0) != validwhen

def awkward_BitMaskedArray_to_IndexedOptionArray(
    toindex, frombitmask, bitmaskoffset, bitmasklength, validwhen, lsb_order
):
    if lsb_order:
        for i in range(bitmasklength):
            byte = frombitmask[bitmaskoffset + i]
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 0] = (i * 8) + 0
            else:
                toindex[(i * 8) + 0] = -1
            byte >>= 1
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 1] = (i * 8) + 1
            else:
                toindex[(i * 8) + 1] = -1
            byte >>= 1
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 2] = (i * 8) + 2
            else:
                toindex[(i * 8) + 2] = -1
            byte >>= 1
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 3] = (i * 8) + 3
            else:
                toindex[(i * 8) + 3] = -1
            byte >>= 1
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 4] = (i * 8) + 4
            else:
                toindex[(i * 8) + 4] = -1
            byte >>= 1
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 5] = (i * 8) + 5
            else:
                toindex[(i * 8) + 5] = -1
            byte >>= 1
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 6] = (i * 8) + 6
            else:
                toindex[(i * 8) + 6] = -1
            byte >>= 1
            if (byte & int(1)) == validwhen:
                toindex[(i * 8) + 7] = (i * 8) + 7
            else:
                toindex[(i * 8) + 7] = -1
    else:
        for i in range(bitmasklength):
            byte = frombitmask[bitmaskoffset + i]
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 0] = (i * 8) + 0
            else:
                toindex[(i * 8) + 0] = -1
            byte <<= 1
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 1] = (i * 8) + 1
            else:
                toindex[(i * 8) + 1] = -1
            byte <<= 1
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 2] = (i * 8) + 2
            else:
                toindex[(i * 8) + 2] = -1
            byte <<= 1
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 3] = (i * 8) + 3
            else:
                toindex[(i * 8) + 3] = -1
            byte <<= 1
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 4] = (i * 8) + 4
            else:
                toindex[(i * 8) + 4] = -1
            byte <<= 1
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 5] = (i * 8) + 5
            else:
                toindex[(i * 8) + 5] = -1
            byte <<= 1
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 6] = (i * 8) + 6
            else:
                toindex[(i * 8) + 6] = -1
            byte <<= 1
            if ((byte & int(128)) != 0) == validwhen:
                toindex[(i * 8) + 7] = (i * 8) + 7
            else:
                toindex[(i * 8) + 7] = -1

awkward_BitMaskedArray_to_IndexedOptionArray64 = awkward_BitMaskedArray_to_IndexedOptionArray


def awkward_reduce_count_64(toptr, parents, parentsoffset, lenparents, outlength):
    for i in range(outlength):
        toptr[i] = 0
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] = toptr[parents[parentsoffset + i]] + 1

def awkward_reduce_countnonzero(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = 0
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] += fromptr[fromptroffset + i] != 0

awkward_reduce_countnonzero_float32_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_int64_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_int8_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_int16_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_int32_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_uint64_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_uint16_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_uint8_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_float64_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_bool_64 = awkward_reduce_countnonzero
awkward_reduce_countnonzero_uint32_64 = awkward_reduce_countnonzero


def awkward_reduce_sum(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = float(0)
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] += float(fromptr[fromptroffset + i])

awkward_reduce_sum_uint64_uint8_64 = awkward_reduce_sum
awkward_reduce_sum_uint32_uint8_64 = awkward_reduce_sum
awkward_reduce_sum_int64_int64_64 = awkward_reduce_sum
awkward_reduce_sum_int64_int32_64 = awkward_reduce_sum
awkward_reduce_sum_uint64_uint16_64 = awkward_reduce_sum
awkward_reduce_sum_float32_float32_64 = awkward_reduce_sum
awkward_reduce_sum_uint32_uint16_64 = awkward_reduce_sum
awkward_reduce_sum_int32_int8_64 = awkward_reduce_sum
awkward_reduce_sum_uint64_uint32_64 = awkward_reduce_sum
awkward_reduce_sum_int32_int16_64 = awkward_reduce_sum
awkward_reduce_sum_uint32_uint32_64 = awkward_reduce_sum
awkward_reduce_sum_uint64_uint64_64 = awkward_reduce_sum
awkward_reduce_sum_float64_float64_64 = awkward_reduce_sum
awkward_reduce_sum_int64_int8_64 = awkward_reduce_sum
awkward_reduce_sum_int64_int16_64 = awkward_reduce_sum
awkward_reduce_sum_int32_int32_64 = awkward_reduce_sum


def awkward_reduce_sum_int64_bool_64(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = 0
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] += fromptr[fromptroffset + i] != 0

def awkward_reduce_sum_int32_bool_64(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = 0
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] += fromptr[fromptroffset + i] != 0

def awkward_reduce_sum_bool(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = int(0)
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] |= fromptr[fromptroffset + i] != 0

awkward_reduce_sum_bool_uint8_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_uint16_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_uint64_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_int32_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_bool_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_uint32_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_float32_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_int16_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_int8_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_int64_64 = awkward_reduce_sum_bool
awkward_reduce_sum_bool_float64_64 = awkward_reduce_sum_bool


def awkward_reduce_prod(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = float(1)
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] *= float(fromptr[fromptroffset + i])

awkward_reduce_prod_int64_int16_64 = awkward_reduce_prod
awkward_reduce_prod_int64_int64_64 = awkward_reduce_prod
awkward_reduce_prod_int32_int16_64 = awkward_reduce_prod
awkward_reduce_prod_int64_int8_64 = awkward_reduce_prod
awkward_reduce_prod_float64_float64_64 = awkward_reduce_prod
awkward_reduce_prod_uint32_uint32_64 = awkward_reduce_prod
awkward_reduce_prod_uint64_uint64_64 = awkward_reduce_prod
awkward_reduce_prod_float32_float32_64 = awkward_reduce_prod
awkward_reduce_prod_int32_int32_64 = awkward_reduce_prod
awkward_reduce_prod_uint32_uint8_64 = awkward_reduce_prod
awkward_reduce_prod_uint64_uint8_64 = awkward_reduce_prod
awkward_reduce_prod_uint32_uint16_64 = awkward_reduce_prod
awkward_reduce_prod_uint64_uint16_64 = awkward_reduce_prod
awkward_reduce_prod_int32_int8_64 = awkward_reduce_prod
awkward_reduce_prod_int64_int32_64 = awkward_reduce_prod
awkward_reduce_prod_uint64_uint32_64 = awkward_reduce_prod


def awkward_reduce_prod_int64_bool_64(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = 1
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] *= fromptr[fromptroffset + i] != 0

def awkward_reduce_prod_int32_bool_64(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = 1
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] *= fromptr[fromptroffset + i] != 0

def awkward_reduce_prod_bool(
    toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = int(1)
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] &= fromptr[fromptroffset + i] != 0

awkward_reduce_prod_bool_bool_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_int8_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_float32_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_int16_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_uint64_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_uint8_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_int32_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_uint32_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_int64_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_float64_64 = awkward_reduce_prod_bool
awkward_reduce_prod_bool_uint16_64 = awkward_reduce_prod_bool


def awkward_reduce_min(
    toptr,
    fromptr,
    fromptroffset,
    parents,
    parentsoffset,
    lenparents,
    outlength,
    identity,
):
    for i in range(outlength):
        toptr[i] = identity
    for i in range(lenparents):
        x = fromptr[fromptroffset + i]
        toptr[parents[parentsoffset + i]] = (
            x
            if x < toptr[parents[parentsoffset + i]]
            else toptr[parents[parentsoffset + i]]
        )

awkward_reduce_min_uint8_uint8_64 = awkward_reduce_min
awkward_reduce_min_uint32_uint32_64 = awkward_reduce_min
awkward_reduce_min_int8_int8_64 = awkward_reduce_min
awkward_reduce_min_float32_float32_64 = awkward_reduce_min
awkward_reduce_min_int64_int64_64 = awkward_reduce_min
awkward_reduce_min_int16_int16_64 = awkward_reduce_min
awkward_reduce_min_int32_int32_64 = awkward_reduce_min
awkward_reduce_min_uint16_uint16_64 = awkward_reduce_min
awkward_reduce_min_float64_float64_64 = awkward_reduce_min
awkward_reduce_min_uint64_uint64_64 = awkward_reduce_min


def awkward_reduce_max(
    toptr,
    fromptr,
    fromptroffset,
    parents,
    parentsoffset,
    lenparents,
    outlength,
    identity,
):
    for i in range(outlength):
        toptr[i] = identity
    for i in range(lenparents):
        x = fromptr[fromptroffset + i]
        toptr[parents[parentsoffset + i]] = (
            x
            if x > toptr[parents[parentsoffset + i]]
            else toptr[parents[parentsoffset + i]]
        )

awkward_reduce_max_int64_int64_64 = awkward_reduce_max
awkward_reduce_max_uint16_uint16_64 = awkward_reduce_max
awkward_reduce_max_float32_float32_64 = awkward_reduce_max
awkward_reduce_max_float64_float64_64 = awkward_reduce_max
awkward_reduce_max_int32_int32_64 = awkward_reduce_max
awkward_reduce_max_int16_int16_64 = awkward_reduce_max
awkward_reduce_max_uint64_uint64_64 = awkward_reduce_max
awkward_reduce_max_int8_int8_64 = awkward_reduce_max
awkward_reduce_max_uint8_uint8_64 = awkward_reduce_max
awkward_reduce_max_uint32_uint32_64 = awkward_reduce_max


def awkward_reduce_argmin(
    toptr,
    fromptr,
    fromptroffset,
    starts,
    startsoffset,
    parents,
    parentsoffset,
    lenparents,
    outlength,
):
    for i in range(outlength):
        toptr[i] = -1
    for i in range(lenparents):
        parent = parents[parentsoffset + i]
        start = starts[startsoffset + parent]
        if (toptr[parent] == -1) or (
            fromptr[fromptroffset + i]
            < fromptr[(fromptroffset + toptr[parent]) + start]
        ):
            toptr[parent] = i - start

awkward_reduce_argmin_int32_64 = awkward_reduce_argmin
awkward_reduce_argmin_uint64_64 = awkward_reduce_argmin
awkward_reduce_argmin_int8_64 = awkward_reduce_argmin
awkward_reduce_argmin_int16_64 = awkward_reduce_argmin
awkward_reduce_argmin_uint8_64 = awkward_reduce_argmin
awkward_reduce_argmin_uint16_64 = awkward_reduce_argmin
awkward_reduce_argmin_float64_64 = awkward_reduce_argmin
awkward_reduce_argmin_uint32_64 = awkward_reduce_argmin
awkward_reduce_argmin_int64_64 = awkward_reduce_argmin
awkward_reduce_argmin_float32_64 = awkward_reduce_argmin


def awkward_reduce_argmin_bool_64(
    toptr,
    fromptr,
    fromptroffset,
    starts,
    startsoffset,
    parents,
    parentsoffset,
    lenparents,
    outlength,
):
    for i in range(outlength):
        toptr[i] = -1
    for i in range(lenparents):
        parent = parents[parentsoffset + i]
        start = starts[startsoffset + parent]
        if (toptr[parent] == -1) or (
            (fromptr[fromptroffset + i] != 0)
            < (fromptr[(fromptroffset + toptr[parent]) + start] != 0)
        ):
            toptr[parent] = i - start

def awkward_reduce_argmax(
    toptr,
    fromptr,
    fromptroffset,
    starts,
    startsoffset,
    parents,
    parentsoffset,
    lenparents,
    outlength,
):
    for i in range(outlength):
        toptr[i] = -1
    for i in range(lenparents):
        parent = parents[parentsoffset + i]
        start = starts[startsoffset + parent]
        if (toptr[parent] == -1) or (
            fromptr[fromptroffset + i]
            > fromptr[(fromptroffset + toptr[parent]) + start]
        ):
            toptr[parent] = i - start

awkward_reduce_argmax_uint64_64 = awkward_reduce_argmax
awkward_reduce_argmax_int64_64 = awkward_reduce_argmax
awkward_reduce_argmax_int32_64 = awkward_reduce_argmax
awkward_reduce_argmax_float64_64 = awkward_reduce_argmax
awkward_reduce_argmax_uint8_64 = awkward_reduce_argmax
awkward_reduce_argmax_uint16_64 = awkward_reduce_argmax
awkward_reduce_argmax_float32_64 = awkward_reduce_argmax
awkward_reduce_argmax_int8_64 = awkward_reduce_argmax
awkward_reduce_argmax_int16_64 = awkward_reduce_argmax
awkward_reduce_argmax_uint32_64 = awkward_reduce_argmax


def awkward_reduce_argmax_bool_64(
    toptr,
    fromptr,
    fromptroffset,
    starts,
    startsoffset,
    parents,
    parentsoffset,
    lenparents,
    outlength,
):
    for i in range(outlength):
        toptr[i] = -1
    for i in range(lenparents):
        parent = parents[parentsoffset + i]
        start = starts[startsoffset + parent]
        if (toptr[parent] == -1) or (
            (fromptr[fromptroffset + i] != 0)
            > (fromptr[(fromptroffset + toptr[parent]) + start] != 0)
        ):
            toptr[parent] = i - start

def awkward_content_reduce_zeroparents_64(toparents, length):
    for i in range(length):
        toparents[i] = 0

def awkward_ListOffsetArray_reduce_global_startstop_64(
    globalstart, globalstop, offsets, offsetsoffset, length
):
    globalstart[0] = offsets[offsetsoffset + 0]
    globalstop[0] = offsets[offsetsoffset + length]

def awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
    maxcount, offsetscopy, offsets, offsetsoffset, length
):
    maxcount[0] = 0
    offsetscopy[0] = offsets[offsetsoffset + 0]
    for i in range(length):
        count = offsets[(offsetsoffset + i) + 1] - offsets[offsetsoffset + i]
        if maxcount[0] < count:
            maxcount[0] = count
        offsetscopy[i + 1] = offsets[(offsetsoffset + i) + 1]

def awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
    nextstarts, nextparents, nextlen
):
    lastnextparent = -1
    for k in range(nextlen):
        if nextparents[k] != lastnextparent:
            nextstarts[nextparents[k]] = k
        lastnextparent = nextparents[k]

def awkward_ListOffsetArray_reduce_nonlocal_findgaps_64(
    gaps, parents, parentsoffset, lenparents
):
    k = 0
    last = -1
    for i in range(lenparents):
        parent = parents[parentsoffset + i]
        if last < parent:
            gaps[k] = parent - last
            k = k + 1
            last = parent

def awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
    outstarts, outstops, distincts, lendistincts, gaps, outlength
):
    j = 0
    k = 0
    maxdistinct = -1
    for i in range(lendistincts):
        if maxdistinct < distincts[i]:
            maxdistinct = distincts[i]
            for gappy in range(gaps[j]):
                outstarts[k] = i
                outstops[k] = i
                k = k + 1
            j = j + 1
        if distincts[i] != -1:
            outstops[k - 1] = i + 1
    while k < outlength:
        outstarts[k] = lendistincts + 1
        outstops[k] = lendistincts + 1
        k = k + 1

def awkward_ListOffsetArray_reduce_local_nextparents_64(
    nextparents, offsets, offsetsoffset, length
):
    initialoffset = offsets[offsetsoffset]
    for i in range(length):
        j = offsets[offsetsoffset + i] - initialoffset
        while j < (offsets[(offsetsoffset + i) + 1] - initialoffset):
            nextparents[j] = i
            j = j + 1

def awkward_ListOffsetArray_reduce_local_outoffsets_64(
    outoffsets, parents, parentsoffset, lenparents, outlength
):
    k = 0
    last = -1
    for i in range(lenparents):
        while last < parents[parentsoffset + i]:
            outoffsets[k] = i
            k = k + 1
            last = last + 1

    while k <= outlength:
        outoffsets[k] = lenparents
        k = k + 1

def awkward_IndexedArray_reduce_next_64(
    nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length
):
    k = 0
    for i in range(length):
        if index[indexoffset + i] >= 0:
            nextcarry[k] = index[indexoffset + i]
            nextparents[k] = parents[parentsoffset + i]
            outindex[i] = k
            k = k + 1
        else:
            outindex[i] = -1

awkward_IndexedArrayU32_reduce_next_64 = awkward_IndexedArray_reduce_next_64
awkward_IndexedArray64_reduce_next_64 = awkward_IndexedArray_reduce_next_64
awkward_IndexedArray32_reduce_next_64 = awkward_IndexedArray_reduce_next_64


def awkward_IndexedArray_reduce_next_fix_offsets_64(
    outoffsets, starts, startsoffset, startslength, outindexlength
):
    for i in range(startslength):
        outoffsets[i] = starts[startsoffset + i]
    outoffsets[startsoffset + startslength] = outindexlength

def awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
    toptr, parents, parentsoffset, lenparents, outlength
):
    for i in range(outlength):
        toptr[i] = 1
    for i in range(lenparents):
        toptr[parents[parentsoffset + i]] = 0

def awkward_ByteMaskedArray_reduce_next_64(
    nextcarry,
    nextparents,
    outindex,
    mask,
    maskoffset,
    parents,
    parentsoffset,
    length,
    validwhen,
):
    k = 0
    for i in range(length):
        if (mask[maskoffset + i] != 0) == validwhen:
            nextcarry[k] = i
            nextparents[k] = parents[parentsoffset + i]
            outindex[i] = k
            k = k + 1
        else:
            outindex[i] = -1

def awkward_regularize_arrayslice(flatheadptr, lenflathead, length):
    for i in range(lenflathead):
        original = flatheadptr[i]
        if flatheadptr[i] < 0:
            flatheadptr[i] += length
        if (flatheadptr[i] < 0) or (flatheadptr[i] >= length):
            raise ValueError("index out of range")

awkward_regularize_arrayslice_64 = awkward_regularize_arrayslice


def awkward_Index8_to_Index64(toptr, fromptr, length):
    for i in range(length):
        toptr[i] = int(fromptr[i])

def awkward_IndexU8_to_Index64(toptr, fromptr, length):
    for i in range(length):
        toptr[i] = int(fromptr[i])

def awkward_Index32_to_Index64(toptr, fromptr, length):
    for i in range(length):
        toptr[i] = int(fromptr[i])

def awkward_IndexU32_to_Index64(toptr, fromptr, length):
    for i in range(length):
        toptr[i] = int(fromptr[i])

def awkward_index_carry(
    toindex, fromindex, carry, fromindexoffset, lenfromindex, length
):
    for i in range(length):
        j = carry[i]
        if j > lenfromindex:
            raise ValueError("index out of range")
        toindex[i] = fromindex[fromindexoffset + j]

awkward_IndexU8_carry_64 = awkward_index_carry
awkward_Index32_carry_64 = awkward_index_carry
awkward_Index64_carry_64 = awkward_index_carry
awkward_IndexU32_carry_64 = awkward_index_carry
awkward_Index8_carry_64 = awkward_index_carry


def awkward_index_carry_nocheck(toindex, fromindex, carry, fromindexoffset, length):
    for i in range(length):
        toindex[i] = fromindex[fromindexoffset + carry[i]]

awkward_Index64_carry_nocheck_64 = awkward_index_carry_nocheck
awkward_Index32_carry_nocheck_64 = awkward_index_carry_nocheck
awkward_IndexU8_carry_nocheck_64 = awkward_index_carry_nocheck
awkward_IndexU32_carry_nocheck_64 = awkward_index_carry_nocheck
awkward_Index8_carry_nocheck_64 = awkward_index_carry_nocheck


def awkward_slicearray_ravel(toptr, fromptr, ndim, shape, strides):
    if ndim == 1:
        for i in range(shape[0]):
            toptr[i] = fromptr[i * strides[0]]
    else:
        for i in range(shape[0]):
            err = awkward_slicearray_ravel(
                toptr[i * shape[1]],
                fromptr[i * strides[0]],
                ndim - 1,
                shape[1],
                strides[1],
            )
            if err.str != nullptr:
                return err

awkward_slicearray_ravel_64 = awkward_slicearray_ravel


def awkward_slicemissing_check_same(
    same, bytemask, bytemaskoffset, missingindex, missingindexoffset, length
):
    same[0] = True
    for i in range(length):
        left = bytemask[bytemaskoffset + i] != 0
        right = missingindex[missingindexoffset + i] < 0
        if left != right:
            same[0] = False
            return

def awkward_carry_arange(toptr, length):
    for i in range(length):
        toptr[i] = i

awkward_carry_arange32 = awkward_carry_arange
awkward_carry_arange64 = awkward_carry_arange
awkward_carry_arangeU32 = awkward_carry_arange


def awkward_Identities_getitem_carry(
    newidentitiesptr, identitiesptr, carryptr, lencarry, offset, width, length
):
    for i in range(lencarry):
        if carryptr[i] >= length:
            raise ValueError("index out of range")
        for j in range(width):
            newidentitiesptr[(width * i) + j] = identitiesptr[
                (offset + (width * carryptr[i])) + j
            ]

awkward_Identities64_getitem_carry_64 = awkward_Identities_getitem_carry
awkward_Identities32_getitem_carry_64 = awkward_Identities_getitem_carry


def awkward_NumpyArray_contiguous_init(toptr, skip, stride):
    for i in range(skip):
        toptr[i] = i * stride

awkward_NumpyArray_contiguous_init_64 = awkward_NumpyArray_contiguous_init


def awkward_NumpyArray_contiguous_next(topos, frompos, length, skip, stride):
    for i in range(length):
        for j in range(skip):
            topos[(i * skip) + j] = frompos[i] + (j * stride)

awkward_NumpyArray_contiguous_next_64 = awkward_NumpyArray_contiguous_next


def awkward_NumpyArray_getitem_next_at(nextcarryptr, carryptr, lencarry, skip, at):
    for i in range(lencarry):
        nextcarryptr[i] = (skip * carryptr[i]) + at

awkward_NumpyArray_getitem_next_at_64 = awkward_NumpyArray_getitem_next_at


def awkward_NumpyArray_getitem_next_range(
    nextcarryptr, carryptr, lencarry, lenhead, skip, start, step
):
    for i in range(lencarry):
        for j in range(lenhead):
            nextcarryptr[(i * lenhead) + j] = ((skip * carryptr[i]) + start) + (
                j * step
            )

awkward_NumpyArray_getitem_next_range_64 = awkward_NumpyArray_getitem_next_range


def awkward_NumpyArray_getitem_next_range_advanced(
    nextcarryptr,
    nextadvancedptr,
    carryptr,
    advancedptr,
    lencarry,
    lenhead,
    skip,
    start,
    step,
):
    for i in range(lencarry):
        for j in range(lenhead):
            nextcarryptr[(i * lenhead) + j] = ((skip * carryptr[i]) + start) + (
                j * step
            )
            nextadvancedptr[(i * lenhead) + j] = advancedptr[i]

awkward_NumpyArray_getitem_next_range_advanced_64 = awkward_NumpyArray_getitem_next_range_advanced


def awkward_NumpyArray_getitem_next_array(
    nextcarryptr, nextadvancedptr, carryptr, flatheadptr, lencarry, lenflathead, skip
):
    for i in range(lencarry):
        for j in range(lenflathead):
            nextcarryptr[(i * lenflathead) + j] = (skip * carryptr[i]) + flatheadptr[j]
            nextadvancedptr[(i * lenflathead) + j] = j

awkward_NumpyArray_getitem_next_array_64 = awkward_NumpyArray_getitem_next_array


def awkward_NumpyArray_getitem_next_array_advanced(
    nextcarryptr, carryptr, advancedptr, flatheadptr, lencarry, skip
):
    for i in range(lencarry):
        nextcarryptr[i] = (skip * carryptr[i]) + flatheadptr[advancedptr[i]]

awkward_NumpyArray_getitem_next_array_advanced_64 = awkward_NumpyArray_getitem_next_array_advanced


def awkward_NumpyArray_getitem_boolean_numtrue(
    numtrue, fromptr, byteoffset, length, stride
):
    numtrue[0] = 0
    i = 0
    while i < length:
        numtrue[0] = numtrue[0] + (fromptr[byteoffset + i] != 0)
        i += stride

def awkward_NumpyArray_getitem_boolean_nonzero(
    toptr, fromptr, byteoffset, length, stride
):
    k = 0
    i = 0
    while i < length:
        if fromptr[byteoffset + i] != 0:
            toptr[k] = i
            k = k + 1
        i += stride

awkward_NumpyArray_getitem_boolean_nonzero_64 = awkward_NumpyArray_getitem_boolean_nonzero


def awkward_ListArray_getitem_next_at(
    tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at
):
    for i in range(lenstarts):
        length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
        regular_at = at
        if regular_at < 0:
            regular_at += length
        if not ((0 <= regular_at) and (regular_at < length)):
            raise ValueError("index out of range")
        tocarry[i] = fromstarts[startsoffset + i] + regular_at

awkward_ListArray64_getitem_next_at_64 = awkward_ListArray_getitem_next_at
awkward_ListArrayU32_getitem_next_at_64 = awkward_ListArray_getitem_next_at
awkward_ListArray32_getitem_next_at_64 = awkward_ListArray_getitem_next_at


def awkward_ListArray_getitem_next_range_carrylength(
    carrylength,
    fromstarts,
    fromstops,
    lenstarts,
    startsoffset,
    stopsoffset,
    start,
    stop,
    step,
):
    carrylength[0] = 0
    for i in range(lenstarts):
        length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
        regular_start = start
        regular_stop = stop
        awkward_regularize_rangeslice(
            regular_start,
            regular_stop,
            step > 0,
            start != kSliceNone,
            stop != kSliceNone,
            length,
        )
        if step > 0:
            j = regular_start
            while j < regular_stop:
                carrylength[0] = carrylength[0] + 1
                j += step
        else:
            j = regular_start
            while j > regular_stop:
                carrylength[0] = carrylength[0] + 1
                j += step

awkward_ListArray64_getitem_next_range_carrylength = awkward_ListArray_getitem_next_range_carrylength
awkward_ListArrayU32_getitem_next_range_carrylength = awkward_ListArray_getitem_next_range_carrylength
awkward_ListArray32_getitem_next_range_carrylength = awkward_ListArray_getitem_next_range_carrylength


def awkward_ListArray_getitem_next_range(
    tooffsets,
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    startsoffset,
    stopsoffset,
    start,
    stop,
    step,
):
    k = 0
    tooffsets[0] = 0
    if step > 0:
        for i in range(lenstarts):
            length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
            regular_start = start
            regular_stop = stop
            awkward_regularize_rangeslice(
                regular_start,
                regular_stop,
                step > 0,
                start != kSliceNone,
                stop != kSliceNone,
                length,
            )
            j = regular_start
            while j < regular_stop:
                tocarry[k] = fromstarts[startsoffset + i] + j
                k = k + 1
                j += step
            tooffsets[i + 1] = float(k)
    else:
        for i in range(lenstarts):
            length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
            regular_start = start
            regular_stop = stop
            awkward_regularize_rangeslice(
                regular_start,
                regular_stop,
                step > 0,
                start != kSliceNone,
                stop != kSliceNone,
                length,
            )
            j = regular_start
            while j > regular_stop:
                tocarry[k] = fromstarts[startsoffset + i] + j
                k = k + 1
                j += step
            tooffsets[i + 1] = float(k)

awkward_ListArray64_getitem_next_range_64 = awkward_ListArray_getitem_next_range
awkward_ListArrayU32_getitem_next_range_64 = awkward_ListArray_getitem_next_range
awkward_ListArray32_getitem_next_range_64 = awkward_ListArray_getitem_next_range


def awkward_ListArray_getitem_next_range_counts(total, fromoffsets, lenstarts):
    total[0] = 0
    for i in range(lenstarts):
        total[0] = (total[0] + fromoffsets[i + 1]) - fromoffsets[i]

awkward_ListArray64_getitem_next_range_counts_64 = awkward_ListArray_getitem_next_range_counts
awkward_ListArray32_getitem_next_range_counts_64 = awkward_ListArray_getitem_next_range_counts
awkward_ListArrayU32_getitem_next_range_counts_64 = awkward_ListArray_getitem_next_range_counts


def awkward_ListArray_getitem_next_range_spreadadvanced(
    toadvanced, fromadvanced, fromoffsets, lenstarts
):
    for i in range(lenstarts):
        count = fromoffsets[i + 1] - fromoffsets[i]
        for j in range(count):
            toadvanced[fromoffsets[i] + j] = fromadvanced[i]

awkward_ListArrayU32_getitem_next_range_spreadadvanced_64 = awkward_ListArray_getitem_next_range_spreadadvanced
awkward_ListArray64_getitem_next_range_spreadadvanced_64 = awkward_ListArray_getitem_next_range_spreadadvanced
awkward_ListArray32_getitem_next_range_spreadadvanced_64 = awkward_ListArray_getitem_next_range_spreadadvanced


def awkward_ListArray_getitem_next_array(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    startsoffset,
    stopsoffset,
    lenstarts,
    lenarray,
    lencontent,
):
    for i in range(lenstarts):
        if fromstops[stopsoffset + i] < fromstarts[startsoffset + i]:
            raise ValueError("stops[i] < starts[i]")
        if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]) and (
            fromstops[stopsoffset + i] > lencontent
        ):
            raise ValueError("stops[i] > len(content)")
        length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
        for j in range(lenarray):
            regular_at = fromarray[j]
            if regular_at < 0:
                regular_at += length
            if not ((0 <= regular_at) and (regular_at < length)):
                raise ValueError("index out of range")
            tocarry[(i * lenarray) + j] = fromstarts[startsoffset + i] + regular_at
            toadvanced[(i * lenarray) + j] = j

awkward_ListArray32_getitem_next_array_64 = awkward_ListArray_getitem_next_array
awkward_ListArrayU32_getitem_next_array_64 = awkward_ListArray_getitem_next_array
awkward_ListArray64_getitem_next_array_64 = awkward_ListArray_getitem_next_array


def awkward_ListArray_getitem_next_array_advanced(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    fromadvanced,
    startsoffset,
    stopsoffset,
    lenstarts,
    lenarray,
    lencontent,
):
    for i in range(lenstarts):
        if fromstops[stopsoffset + i] < fromstarts[startsoffset + i]:
            raise ValueError("stops[i] < starts[i]")
        if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]) and (
            fromstops[stopsoffset + i] > lencontent
        ):
            raise ValueError("stops[i] > len(content)")
        length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i]
        regular_at = fromarray[fromadvanced[i]]
        if regular_at < 0:
            regular_at += length
        if not ((0 <= regular_at) and (regular_at < length)):
            raise ValueError("index out of range")
        tocarry[i] = fromstarts[startsoffset + i] + regular_at
        toadvanced[i] = i

awkward_ListArray32_getitem_next_array_advanced_64 = awkward_ListArray_getitem_next_array_advanced
awkward_ListArray64_getitem_next_array_advanced_64 = awkward_ListArray_getitem_next_array_advanced
awkward_ListArrayU32_getitem_next_array_advanced_64 = awkward_ListArray_getitem_next_array_advanced


def awkward_ListArray_getitem_carry(
    tostarts,
    tostops,
    fromstarts,
    fromstops,
    fromcarry,
    startsoffset,
    stopsoffset,
    lenstarts,
    lencarry,
):
    for i in range(lencarry):
        if fromcarry[i] >= lenstarts:
            raise ValueError("index out of range")
        tostarts[i] = float(fromstarts[startsoffset + fromcarry[i]])
        tostops[i] = float(fromstops[stopsoffset + fromcarry[i]])

awkward_ListArray64_getitem_carry_64 = awkward_ListArray_getitem_carry
awkward_ListArray32_getitem_carry_64 = awkward_ListArray_getitem_carry
awkward_ListArrayU32_getitem_carry_64 = awkward_ListArray_getitem_carry


def awkward_RegularArray_getitem_next_at(tocarry, at, length, size):
    regular_at = at
    if regular_at < 0:
        regular_at += size
    if not ((0 <= regular_at) and (regular_at < size)):
        raise ValueError("index out of range")
    for i in range(length):
        tocarry[i] = (i * size) + regular_at

awkward_RegularArray_getitem_next_at_64 = awkward_RegularArray_getitem_next_at


def awkward_RegularArray_getitem_next_range(
    tocarry, regular_start, step, length, size, nextsize
):
    for i in range(length):
        for j in range(nextsize):
            tocarry[(i * nextsize) + j] = ((i * size) + regular_start) + (j * step)

awkward_RegularArray_getitem_next_range_64 = awkward_RegularArray_getitem_next_range


def awkward_RegularArray_getitem_next_range_spreadadvanced(
    toadvanced, fromadvanced, length, nextsize
):
    for i in range(length):
        for j in range(nextsize):
            toadvanced[(i * nextsize) + j] = fromadvanced[i]

awkward_RegularArray_getitem_next_range_spreadadvanced_64 = awkward_RegularArray_getitem_next_range_spreadadvanced


def awkward_RegularArray_getitem_next_array_regularize(
    toarray, fromarray, lenarray, size
):
    for j in range(lenarray):
        toarray[j] = fromarray[j]
        if toarray[j] < 0:
            toarray[j] += size
        if not ((0 <= toarray[j]) and (toarray[j] < size)):
            raise ValueError("index out of range")

awkward_RegularArray_getitem_next_array_regularize_64 = awkward_RegularArray_getitem_next_array_regularize


def awkward_RegularArray_getitem_next_array(
    tocarry, toadvanced, fromarray, length, lenarray, size
):
    for i in range(length):
        for j in range(lenarray):
            tocarry[(i * lenarray) + j] = (i * size) + fromarray[j]
            toadvanced[(i * lenarray) + j] = j

awkward_RegularArray_getitem_next_array_64 = awkward_RegularArray_getitem_next_array


def awkward_RegularArray_getitem_next_array_advanced(
    tocarry, toadvanced, fromadvanced, fromarray, length, lenarray, size
):
    for i in range(length):
        tocarry[i] = (i * size) + fromarray[fromadvanced[i]]
        toadvanced[i] = i

awkward_RegularArray_getitem_next_array_advanced_64 = awkward_RegularArray_getitem_next_array_advanced


def awkward_RegularArray_getitem_carry(tocarry, fromcarry, lencarry, size):
    for i in range(lencarry):
        for j in range(size):
            tocarry[(i * size) + j] = (fromcarry[i] * size) + j

awkward_RegularArray_getitem_carry_64 = awkward_RegularArray_getitem_carry


def awkward_IndexedArray_numnull(numnull, fromindex, indexoffset, lenindex):
    numnull[0] = 0
    for i in range(lenindex):
        if fromindex[indexoffset + i] < 0:
            numnull[0] = numnull[0] + 1

awkward_IndexedArray64_numnull = awkward_IndexedArray_numnull
awkward_IndexedArray32_numnull = awkward_IndexedArray_numnull
awkward_IndexedArrayU32_numnull = awkward_IndexedArray_numnull


def awkward_IndexedArray_getitem_nextcarry_outindex(
    tocarry, toindex, fromindex, indexoffset, lenindex, lencontent
):
    k = 0
    for i in range(lenindex):
        j = fromindex[indexoffset + i]
        if j >= lencontent:
            raise ValueError("index out of range")
        else:
            if j < 0:
                toindex[i] = -1
            else:
                tocarry[k] = j
                toindex[i] = float(k)
                k = k + 1

awkward_IndexedArrayU32_getitem_nextcarry_outindex_64 = awkward_IndexedArray_getitem_nextcarry_outindex
awkward_IndexedArray32_getitem_nextcarry_outindex_64 = awkward_IndexedArray_getitem_nextcarry_outindex
awkward_IndexedArray64_getitem_nextcarry_outindex_64 = awkward_IndexedArray_getitem_nextcarry_outindex


def awkward_IndexedArray_getitem_nextcarry_outindex_mask(
    tocarry, toindex, fromindex, indexoffset, lenindex, lencontent
):
    k = 0
    for i in range(lenindex):
        j = fromindex[indexoffset + i]
        if j >= lencontent:
            raise ValueError("index out of range")
        else:
            if j < 0:
                toindex[i] = -1
            else:
                tocarry[k] = j
                toindex[i] = float(k)
                k = k + 1

awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64 = awkward_IndexedArray_getitem_nextcarry_outindex_mask
awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64 = awkward_IndexedArray_getitem_nextcarry_outindex_mask
awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64 = awkward_IndexedArray_getitem_nextcarry_outindex_mask


def awkward_ListOffsetArray_getitem_adjust_offsets(
    tooffsets,
    tononzero,
    fromoffsets,
    offsetsoffset,
    length,
    nonzero,
    nonzerooffset,
    nonzerolength,
):
    j = 0
    tooffsets[0] = fromoffsets[offsetsoffset + 0]
    for i in range(length):
        slicestart = fromoffsets[offsetsoffset + i]
        slicestop = fromoffsets[(offsetsoffset + i) + 1]
        count = 0
        while (j < nonzerolength) and (nonzero[nonzerooffset + j] < slicestop):
            tononzero[j] = nonzero[nonzerooffset + j] - slicestart
            j = j + 1
            count = count + 1

        tooffsets[i + 1] = tooffsets[i] + count

awkward_ListOffsetArray_getitem_adjust_offsets_64 = awkward_ListOffsetArray_getitem_adjust_offsets


def awkward_ListOffsetArray_getitem_adjust_offsets_index(
    tooffsets,
    tononzero,
    fromoffsets,
    offsetsoffset,
    length,
    index,
    indexoffset,
    indexlength,
    nonzero,
    nonzerooffset,
    nonzerolength,
    originalmask,
    maskoffset,
    masklength,
):
    k = 0
    tooffsets[0] = fromoffsets[offsetsoffset + 0]
    for i in range(length):
        slicestart = fromoffsets[offsetsoffset + i]
        slicestop = fromoffsets[(offsetsoffset + i) + 1]
        numnull = 0
        for j in range(slicestart, slicestop):
            numnull += 1 if originalmask[maskoffset + j] != 0 else 0
        nullcount = 0
        count = 0
        while (k < indexlength) and (
            ((index[indexoffset + k] < 0) and (nullcount < numnull))
            or (
                (
                    (index[indexoffset + k] >= 0)
                    and (index[indexoffset + k] < nonzerolength)
                )
                and (nonzero[nonzerooffset + index[indexoffset + k]] < slicestop)
            )
        ):
            if index[indexoffset + k] < 0:
                nullcount = nullcount + 1
            else:
                j = index[indexoffset + k]
                tononzero[j] = nonzero[nonzerooffset + j] - slicestart
            k = k + 1
            count = count + 1

        tooffsets[i + 1] = tooffsets[i] + count

awkward_ListOffsetArray_getitem_adjust_offsets_index_64 = awkward_ListOffsetArray_getitem_adjust_offsets_index


def awkward_IndexedArray_getitem_adjust_outindex(
    tomask,
    toindex,
    tononzero,
    fromindex,
    fromindexoffset,
    fromindexlength,
    nonzero,
    nonzerooffset,
    nonzerolength,
):
    j = 0
    k = 0
    for i in range(fromindexlength):
        fromval = fromindex[fromindexoffset + i]
        tomask[i] = fromval < 0
        if fromval < 0:
            toindex[k] = -1
            k = k + 1
        else:
            if (j < nonzerolength) and (fromval == nonzero[nonzerooffset + j]):
                tononzero[j] = fromval + (k - j)
                toindex[k] = j
                j = j + 1
                k = k + 1

awkward_IndexedArray_getitem_adjust_outindex_64 = awkward_IndexedArray_getitem_adjust_outindex


def awkward_IndexedArray_getitem_nextcarry(
    tocarry, fromindex, indexoffset, lenindex, lencontent
):
    k = 0
    for i in range(lenindex):
        j = fromindex[indexoffset + i]
        if (j < 0) or (j >= lencontent):
            raise ValueError("index out of range")
        else:
            tocarry[k] = j
            k = k + 1

awkward_IndexedArray64_getitem_nextcarry_64 = awkward_IndexedArray_getitem_nextcarry
awkward_IndexedArrayU32_getitem_nextcarry_64 = awkward_IndexedArray_getitem_nextcarry
awkward_IndexedArray32_getitem_nextcarry_64 = awkward_IndexedArray_getitem_nextcarry


def awkward_IndexedArray_getitem_carry(
    toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry
):
    for i in range(lencarry):
        if fromcarry[i] >= lenindex:
            raise ValueError("index out of range")
        toindex[i] = float(fromindex[indexoffset + fromcarry[i]])

awkward_IndexedArrayU32_getitem_carry_64 = awkward_IndexedArray_getitem_carry
awkward_IndexedArray64_getitem_carry_64 = awkward_IndexedArray_getitem_carry
awkward_IndexedArray32_getitem_carry_64 = awkward_IndexedArray_getitem_carry


def awkward_UnionArray_regular_index_getsize(size, fromtags, tagsoffset, length):
    size[0] = 0
    for i in range(length):
        tag = int(fromtags[tagsoffset + i])
        if size[0] < tag:
            size[0] = tag
    size[0] = size[0] + 1

awkward_UnionArray8_regular_index_getsize = awkward_UnionArray_regular_index_getsize


def awkward_UnionArray_regular_index(
    toindex, current, size, fromtags, tagsoffset, length
):
    count = 0
    for k in range(size):
        current[k] = 0
    for i in range(length):
        tag = fromtags[tagsoffset + i]
        toindex[i] = current[tag]
        current[tag] = current[tag] + 1

awkward_UnionArray8_32_regular_index = awkward_UnionArray_regular_index
awkward_UnionArray8_U32_regular_index = awkward_UnionArray_regular_index
awkward_UnionArray8_64_regular_index = awkward_UnionArray_regular_index


def awkward_UnionArray_project(
    lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which
):
    lenout[0] = 0
    for i in range(length):
        if fromtags[tagsoffset + i] == which:
            tocarry[lenout[0]] = fromindex[indexoffset + i]
            lenout[0] = lenout[0] + 1

awkward_UnionArray8_U32_project_64 = awkward_UnionArray_project
awkward_UnionArray8_64_project_64 = awkward_UnionArray_project
awkward_UnionArray8_32_project_64 = awkward_UnionArray_project


def awkward_missing_repeat(
    outindex, index, indexoffset, indexlength, repetitions, regularsize
):
    for i in range(repetitions):
        for j in range(indexlength):
            base = index[indexoffset + j]
            outindex[(i * indexlength) + j] = base + i * regularsize if base >= 0 else 0

awkward_missing_repeat_64 = awkward_missing_repeat


def awkward_RegularArray_getitem_jagged_expand(
    multistarts, multistops, singleoffsets, regularsize, regularlength
):
    for i in range(regularlength):
        for j in range(regularsize):
            multistarts[(i * regularsize) + j] = singleoffsets[j]
            multistops[(i * regularsize) + j] = singleoffsets[j + 1]

awkward_RegularArray_getitem_jagged_expand_64 = awkward_RegularArray_getitem_jagged_expand


def awkward_ListArray_getitem_jagged_expand(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstartsoffset,
    fromstops,
    fromstopsoffset,
    jaggedsize,
    length,
):
    for i in range(length):
        start = fromstarts[fromstartsoffset + i]
        stop = fromstops[fromstopsoffset + i]
        if stop < start:
            raise ValueError("stops[i] < starts[i]")
        if (stop - start) != jaggedsize:
            raise ValueError("cannot fit jagged slice into nested list")
        for j in range(jaggedsize):
            multistarts[(i * jaggedsize) + j] = singleoffsets[j]
            multistops[(i * jaggedsize) + j] = singleoffsets[j + 1]
            tocarry[(i * jaggedsize) + j] = start + j

awkward_ListArrayU32_getitem_jagged_expand_64 = awkward_ListArray_getitem_jagged_expand
awkward_ListArray32_getitem_jagged_expand_64 = awkward_ListArray_getitem_jagged_expand
awkward_ListArray64_getitem_jagged_expand_64 = awkward_ListArray_getitem_jagged_expand


def awkward_ListArray_getitem_jagged_carrylen(
    carrylen,
    slicestarts,
    slicestartsoffset,
    slicestops,
    slicestopsoffset,
    sliceouterlen,
):
    carrylen[0] = 0
    for i in range(sliceouterlen):
        carrylen[0] = carrylen[0] + int(
            slicestops[slicestopsoffset + i] - slicestarts[slicestartsoffset + i]
        )

awkward_ListArray_getitem_jagged_carrylen_64 = awkward_ListArray_getitem_jagged_carrylen


def awkward_ListArray_getitem_jagged_apply(
    tooffsets,
    tocarry,
    slicestarts,
    slicestartsoffset,
    slicestops,
    slicestopsoffset,
    sliceouterlen,
    sliceindex,
    sliceindexoffset,
    sliceinnerlen,
    fromstarts,
    fromstartsoffset,
    fromstops,
    fromstopsoffset,
    contentlen,
):
    k = 0
    for i in range(sliceouterlen):
        slicestart = slicestarts[slicestartsoffset + i]
        slicestop = slicestops[slicestopsoffset + i]
        tooffsets[i] = float(k)
        if slicestart != slicestop:
            if slicestop < slicestart:
                raise ValueError("jagged slice's stops[i] < starts[i]")
            if slicestop > sliceinnerlen:
                raise ValueError("jagged slice's offsets extend beyond its content")
            start = int(fromstarts[fromstartsoffset + i])
            stop = int(fromstops[fromstopsoffset + i])
            if stop < start:
                raise ValueError("stops[i] < starts[i]")
            if (start != stop) and (stop > contentlen):
                raise ValueError("stops[i] > len(content)")
            count = stop - start
            for j in range(slicestart, slicestop):
                index = int(sliceindex[sliceindexoffset + j])
                if index < 0:
                    index += count
                if not ((0 <= index) and (index < count)):
                    raise ValueError("index out of range")
                tocarry[k] = start + index
                k = k + 1
        tooffsets[i + 1] = float(k)

awkward_ListArrayU32_getitem_jagged_apply_64 = awkward_ListArray_getitem_jagged_apply
awkward_ListArray64_getitem_jagged_apply_64 = awkward_ListArray_getitem_jagged_apply
awkward_ListArray32_getitem_jagged_apply_64 = awkward_ListArray_getitem_jagged_apply


def awkward_ListArray_getitem_jagged_numvalid(
    numvalid,
    slicestarts,
    slicestartsoffset,
    slicestops,
    slicestopsoffset,
    length,
    missing,
    missingoffset,
    missinglength,
):
    numvalid[0] = 0
    for i in range(length):
        slicestart = slicestarts[slicestartsoffset + i]
        slicestop = slicestops[slicestopsoffset + i]
        if slicestart != slicestop:
            if slicestop < slicestart:
                raise ValueError("jagged slice's stops[i] < starts[i]")
            if slicestop > missinglength:
                raise ValueError("jagged slice's offsets extend beyond its content")
            for j in range(slicestart, slicestop):
                numvalid[0] = numvalid[0] + 1 if missing[missingoffset + j] >= 0 else 0

awkward_ListArray_getitem_jagged_numvalid_64 = awkward_ListArray_getitem_jagged_numvalid


def awkward_ListArray_getitem_jagged_shrink(
    tocarry,
    tosmalloffsets,
    tolargeoffsets,
    slicestarts,
    slicestartsoffset,
    slicestops,
    slicestopsoffset,
    length,
    missing,
    missingoffset,
):
    k = 0
    if length == 0:
        tosmalloffsets[0] = 0
        tolargeoffsets[0] = 0
    else:
        tosmalloffsets[0] = slicestarts[slicestartsoffset + 0]
        tolargeoffsets[0] = slicestarts[slicestartsoffset + 0]
    for i in range(length):
        slicestart = slicestarts[slicestartsoffset + i]
        slicestop = slicestops[slicestopsoffset + i]
        if slicestart != slicestop:
            smallcount = 0
            for j in range(slicestart, slicestop):
                if missing[missingoffset + j] >= 0:
                    tocarry[k] = j
                    k = k + 1
                    smallcount = smallcount + 1
            tosmalloffsets[i + 1] = tosmalloffsets[i] + smallcount
        else:
            tosmalloffsets[i + 1] = tosmalloffsets[i]
        tolargeoffsets[i + 1] = tolargeoffsets[i] + (slicestop - slicestart)

awkward_ListArray_getitem_jagged_shrink_64 = awkward_ListArray_getitem_jagged_shrink


def awkward_ListArray_getitem_jagged_descend(
    tooffsets,
    slicestarts,
    slicestartsoffset,
    slicestops,
    slicestopsoffset,
    sliceouterlen,
    fromstarts,
    fromstartsoffset,
    fromstops,
    fromstopsoffset,
):
    if sliceouterlen == 0:
        tooffsets[0] = 0
    else:
        tooffsets[0] = slicestarts[slicestartsoffset + 0]
    for i in range(sliceouterlen):
        slicecount = int(
            slicestops[slicestopsoffset + i] - slicestarts[slicestartsoffset + i]
        )
        count = int(fromstops[fromstopsoffset + i] - fromstarts[fromstartsoffset + i])
        if slicecount != count:
            raise ValueError(
                "jagged slice inner length differs from array inner length"
            )
        tooffsets[i + 1] = tooffsets[i] + float(count)

awkward_ListArrayU32_getitem_jagged_descend_64 = awkward_ListArray_getitem_jagged_descend
awkward_ListArray64_getitem_jagged_descend_64 = awkward_ListArray_getitem_jagged_descend
awkward_ListArray32_getitem_jagged_descend_64 = awkward_ListArray_getitem_jagged_descend


def awkward_ByteMaskedArray_getitem_carry(
    tomask, frommask, frommaskoffset, lenmask, fromcarry, lencarry
):
    for i in range(lencarry):
        if fromcarry[i] >= lenmask:
            raise ValueError("index out of range")
        tomask[i] = frommask[frommaskoffset + fromcarry[i]]

awkward_ByteMaskedArray_getitem_carry_64 = awkward_ByteMaskedArray_getitem_carry


def awkward_ByteMaskedArray_numnull(numnull, mask, maskoffset, length, validwhen):
    numnull[0] = 0
    for i in range(length):
        if (mask[maskoffset + i] != 0) != validwhen:
            numnull[0] = numnull[0] + 1

def awkward_ByteMaskedArray_getitem_nextcarry(
    tocarry, mask, maskoffset, length, validwhen
):
    k = 0
    for i in range(length):
        if (mask[maskoffset + i] != 0) == validwhen:
            tocarry[k] = i
            k = k + 1

awkward_ByteMaskedArray_getitem_nextcarry_64 = awkward_ByteMaskedArray_getitem_nextcarry


def awkward_ByteMaskedArray_getitem_nextcarry_outindex(
    tocarry, toindex, mask, maskoffset, length, validwhen
):
    k = 0
    for i in range(length):
        if (mask[maskoffset + i] != 0) == validwhen:
            tocarry[k] = i
            toindex[i] = float(k)
            k = k + 1
        else:
            toindex[i] = -1

awkward_ByteMaskedArray_getitem_nextcarry_outindex_64 = awkward_ByteMaskedArray_getitem_nextcarry_outindex


def awkward_ByteMaskedArray_toIndexedOptionArray(
    toindex, mask, maskoffset, length, validwhen
):
    for i in range(length):
        toindex[i] = i if (mask[maskoffset + i] != 0) == validwhen else -1

awkward_ByteMaskedArray_toIndexedOptionArray64 = awkward_ByteMaskedArray_toIndexedOptionArray


def awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
    index_in,
    index_in_offset,
    offsets_in,
    offsets_in_offset,
    mask_out,
    starts_out,
    stops_out,
    length,
):
    k = 0
    i = 0
    while i < length:
        starts_out[i] = offsets_in[k + offsets_in_offset]
        if index_in[i + index_in_offset] < 0:
            mask_out[i] = -1
            stops_out[i] = offsets_in[k + offsets_in_offset]
        else:
            mask_out[i] = i
            k += 1
            stops_out[i] = offsets_in[k + offsets_in_offset]
        i = i + 1

def awkward_MaskedArray_getitem_next_jagged_project(
    index,
    index_offset,
    starts_in,
    starts_offset,
    stops_in,
    stops_offset,
    starts_out,
    stops_out,
    length,
):
    k = 0
    i = 0
    while i < length:
        if index[i + index_offset] >= 0:
            starts_out[k] = starts_in[i + starts_offset]
            stops_out[k] = stops_in[i + stops_offset]
            k = k + 1
        i = i + 1

awkward_MaskedArray32_getitem_next_jagged_project = awkward_MaskedArray_getitem_next_jagged_project
awkward_MaskedArray64_getitem_next_jagged_project = awkward_MaskedArray_getitem_next_jagged_project
awkward_MaskedArrayU32_getitem_next_jagged_project = awkward_MaskedArray_getitem_next_jagged_project


