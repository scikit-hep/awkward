import pytest
import awkward as ak


def test_executor():
    print("=== Basic lazy evaluation ===")
    arr = ak.Array([[1, 2, 3], [4, 5], [6, 7, 8, 9]], backend="cuda")

    # Create lazy wrapper
    lazy_arr =ak.cuda.lazy(arr)

    # Build computation graph without executing
    transformed = lazy_arr * 2 + 1
    result = transformed.filter(lazy_arr > 3)

    # Visualize the IR
    print("IR Graph:")
    print(result.visualize())
    print()

    # Execute the computation
    print("Computed result:")
    print(result.compute())
    print()

    # Alternative: filter based on the transformed values
    print("Alternative - filter transformed values > 5:")
    result2 = transformed.filter(transformed > 5)
    print(result2.compute())
    print()

    # Test combinations
    print("=== Combinations ===")
    arr2 = ak.Array([[1, 2, 3], [4, 5]], backend="cuda")
    lazy_arr2 = ak.cuda.lazy(arr2)

    # Generate all 2-element combinations
    pairs = lazy_arr2.combinations(2)
    print("IR for combinations(2):")
    print(pairs.visualize())
    print()
    print("Result:")
    print(pairs.compute())
    print()

    # Combinations with field access
    print("=== Field access on combinations ===")
    arr3 = ak.Array([[1, 2, 3], [4, 5]], backend="cuda")
    lazy_arr3 = ak.cuda.lazy(arr3)


    pairs = lazy_arr3.combinations(2)
    first = pairs['0']
    second = pairs['1']
    pair_sums = first + second

    print("IR for sum of pairs:")
    print(pair_sums.visualize())
    print()

    print("Result:")
    print(pair_sums.compute())

