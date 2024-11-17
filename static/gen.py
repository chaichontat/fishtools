# %%
import numpy as np

np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
from math import comb


# %%
def hamming(a: int, b: int):
    return (a ^ b).bit_count()


def bit_count(arr):
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = np.array(-1).astype(arr.dtype)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))


def gen_mhd(n: int, on: int, min_dist: int = 4, seed: int = 0):
    assert n < 32
    rand = np.random.default_rng(seed)

    while (s := rand.integers(0, 2**n - 1)).bit_count() != on:
        ...

    # out = [s]
    out = np.zeros((2 ** (n - 1)), dtype=np.uint32)  # space saving. we're not getting more than half.
    out[0] = s
    cnt = 1

    for i in range(2**n):
        if not (i.bit_count() == on):  # or i.bit_count() == 5):
            continue

        if np.any(bit_count(out[:cnt] ^ i) < min_dist):
            continue

        # if any(hamming(i, j) < min_dist for j in out[:cnt]):
        #     continue

        out[cnt] = i
        if ((np.array([out[cnt]])[:, None] & (1 << np.arange(n))) > 0).astype(int).sum() != on:
            raise ValueError(i)
        cnt += 1

    return out[:cnt]


def n_to_bit(arr, n: int, on: int):
    # n = int(np.max(arr)).bit_length()
    # assert 1 << (n + 1) > np.max(arr) >= 1 << (n)

    arr = ((arr[:, None] & (1 << np.arange(n))) > 0).astype(int)
    print(arr.shape)
    assert np.all(arr.sum(axis=1) == on)
    return arr


on = 3

min_dist = 2

for n in range(29, 31):
    for i in range(5):
        print(len(x := gen_mhd(n, on, seed=0, min_dist=min_dist)))

    np.savetxt(f"static/{n}bit_on{on}_dist{min_dist}.csv", n_to_bit(x, n, on), fmt="%d", delimiter=",")

# %%

# %%
# from itertools import product

# out = [np.array([0, 1, 2, 3, 2, 0])]
# for x in product(*([list(range(4))] * 6)):
#     if len([y for y in x if y == 0]) != 3:
#         continue

#     arr = np.array(x)
#     for prev in out:
#         if np.sum(arr != prev) < 1:
#             break
#     else:
#         out.append(arr)
# len(out)


# %%
def hamming_code(data_bits):
    # data_bits should be a numpy array of 11 bits
    assert len(data_bits) == 11, "Input should be 11 bits long"

    # Define the generator matrix for Hamming(15, 11)
    G = np.array(
        [
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # Calculate the code word
    code_word = np.dot(data_bits, G) % 2

    return code_word


from itertools import product

out = []
for i in product([0, 1], repeat=11):
    if (arr := np.array(i)).sum() > 4:
        continue

    if (res := hamming_code(arr)).sum() == 4:
        out.append(res)

# %%


def hamming(i, j):
    return np.sum(i ^ j)


# %%
mhd2 = np.loadtxt("static/MHD2_16bits.csv", delimiter=",", dtype=np.uint8)
# %%
out = [mhd2[14]]
for i in mhd2:
    for j in out:
        if (i == j).all():
            break
        if (i != j).sum() < 3:
            break
    else:
        out.append(i)
print(len(out))
# %%
idd = np.eye(4, 4, dtype=np.uint8)
# %%
out = []
for i in product(range(4), repeat=5):
    out.append(np.concatenate([idd[i[0]], idd[i[1]], idd[i[2]], idd[i[3]], idd[i[4]]]))
# %%

for i in out:
    for j in out:
        if i is j:
            continue
        assert hamming(i, j) >= 3, (i, j)
# %%
from itertools import combinations

import numpy as np
from bitarray import bitarray
from numba import njit


def generate_three_ones_numbers(bits=48):
    # Get all possible positions for the three 1s
    positions = range(bits)
    # Each combination represents positions of 1s
    for pos in combinations(positions, 3):
        # Create number by OR-ing shifted 1s
        num = sum(1 << p for p in pos)
        yield num


# Example usage:
numbers = list(generate_three_ones_numbers())
# Print first few numbers to verify
for num in numbers[:10]:
    print(f"{num:>4}: {bin(num)[2:]:>48}")


# %%
# @njit
def popcount(x):
    """HAKMEM or Wegner algorithm."""
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F


# @njit
def hamming_distance(x, y):
    return popcount(x ^ y)


@njit
def gray_code(i):
    return i ^ (i >> 1)


rand = np.random.default_rng(0)


# @njit
def generate_numbers(n, on, min_dist, rand):
    max_val = 2**n - 1

    # Find initial number
    while True:
        s = rand.integers(0, max_val)
        if popcount(s) == on:
            break

    n_possible = comb(n, 3)
    print(n_possible)

    out = np.zeros(n_possible, dtype=np.uint64)  # Pre-allocate with maximum possible size
    out[0] = s
    cnt = 1

    positions = range(n)
    for pos in combinations(positions, on):
        candidate = sum(1 << p for p in pos)
        # candidate = gray_code(i)
        # if popcount(candidate) != on:
        #     continue

        is_valid = True
        for j in range(cnt):
            if hamming_distance(int(candidate), int(out[j])) < min_dist:
                is_valid = False
                break

        if is_valid:
            out[cnt] = candidate
            cnt += 1

    return out[:cnt]


def bits_to_int(bits):
    return int(bits.to01(), 2)


def int_to_bits(x, n):
    return bitarray(f"{x:0{n}b}")


def memory_efficient_generate(n, on, min_dist, seed=None):
    numbers = generate_numbers(n, on, min_dist, seed)
    bit_arrays = [int_to_bits(num, n) for num in numbers]
    return bit_arrays


# Example usage
n, on, min_dist = 32, 4, 2
result = generate_numbers(n, on, min_dist, rand)
print(f"Generated {len(result)} numbers")
# print(result)
# %%
from itertools import combinations


def generate_three_ones_numbers(bits=48):
    positions = range(bits)
    for pos in combinations(positions, 3):
        num = sum(1 << p for p in pos)
        yield num
