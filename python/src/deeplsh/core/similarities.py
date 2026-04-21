def get_index_sim(n_items, index_a, index_b):
    if index_a == index_b:
        raise ValueError("Pair indices must be different.")
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    return int(index_a * n_items - (index_a * (index_a + 1)) // 2 + (index_b - index_a - 1))


def get_indices_sim(n_items, pair_index):
    pair_index = int(pair_index)
    offset = 0
    for index_a in range(n_items - 1):
        row_size = n_items - index_a - 1
        if pair_index < offset + row_size:
            return index_a, index_a + 1 + (pair_index - offset)
        offset += row_size
    raise ValueError(f"Pair index out of range: {pair_index}")
