import numpy as np

def mst(dists, n):
    ij = np.empty((n - 1, 2), dtype=np.int)
    Z = ij
    l = np.empty(n-1)
    l_ = l

    # Which nodes were already merged.
    merged = np.zeros(n, dtype=np.int)

    # Best distance of node i to current tree
    D = np.empty(n)
    D[:] = - np.inf
    j = np.empty(n, dtype=np.int)

    x = 0 # The node just added to the tree


    for k in range(n - 1):
        merged[x] = 1
        current_max = -np.inf
        for i in range(n):
            if merged[i] == 1:
                continue

            dist = dists[x,i]
            if D[i] < dist:
                D[i] = dist
                j[i] = x

            if current_max < D[i]:
                current_max = D[i]
                y = i

        # for linkage, this works if you assign it x instead, but the proof is subtle
        Z[k, 0] = j[y]
        # Z[k, 0] = x
        Z[k, 1] = y
        # Z[k, 2] = current_min
        l_[k] = current_max
        x = y

    # Sort Z by distances
    order = np.argsort(l, kind='mergesort')[::-1]
    ij = ij[order]
    l = l[order]
    return ij, l

def reorder( A,  idx, n):
    """
    A : (n, n)
    idx: (n)
    """
    B = np.empty((n, n))
    B_ = B
    for i in range(n):
        k = idx[i]
        for j in range(n):
            B_[i, j] = A[k,idx[j]]
    return B