class TensorCoreWMMA:
    """
    Simulates Tensor Core Warp Matrix Multiply-Accumulate (WMMA).
    Multiplies Tile A (MxK) and Tile B (KxN) and accumulates into Tile C (MxN): D = A * B + C.
    """
    def __init__(self, m=4, n=4, k=4):
        self.M = m
        self.N = n
        self.K = k

    def wmma_sync(self, a_frag, b_frag, c_frag):
        d_frag = [[c_frag[i][j] for j in range(self.N)] for i in range(self.M)]
        for i in range(self.M):
            for j in range(self.N):
                acc = 0.0
                for p in range(self.K):
                    acc += a_frag[i][p] * b_frag[p][j]
                d_frag[i][j] += acc
        return d_frag
