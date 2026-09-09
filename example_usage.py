from client import TensorCoreWMMA

def main():
    print("=== Testing Tensor Core WMMA Emulator ===")
    wmma = TensorCoreWMMA(m=2, n=2, k=2)
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = [[1, 1], [1, 1]]

    # D = A * B + C
    D = wmma.wmma_sync(A, B, C)
    print("Matrix A:", A)
    print("Matrix B:", B)
    print("Accumulator C:", C)
    print("WMMA Result D = A*B + C:", D)
    assert D == [[20, 23], [44, 51]]
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
