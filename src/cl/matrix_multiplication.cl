#define TILE_SIZE 16
#define THREAD_WORK 16

__kernel void matrix_multiplication(const __global float* a,
                                    const __global float* b,
                                    __global float* c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for(int k = 0; k < K; ++k)
    {
        sum += a[j * K + k]* b[k * N + i];
    }
    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_1(const __global float* a,
                                      const __global float* b,
                                      __global float* c,
                                      unsigned int M,
                                      unsigned int K,
                                      unsigned int N)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (global_i >= M || global_j > N || local_i > TILE_SIZE || local_j > TILE_SIZE)
    {
        return;
    }

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK)
    {
        tileA[local_j][local_i] = a[global_j * K + TILE_SIZE * tileK + local_i];
        tileB[local_j][local_i] = b[(TILE_SIZE * tileK + local_j) * N + global_i];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[global_j * N + global_i] = sum;

}
