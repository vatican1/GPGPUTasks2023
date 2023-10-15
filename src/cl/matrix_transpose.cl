#define TILE_SIZE 16

__kernel void matrix_transpose(__global float* as,
                               __global float* as_t,
                               unsigned int m,
                               unsigned int k)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    if (global_j >= m || global_i >= k)
    {
        return;
    }

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if ((local_i >= TILE_SIZE) || (local_j >= TILE_SIZE))
    {
        return;
    }

    __local float tile[TILE_SIZE][TILE_SIZE];
    tile[local_i][local_j] = as[global_j * k + global_i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int residual_i = global_i % TILE_SIZE;
    int residual_j = global_j % TILE_SIZE;
    int int_part_i = global_i - residual_i;
    int int_part_j = global_j - residual_j;

    as_t[(int_part_i + residual_j) * m + int_part_j + residual_i] = tile[local_j][local_i];


}
