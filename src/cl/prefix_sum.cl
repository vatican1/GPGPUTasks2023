
__kernel void reduce(__global unsigned int *as, unsigned int block_size, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int ind = gid * 2 * block_size - 1;
    if(ind + 2 * block_size >= n)
        return;
     as[ind + 2 * block_size] += as[ind + block_size];
}

__kernel void sum(__global unsigned int *as, __global unsigned int *bs, unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int block_ind = ((gid / block_size) * 2 + 1) * block_size;
    bs[block_ind + gid % block_size - 1] += as[block_ind - 1];
}
