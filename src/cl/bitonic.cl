enum ORDER { Lower = 0, Upper = 1};

__kernel void bitonic(__global float *as,
                      unsigned int small_block_size,
                      unsigned int step,
                      unsigned int n)
{
    unsigned int gid = get_global_id(0);

    unsigned int shift = small_block_size / 2;
    unsigned int amount_prev_blocks = gid / shift;
    unsigned int ind = small_block_size * amount_prev_blocks + (gid % shift);
    unsigned int other_ind = ind + shift;

    unsigned int big_block_number = ind / step;
    enum ORDER order = (big_block_number & 1) ? Upper: Lower;
    if(other_ind < n)
    {
        if((order == Lower && as[ind] > as[other_ind]) ||
            (order == Upper && as[ind] < as[other_ind]))
        {
            float tmp = as[ind];                                                                                               \
            as[ind] = as[other_ind];                                                                                                 \
            as[other_ind] = tmp;
        }
    }
}
