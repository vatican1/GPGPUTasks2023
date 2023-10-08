int calc_shif(__global float * as,
               const unsigned int left_,
               const unsigned int right_,
               const float value,
               bool flag)
{
    unsigned int left = left_;
    unsigned int right =  right_;
    unsigned int middle = (left + right) / 2;
    while (right > left)
    {
        if ((flag && (as[middle] >= value)) || as[middle] > value)
        {
            right = middle;
        }
        else
        {
            left = middle + 1;
        }
        middle = (left + right) / 2;
    }
    return left - left_;
}



__kernel void merge(__global float * as,
                    __global float * bs,
                    unsigned int k,
                    unsigned int n)
{
    int id = get_global_id(0);
    if (id >= n)
        return;

    float value = as[id];

    unsigned int left_start = id - id % (2 * k);
    unsigned int rigth_start = left_start + k;
    unsigned int left_end =  rigth_start;
    unsigned int right_end = (rigth_start + k <= n) ? rigth_start + k : n;
    unsigned int default_start = left_start;

    unsigned int new_index;
    if (id < rigth_start)
    {
        new_index =  default_start + calc_shif(as, rigth_start, right_end, value, true) + (id - left_start);
    }
    else
    {
        new_index =  default_start + calc_shif(as, left_start, left_end, value, false) + (id - rigth_start);
    }

     bs[new_index] = value;
}
