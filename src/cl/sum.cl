#define VALUES_PER_WORK_ITEM 32
#define WORKGROUP_SIZE 128

__kernel void atomic_sum(__global const int *arr,
                         __global unsigned int *sum,
                         unsigned int n)
{
    unsigned int id = get_global_id(0);
    if (id < n)
    {
        atomic_add(sum, arr[id]);
    }
}

__kernel void loop_sum(__global const int *arr,
                      __global unsigned int *sum,
                      unsigned int n)
{
    const unsigned int idx = get_global_id(0);
    unsigned int res = 0;
    for (int i = idx * VALUES_PER_WORK_ITEM; i < (idx + 1) * VALUES_PER_WORK_ITEM; ++i)
    {
        if (i < n)
        {
            res += arr[i];
        }
    }

    atomic_add(sum, res);
}

__kernel void loop_coalesced_sum(__global const int *arr,
                                 __global unsigned int *sum,
                                 unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i)
    {
        int idx = wid * grs * VALUES_PER_WORK_ITEM + i * grs + lid;
        if (idx < n)
        {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void sum_local_mem(__global const int *arr,
                            __global unsigned int *sum,
                            unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0)
    {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i)
        {
            group_res += buf[i];
        }

        atomic_add(sum, group_res);
    }
}

__kernel void tree_sum(__global const int *arr,
                       __global unsigned int *sum,
                       const unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf [WORKGROUP_SIZE];
    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2)
    {
        if (2 * lid < nValues)
        {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        atomic_add(sum, buf[0]);
    }
}


