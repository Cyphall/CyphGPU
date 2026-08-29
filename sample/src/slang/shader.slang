cbuffer Parameters
{
	uint32_t u_loop_count;
	uint32_t* u_src_ptr;
	uint32_t* u_dst_ptr;
};

[[shader("compute")]]
[[numthreads(4, 1, 1)]]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint32_t total = 0;

	for (uint32_t i = 0; i < u_loop_count; i++)
		u_dst_ptr[tid.x] += u_src_ptr[i]; // crash, u_src_ptr is dereferenced even though u_loop_count is 0

	u_dst_ptr[tid.x] = total;
}