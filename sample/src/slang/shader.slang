static const float3 LIGHT_DIR = {0.424, 0.566, 0.707};

cbuffer Parameters
{
    uint2 u_size;
    RWTexture2D::Handle u_image;
    uint32_t u_loop_count;
};

struct CIn
{
	uint3 tid : SV_DispatchThreadID;
};

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void main(CIn input)
{
	if (any(input.tid.xy >= u_size))
		return;

	for (uint32_t i = 0; i < u_loop_count; i++)
	{
		float4 value = u_image.Load(input.tid.xy);

		value.r += 1.0;
		value.g += 2.0;
		value.b += 3.0;

		u_image.Store(input.tid.xy, value);
	}
}