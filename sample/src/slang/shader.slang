static const float3 LIGHT_DIR = {0.424, 0.566, 0.707};

cbuffer Parameters
{
	float4x4 u_mvp_matrix;
	float3* u_positions;
	float2* u_tex_coords;
	Texture2D.Handle u_texture;
	SamplerState.Handle u_sampler;
};

struct VIn
{
	uint vertex_index : SV_VulkanVertexID;
};

struct V2F
{
	float4 raster_position : SV_Position;
	float3 position;
	float2 tex_coord;
};

struct FOut
{
	float4 color : SV_Target0;
};

[[shader("vertex")]]
V2F main(VIn input)
{
	V2F output;
	output.raster_position = mul(u_mvp_matrix, float4(u_positions[input.vertex_index], 1.0));
	output.position = output.raster_position.xyz;
	output.tex_coord = u_tex_coords[input.vertex_index];
	return output;
}

[[shader("fragment")]]
FOut main(V2F input)
{
	float3 d_x = ddx(input.position);
	float3 d_y = ddy(input.position);
	float3 normal = normalize(cross(d_x, d_y));
	float light = max(dot(LIGHT_DIR, normal), 0.0);

	FOut output;
	output.color = u_texture.Sample(u_sampler, input.tex_coord) * float4(float3(light), 1.0);
	return output;
}