struct ImDrawVert
{
    float2 pos;
    float2 uv;
    uint32_t col;
};

cbuffer Parameters0
{
    ImDrawVert* u_vertices;
    float2 u_scale;
    float2 u_offset;
};

cbuffer Parameters1
{
	SamplerState.Handle u_sampler;
};

cbuffer Parameters2
{
	Texture2D.Handle u_image;
};

struct VIn
{
    uint32_t vertex_index : SV_VulkanVertexID;
};

struct V2F
{
    float4 raster_position : SV_Position;
    float4 color;
    float2 uv;
};

struct FOut
{
    float4 color : SV_Target0;
};

[[shader("vertex")]]
V2F main(VIn input)
{
	ImDrawVert vertex = u_vertices[input.vertex_index];

	V2F output;
	output.raster_position = float4(vertex.pos * u_scale + u_offset, 0, 1);
	output.color = unpackUnorm4x8ToFloat(vertex.col);
	output.uv = vertex.uv;
	return output;
}

[[shader("fragment")]]
FOut main(V2F input)
{
	FOut output;
	output.color = input.color * u_image.Sample(u_sampler, input.uv);
	return output;
}