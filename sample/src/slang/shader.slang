cbuffer GlobalParameters
{
	float4x4 u_viewProjectionMatrix;
};

cbuffer DrawParameters
{
	float4x4 u_modelMatrix;
	float3* u_positions;
};

struct VIn
{
	uint vertexIndex : SV_VulkanVertexID;
};

struct V2F
{
	float4 rasterPosition : SV_Position;
};

struct FOut
{
	float4 color : SV_Target0;
};

[[shader("vertex")]]
V2F main(VIn input)
{
	V2F output;
	output.rasterPosition = mul(mul(float4(u_positions[input.vertexIndex], 1.0), u_modelMatrix), u_viewProjectionMatrix);
	return output;
}

[[shader("fragment")]]
FOut main(V2F input)
{
	FOut output;
	output.color = {1.0, 0.0, 0.0, 1.0};
	return output;
}