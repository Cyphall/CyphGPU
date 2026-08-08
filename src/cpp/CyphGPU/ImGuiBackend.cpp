#include "ImGuiBackend.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/GraphicsPassContext.hpp>
#include <CyphGPU/Sampler.hpp>

#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <vector>

using namespace cgpu::shader_types;

// NOLINTBEGIN(*-owning-memory)
namespace
{
// NOLINTNEXTLINE(*-identifier-naming)
struct ImGui_ImplCyphGPU_BackendData
{
	cgpu::DeviceSessionPtr device_session{};
	cgpu::SamplerPtr sampler_linear{};
	cgpu::SamplerPtr sampler_nearest{};
	cgpu::VertexInputStatePtr vertex_input_state{};
	cgpu::PreRasterizationShaderStatePtr pre_rasterization_shader_state{};
	cgpu::FragmentShaderStatePtr fragment_shader_state{};
	cgpu::FragmentOutputStatePtr fragment_output_state{};

	// ImTextureID -> cgpu::ImagePtr map
	std::vector<std::pair<cgpu::ImagePtr, cgpu::Image::SampledDescriptorOverrides>> referenced_images{};
};

// NOLINTNEXTLINE(*-identifier-naming)
struct ImGui_ImplCyphGPU_BackendTextureData
{
	cgpu::ImagePtr image{};
	cgpu::Image::SampledDescriptorOverrides overrides{};
};

// NOLINTNEXTLINE(*-identifier-naming)
struct ImGui_ImplCyphGPU_RenderState
{
	cgpu::GraphicsPassContext* ctx{};
	glm::uvec2 render_extent{};
	cgpu::SamplerPtr sampler{};
};

// NOLINTNEXTLINE(*-identifier-naming)
void ImGui_ImplCyphGPU_CreateTexture(ImTextureData& texture)
{
	ImGui_ImplCyphGPU_BackendData& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);

	vk::Format format{};
	vk::ComponentMapping swizzle{};
	switch (texture.Format)
	{
	case ImTextureFormat_RGBA32:
		format = vk::Format::eR8G8B8A8Unorm;
		swizzle = {vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA};
		break;
	case ImTextureFormat_Alpha8:
		format = vk::Format::eR8Unorm;
		swizzle = {vk::ComponentSwizzle::eOne, vk::ComponentSwizzle::eOne, vk::ComponentSwizzle::eOne, vk::ComponentSwizzle::eR};
		break;
	default:
		throw std::runtime_error(std::format("Unknown texture format: {}", magic_enum::enum_name(texture.Format)));
	}

	texture.BackendUserData = new ImGui_ImplCyphGPU_BackendTextureData{};

	ImGui_ImplCyphGPU_BackendTextureData& btd = *static_cast<ImGui_ImplCyphGPU_BackendTextureData*>(texture.BackendUserData);

	btd.image = cgpu::Image::create(
		bd.device_session,
		{
			.name = "ImGui texture",
			.format = format,
			.extent = {texture.Width, texture.Height, 1},
			.usages = vk::ImageUsageFlagBits::eTransferDst |
	                  vk::ImageUsageFlagBits::eSampled,
		}
	);

	btd.overrides.swizzle = swizzle;

	texture.SetTexID(ImGui_ImplCyphGPU_ToTextureID(btd.image, btd.overrides));
}

// NOLINTNEXTLINE(*-identifier-naming)
void ImGui_ImplCyphGPU_UploadTexture(cgpu::CommandRecorder& cmd_rec, ImTextureData& texture, cgpu::Range<glm::uvec2> upload_region)
{
	ImGui_ImplCyphGPU_BackendData& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);
	ImGui_ImplCyphGPU_BackendTextureData& btd = *static_cast<ImGui_ImplCyphGPU_BackendTextureData*>(texture.BackendUserData);

	vk::DeviceSize row_size = static_cast<size_t>(upload_region.size.x) * texture.BytesPerPixel;

	cgpu::BufferPtr staging_buffer = cgpu::Buffer::create(
		bd.device_session,
		{
			.name = "ImGui texture staging buffer",
			.size = upload_region.size.y * row_size,
			.usages = vk::BufferUsageFlagBits2::eTransferSrc,
			.memory_type = cgpu::MemoryType::eCPUUncached,
		}
	);

	std::byte* ptr = staging_buffer->getHostPtr();
	for (uint32_t i = 0; i < upload_region.size.y; i++)
	{
		std::memcpy(
			ptr,
			texture.GetPixelsAt(
				static_cast<int>(upload_region.offset.x),
				static_cast<int>(upload_region.offset.y + i)
			),
			row_size
		);

		ptr += row_size;
	}

	cmd_rec.barrier({
		.src_stages = vk::PipelineStageFlagBits2::eAllCommands,
		.src_accesses = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		.dst_stages = vk::PipelineStageFlagBits2::eCopy,
		.dst_accesses = vk::AccessFlagBits2::eTransferWrite,
	});

	cmd_rec.copyBufferToImage({
		.src_buffer = staging_buffer,
		.dst_image = btd.image,
		.ranges = {{
			{
				.dst = {{
					.pixels = {{{upload_region.offset, 0}, {upload_region.size, 1}}},
				}},
			},
		}},
	});

	cmd_rec.barrier({
		.src_stages = vk::PipelineStageFlagBits2::eCopy,
		.src_accesses = vk::AccessFlagBits2::eTransferWrite,
		.dst_stages = vk::PipelineStageFlagBits2::eAllCommands,
		.dst_accesses = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	});
}

// NOLINTNEXTLINE(*-identifier-naming)
void ImGui_ImplCyphGPU_DestroyTexture(ImTextureData& texture)
{
	delete static_cast<ImGui_ImplCyphGPU_BackendTextureData*>(texture.BackendUserData);
	texture.BackendUserData = nullptr;

	texture.SetTexID(ImTextureID_Invalid);
}

// NOLINTNEXTLINE(*-identifier-naming)
void ImGui_ImplCyphGPU_UpdateTexture(cgpu::CommandRecorder& cmd_rec, ImTextureData& texture)
{
	switch (texture.Status)
	{
	case ImTextureStatus_WantCreate:
		ImGui_ImplCyphGPU_CreateTexture(texture);
		ImGui_ImplCyphGPU_UploadTexture(cmd_rec, texture, {{0, 0}, {texture.Width, texture.Height}});
		texture.SetStatus(ImTextureStatus_OK);
		break;
	case ImTextureStatus_WantUpdates:
		ImGui_ImplCyphGPU_UploadTexture(cmd_rec, texture, {{texture.UpdateRect.x, texture.UpdateRect.y}, {texture.UpdateRect.w, texture.UpdateRect.h}});
		texture.SetStatus(ImTextureStatus_OK);
		break;
	case ImTextureStatus_WantDestroy:
		ImGui_ImplCyphGPU_DestroyTexture(texture);
		texture.SetStatus(ImTextureStatus_Destroyed);
		break;
	default:
		throw std::logic_error(std::format("Unhandled texture state: {}", magic_enum::enum_name(texture.Status)));
	}
}

// NOLINTNEXTLINE(*-identifier-naming)
void ImGui_ImplCyphGPU_DrawCallback_ResetRenderState(const ImDrawList*, const ImDrawCmd*)
{
	auto& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);
	auto& rs = *static_cast<ImGui_ImplCyphGPU_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);

	rs.ctx->bindPipelineStates(
		bd.vertex_input_state,
		bd.pre_rasterization_shader_state,
		bd.fragment_shader_state,
		bd.fragment_output_state
	);

	rs.ctx->setViewport({
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(rs.render_extent.x),
		.height = static_cast<float>(rs.render_extent.y),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	});

	rs.sampler = bd.sampler_linear;
}

// NOLINTNEXTLINE(*-identifier-naming)
void ImGui_ImplCyphGPU_DrawCallback_SetSamplerLinear(const ImDrawList*, const ImDrawCmd*)
{
	auto& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);
	auto& rs = *static_cast<ImGui_ImplCyphGPU_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);

	rs.sampler = bd.sampler_linear;
}

// NOLINTNEXTLINE(*-identifier-naming)
void ImGui_ImplCyphGPU_DrawCallback_SetSamplerNearest(const ImDrawList*, const ImDrawCmd*)
{
	auto& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);
	auto& rs = *static_cast<ImGui_ImplCyphGPU_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);

	rs.sampler = bd.sampler_nearest;
}
}

void ImGui_ImplCyphGPU_Init(const cgpu::DeviceSessionPtr& device_session, vk::Format output_image_format)
{
	ImGuiIO& io = ImGui::GetIO();
	io.BackendRendererName = "CyphGPU ImGui backend";
	io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
	io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
	io.BackendRendererUserData = new ImGui_ImplCyphGPU_BackendData{};

	ImGui_ImplCyphGPU_BackendData& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(io.BackendRendererUserData);

	const auto& vulkan10_props = device_session->getDevice()->getProperties<vk::PhysicalDeviceProperties2>().properties.limits;

	ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
	platform_io.Renderer_TextureMaxWidth = static_cast<int>(vulkan10_props.maxImageDimension2D);
	platform_io.Renderer_TextureMaxHeight = static_cast<int>(vulkan10_props.maxImageDimension2D);
	platform_io.DrawCallback_ResetRenderState = ImGui_ImplCyphGPU_DrawCallback_ResetRenderState;
	platform_io.DrawCallback_SetSamplerLinear = ImGui_ImplCyphGPU_DrawCallback_SetSamplerLinear;
	platform_io.DrawCallback_SetSamplerNearest = ImGui_ImplCyphGPU_DrawCallback_SetSamplerNearest;

	bd.device_session = device_session;
	bd.sampler_linear = cgpu::Sampler::create(
		device_session,
		{
			.min_filter = vk::Filter::eLinear,
			.mag_filter = vk::Filter::eLinear,
		}
	);
	bd.sampler_nearest = cgpu::Sampler::create(
		device_session,
		{
			.min_filter = vk::Filter::eNearest,
			.mag_filter = vk::Filter::eNearest,
		}
	);
	bd.vertex_input_state = cgpu::VertexInputState::create(
		device_session,
		{
			.topology = vk::PrimitiveTopology::eTriangleList,
		}
	);
	bd.pre_rasterization_shader_state = cgpu::PreRasterizationShaderState::create(
		device_session,
		{
			.vertex_shader = {.source = "CyphGPU/ImGui.slang"},
			.cull_mode = vk::CullModeFlagBits::eNone,
		}
	);
	bd.fragment_shader_state = cgpu::FragmentShaderState::create(
		device_session,
		{
			.fragment_shader = {{.source = "CyphGPU/ImGui.slang"}},
		}
	);
	bd.fragment_output_state = cgpu::FragmentOutputState::create(
		device_session,
		{
			.color_attachments = {
				{
					.format = cgpu::getLinearEquivalent(output_image_format),
					.blend = {{
						.color = {
							.src_factor = vk::BlendFactor::eSrcAlpha,
							.dst_factor = vk::BlendFactor::eOneMinusSrcAlpha,
							.op = vk::BlendOp::eAdd,
						},
						.alpha = {
							.src_factor = vk::BlendFactor::eOne,
							.dst_factor = vk::BlendFactor::eOneMinusSrcAlpha,
							.op = vk::BlendOp::eAdd,
						},
					}},
				},
			},
		}
	);
}

void ImGui_ImplCyphGPU_NewFrame()
{
	ImGui_ImplCyphGPU_BackendData& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);

	bd.referenced_images.clear();

	for (ImTextureData* texture : ImGui::GetPlatformIO().Textures)
	{
		if (texture->BackendUserData == nullptr)
		{
			continue;
		}

		ImGui_ImplCyphGPU_BackendTextureData& btd = *static_cast<ImGui_ImplCyphGPU_BackendTextureData*>(texture->BackendUserData);
		texture->SetTexID(ImGui_ImplCyphGPU_ToTextureID(btd.image, btd.overrides));
	}
}

void ImGui_ImplCyphGPU_RenderDrawData(const ImDrawData& draw_data, cgpu::CommandRecorder& cmd_rec, const cgpu::ImagePtr& output_image)
{
	ImGui_ImplCyphGPU_BackendData& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);

	if (draw_data.Textures != nullptr)
	{
		for (ImTextureData* texture : *draw_data.Textures)
		{
			if (texture->Status != ImTextureStatus_OK)
			{
				ImGui_ImplCyphGPU_UpdateTexture(cmd_rec, *texture);
			}
		}
	}

	glm::uvec2 render_extent = output_image->getDesc().extent;

	std::optional<cgpu::BufferPtr> vertex_buffer;
	if (draw_data.TotalVtxCount > 0)
	{
		vertex_buffer = cgpu::Buffer::create(
			bd.device_session,
			{
				.name = "ImGui vertex buffer",
				.size = static_cast<vk::DeviceSize>(draw_data.TotalVtxCount) * sizeof(ImDrawVert),
				.memory_type = cgpu::MemoryType::eCPUVisibleGPU,
			}
		);

		ImDrawVert* vertex_ptr = (*vertex_buffer)->getHostPtr<ImDrawVert>();
		for (int i = 0; i < draw_data.CmdLists.Size; i++)
		{
			const ImDrawList& cmd_list = *draw_data.CmdLists[i];

			std::copy_n(cmd_list.VtxBuffer.Data, cmd_list.VtxBuffer.Size, vertex_ptr);

			vertex_ptr += cmd_list.VtxBuffer.Size;
		}
	}

	std::optional<cgpu::BufferPtr> index_buffer;
	if (draw_data.TotalIdxCount > 0)
	{
		index_buffer = cgpu::Buffer::create(
			bd.device_session,
			{
				.name = "ImGui index buffer",
				.size = static_cast<vk::DeviceSize>(draw_data.TotalIdxCount) * sizeof(ImDrawIdx),
				.usages = vk::BufferUsageFlagBits2::eIndexBuffer,
				.memory_type = cgpu::MemoryType::eCPUVisibleGPU,
			}
		);

		ImDrawIdx* index_ptr = (*index_buffer)->getHostPtr<ImDrawIdx>();
		for (int i = 0; i < draw_data.CmdLists.Size; i++)
		{
			const ImDrawList& cmd_list = *draw_data.CmdLists[i];

			std::copy_n(cmd_list.IdxBuffer.Data, cmd_list.IdxBuffer.Size, index_ptr);

			index_ptr += cmd_list.IdxBuffer.Size;
		}
	}

	cmd_rec.barrier({
		.src_stages = vk::PipelineStageFlagBits2::eAllCommands,
		.src_accesses = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		.dst_stages = vk::PipelineStageFlagBits2::eAllCommands,
		.dst_accesses = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	});

	cmd_rec.graphicsPass({
		.color_attachments = {{{
			.image = output_image,
			.format = cgpu::getLinearEquivalent(output_image->getDesc().format),
			.load_op = vk::AttachmentLoadOp::eLoad,
			.store_op = vk::AttachmentStoreOp::eStore,
			.clear_color_value = glm::vec4{0.0f, 0.0f, 0.0f, 1.0f},
		}}},
		.callback = [&](cgpu::GraphicsPassContext& ctx) {
			ImGui_ImplCyphGPU_RenderState render_state{
				.ctx = &ctx,
				.render_extent = render_extent,
				.sampler = bd.sampler_linear,
			};

			ImGui::GetPlatformIO().Renderer_RenderState = &render_state;

			ImGui_ImplCyphGPU_DrawCallback_ResetRenderState(nullptr, nullptr);

			glm::vec2 clip_offset = std::bit_cast<glm::vec2>(draw_data.DisplayPos);
			glm::vec2 clip_scale = std::bit_cast<glm::vec2>(draw_data.FramebufferScale);

			int global_vertex_offset = 0;
			int global_index_offset = 0;
			for (const ImDrawList* cmd_list : draw_data.CmdLists)
			{
				for (const ImDrawCmd& cmd : cmd_list->CmdBuffer)
				{
					if (cmd.UserCallback != nullptr)
					{
						cmd.UserCallback(cmd_list, &cmd);
					}
					else
					{
						glm::vec2 clip_min = {(cmd.ClipRect.x - clip_offset.x) * clip_scale.x, (cmd.ClipRect.y - clip_offset.y) * clip_scale.y};
						glm::vec2 clip_max = {(cmd.ClipRect.z - clip_offset.x) * clip_scale.x, (cmd.ClipRect.w - clip_offset.y) * clip_scale.y};

						clip_min = glm::max(clip_min, {0.0f, 0.0f});
						clip_max = glm::min(clip_max, glm::vec2{render_extent});
						if (glm::any(glm::lessThanEqual(clip_max, clip_min)))
						{
							continue;
						}

						ctx.setScissor({
							.offset = std::bit_cast<vk::Offset2D>(static_cast<glm::ivec2>(clip_min)),
							.extent = std::bit_cast<vk::Extent2D>(static_cast<glm::uvec2>(clip_max - clip_min)),
						});

						ctx.bindIndexBuffer(*index_buffer, sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

						struct
						{
							ImDrawVert* vertices{};
							float2 scale{};
							float2 offset{};
							Texture2D<>::Handle image{};
							SamplerState::Handle sampler{};
						} parameters{};

						parameters.vertices = ctx.getBufferDevicePtr<ImDrawVert>(*vertex_buffer, cgpu::CommandRecorder::ResourceAccess::eReadonly);

						parameters.scale = glm::vec2{
							2.0f / draw_data.DisplaySize.x,
							2.0f / draw_data.DisplaySize.y,
						};

						parameters.offset = glm::vec2{
							-1.0f - draw_data.DisplayPos.x * (2.0f / draw_data.DisplaySize.x),
							-1.0f - draw_data.DisplayPos.y * (2.0f / draw_data.DisplaySize.y),
						};

						auto [image, overrides] = bd.referenced_images[cmd.GetTexID() - 1];
						overrides.format = cgpu::getLinearEquivalent(overrides.format ? *overrides.format : image->getDesc().format);
						parameters.image = ctx.getSampledImageDescriptor(image, overrides);

						parameters.sampler = render_state.sampler->getDescriptor();

						ctx.drawIndexed(
							cmd.ElemCount,
							1,
							cmd.IdxOffset + global_index_offset,
							static_cast<int>(cmd.VtxOffset + global_vertex_offset),
							0,
							parameters
						);
					}
				}
				global_index_offset += cmd_list->IdxBuffer.Size;
				global_vertex_offset += cmd_list->VtxBuffer.Size;
			}

			ImGui::GetPlatformIO().Renderer_RenderState = nullptr;
		},
	});

	cmd_rec.barrier({
		.src_stages = vk::PipelineStageFlagBits2::eAllCommands,
		.src_accesses = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		.dst_stages = vk::PipelineStageFlagBits2::eAllCommands,
		.dst_accesses = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	});
}

void ImGui_ImplCyphGPU_Shutdown()
{
	for (ImTextureData* texture : ImGui::GetPlatformIO().Textures)
	{
		assert(texture->RefCount == 1);

		ImGui_ImplCyphGPU_DestroyTexture(*texture);
		texture->SetStatus(ImTextureStatus_Destroyed);
	}

	ImGuiIO& io = ImGui::GetIO();
	io.BackendRendererName = nullptr;
	io.BackendFlags &= ~ImGuiBackendFlags_RendererHasVtxOffset;
	io.BackendFlags &= ~ImGuiBackendFlags_RendererHasTextures;
	delete static_cast<ImGui_ImplCyphGPU_BackendData*>(io.BackendRendererUserData);
	io.BackendRendererUserData = nullptr;

	ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
	platform_io.Renderer_TextureMaxWidth = 0;
	platform_io.Renderer_TextureMaxHeight = 0;
	platform_io.DrawCallback_ResetRenderState = nullptr;
	platform_io.DrawCallback_SetSamplerLinear = nullptr;
	platform_io.DrawCallback_SetSamplerNearest = nullptr;
}

ImTextureID ImGui_ImplCyphGPU_ToTextureID(const cgpu::ImagePtr& image, const cgpu::Image::SampledDescriptorOverrides& overrides)
{
	ImGui_ImplCyphGPU_BackendData& bd = *static_cast<ImGui_ImplCyphGPU_BackendData*>(ImGui::GetIO().BackendRendererUserData);

	bd.referenced_images.emplace_back(image, overrides);

	return bd.referenced_images.size();
}

// NOLINTEND(*-owning-memory)
