#include <CyphGPU_sample/mesh.h>
#include <CyphGPU_sample/texture.h>

#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContext.hpp>
#include <CyphGPU/Context.hpp>
#include <CyphGPU/ContextSession.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/GraphicsPassContext.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Sampler.hpp>
#include <CyphGPU/ShaderTypes.hpp>
#include <CyphGPU/Surface.hpp>
#include <CyphGPU/Swapchain.hpp>
#include <GLFW/glfw3.h>
#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace cgpu::shader_types;

CGPU_DECLARE_SHADER_BUNDLE(shaders)

namespace
{
void uploadVertexBuffers(
	const cgpu::DeviceSessionPtr& device_session,
	cgpu::BufferPtr& position_buffer,
	cgpu::BufferPtr& tex_coord_buffer
)
{
	position_buffer = cgpu::Buffer::create(
		device_session,
		{
			.name = "Position buffer",
			.size = POSITION_DATA.size() * sizeof(glm::vec3),
			.memory_type = cgpu::MemoryType::eCPUVisibleGPU,
			.min_alignment = alignof(glm::vec3),
		}
	);

	tex_coord_buffer = cgpu::Buffer::create(
		device_session,
		{
			.name = "Tex coord buffer",
			.size = TEX_COORD_DATA.size() * sizeof(glm::vec2),
			.memory_type = cgpu::MemoryType::eCPUVisibleGPU,
			.min_alignment = alignof(glm::vec2),
		}
	);

	std::memcpy(position_buffer->getHostPtr(), POSITION_DATA.data(), position_buffer->getDesc().size);
	std::memcpy(tex_coord_buffer->getHostPtr(), TEX_COORD_DATA.data(), tex_coord_buffer->getDesc().size);
}

void uploadTexture(
	const cgpu::DeviceSessionPtr& device_session,
	cgpu::CommandRecorder& rec,
	cgpu::ImagePtr& texture
)
{
	int width{};
	int height{};
	stbi_uc* data = stbi_load_from_memory(TEXTURE_DATA.data(), TEXTURE_DATA.size(), &width, &height, nullptr, STBI_rgb_alpha);

	texture = cgpu::Image::create(
		device_session,
		{
			.name = "Texture",
			.format = vk::Format::eR8G8B8A8Srgb,
			.extent = {width, height, 1},
			.usages = vk::ImageUsageFlagBits::eTransferDst |
	                  vk::ImageUsageFlagBits::eSampled,
		}
	);

	cgpu::BufferPtr staging_buffer = cgpu::Buffer::create(
		device_session,
		{
			.name = "Texture staging buffer",
			.size = cgpu::calcImageByteSize(texture->getDesc().format, texture->getDesc().extent, 1),
			.usages = vk::BufferUsageFlagBits2::eTransferSrc,
			.memory_type = cgpu::MemoryType::eCPUUncached,
		}
	);

	std::memcpy(staging_buffer->getHostPtr(), data, staging_buffer->getDesc().size);

	stbi_image_free(data);

	rec.copyBufferToImage({
		.src_buffer = staging_buffer,
		.dst_image = texture,
	});
}

void recreateSwapchain(
	const cgpu::DeviceSessionPtr& device_session,
	const cgpu::SurfacePtr& surface,
	const glm::uvec2& extent,
	cgpu::SwapchainPtr& swapchain,
	cgpu::ImagePtr& depth_image,
	const std::optional<cgpu::SwapchainPtr>& old_swapchain
)
{
	swapchain = cgpu::Swapchain::create(
		device_session,
		surface,
		{
			.format = {vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear},
			.preferred_extent = extent,
			.usages = vk::ImageUsageFlagBits::eColorAttachment,
			.old_swapchain = old_swapchain,
		}
	);

	depth_image = cgpu::Image::create(
		device_session,
		{
			.name = "Depth image",
			.format = vk::Format::eD32Sfloat,
			.extent = {extent, 1},
			.usages = vk::ImageUsageFlagBits::eDepthStencilAttachment,
		}
	);
}
}

int main()
{
	// Create context
	cgpu::ContextPtr context = cgpu::Context::create({
		.shader_bundles = {&shaders},
	});

	// Create context session
	cgpu::ContextSessionPtr context_session = cgpu::ContextSession::create(
		context,
		{
			.application_name = "CyphGPU sample",
		}
	);

	// Select device
	std::optional<cgpu::DevicePtr> selected_device;
	for (const cgpu::DevicePtr& device : context_session->getDevices())
	{
		if (device->getCapabilities() & cgpu::Device::Capability::eCore &&
		    device->getCapabilities() & cgpu::Device::Capability::eSwapchain)
		{
			selected_device = device;
			break;
		}
	}

	if (!selected_device)
	{
		spdlog::error("Could not find a compatible device.");
		return 1;
	}

	// Create GLFW window
	glfwInitVulkanLoader(context_session->getDispatcher().vkGetInstanceProcAddr);
	glfwInit();
	auto terminate_glfw = boost::scope::make_scope_exit([&] { glfwTerminate(); });

	glfwDefaultWindowHints();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
	GLFWwindow* window = glfwCreateWindow(500, 500, "CyphGPU sample", nullptr, nullptr);
	auto destroy_glfw_window = boost::scope::make_scope_exit([&] { glfwDestroyWindow(window); });

	// Create surface
	VkSurfaceKHR surface_raw{};
	vk::detail::resultCheck(
		static_cast<vk::Result>(
			glfwCreateWindowSurface(context_session->getHandle(), window, nullptr, &surface_raw)
		),
		"glfwCreateWindowSurface"
	);

	cgpu::SurfacePtr surface = cgpu::Surface::create(
		context_session,
		{
			.surface = surface_raw,
		}
	);

	// Create device session
	cgpu::DeviceSessionPtr device_session = cgpu::DeviceSession::create(
		*selected_device,
		{}
	);
	auto clean_device = boost::scope::make_scope_exit([&] { device_session->waitIdle(); });

	// Upload vertex buffers
	cgpu::BufferPtr position_buffer;
	cgpu::BufferPtr tex_coord_buffer;
	uploadVertexBuffers(device_session, position_buffer, tex_coord_buffer);

	// Create sampler
	cgpu::SamplerPtr sampler = cgpu::Sampler::create(
		device_session,
		{
			.wrapping_u = vk::SamplerAddressMode::eClampToEdge,
			.wrapping_v = vk::SamplerAddressMode::eClampToEdge,
		}
	);

	// Create pipeline states
	cgpu::VertexInputStatePtr vertex_input_state = cgpu::VertexInputState::create(
		device_session,
		{
			.topology = vk::PrimitiveTopology::eTriangleList,
		}
	);

	cgpu::PreRasterizationShaderStatePtr pre_rasterization_shader_state = cgpu::PreRasterizationShaderState::create(
		device_session,
		{
			.vertex_shader = {.source = "shader.slang"},
		}
	);

	cgpu::FragmentShaderStatePtr fragment_shader_state = cgpu::FragmentShaderState::create(
		device_session,
		{
			.fragment_shader = {{.source = "shader.slang"}},
			.depth_state = {{
				.test_pass_condition = vk::CompareOp::eLess,
				.write_enabled = true,
			}},
		}
	);

	cgpu::FragmentOutputStatePtr fragment_output_state = cgpu::FragmentOutputState::create(
		device_session,
		{
			.color_attachments = {
				{
					.format = vk::Format::eB8G8R8A8Srgb,
				},
			},
			.depth_stencil_attachment = {{
				.format = vk::Format::eD32Sfloat,
			}},
		}
	);

	// Create swapchain
	glm::ivec2 extent;
	glfwGetFramebufferSize(window, &extent.x, &extent.y);

	cgpu::SwapchainPtr swapchain;
	cgpu::ImagePtr depth_image;
	recreateSwapchain(device_session, surface, extent, swapchain, depth_image, std::nullopt);

	cgpu::CommandContext cmd_ctx{device_session};

	// Run render loop
	float rotation = 0.0f;
	std::optional<cgpu::ImagePtr> texture;
	while (glfwWindowShouldClose(window) == GLFW_FALSE)
	{
		glfwPollEvents();

		while (!swapchain->tryGetImage())
		{
			glfwGetFramebufferSize(window, &extent.x, &extent.y);
			if (extent.x == 0 || extent.y == 0)
			{
				glfwWaitEvents();
				continue;
			}

			recreateSwapchain(device_session, surface, extent, swapchain, depth_image, swapchain);
		}

		rotation += 4.0f;

		{
			cgpu::CommandRecorder cmd_rec = cmd_ctx.createRecorder(device_session->getMainQueue());

			if (!texture)
			{
				uploadTexture(device_session, cmd_rec, texture.emplace());

				cmd_rec.barrier({
					.src_stages = vk::PipelineStageFlagBits2::eCopy,
					.src_accesses = vk::AccessFlagBits2::eTransferWrite,
					.dst_stages = vk::PipelineStageFlagBits2::eFragmentShader,
					.dst_accesses = vk::AccessFlagBits2::eShaderSampledRead,
				});
			}

			cmd_rec.graphicsPass({
				.color_attachments = {{{
					.image = *swapchain->tryGetImage(),
					.load_op = vk::AttachmentLoadOp::eClear,
					.store_op = vk::AttachmentStoreOp::eStore,
					.clear_color_value = glm::vec4{0.033f, 0.033f, 0.033f, 1.0f},
				}}},
				.depth_stencil_attachment = {{
					.image = depth_image,
					.load_op = vk::AttachmentLoadOp::eClear,
					.store_op = vk::AttachmentStoreOp::eDontCare,
					.clear_depth_value = 1.0f,
				}},
				.callback = [&](cgpu::GraphicsPassContext& ctx) {
					ctx.bindPipelineStates(
						vertex_input_state,
						pre_rasterization_shader_state,
						fragment_shader_state,
						fragment_output_state
					);

					struct
					{
						float4x4 u_mvp_matrix{};
						float3* u_positions{};
						float2* u_tex_coords{};
						Texture2D<>::Handle u_texture{};
						SamplerState::Handle u_sampler{};
					} parameters{};

					glm::mat4 p_matrix = glm::perspective(
						glm::radians(45.0f),
						static_cast<float>(swapchain->getExtent().x) / static_cast<float>(swapchain->getExtent().y),
						0.1f,
						100.0f
					);
					p_matrix[1][1] *= -1.0f;

					glm::mat4 v_matrix = glm::lookAt(
						glm::vec3{0.0f, 3.0f, 5.0f},
						glm::vec3{0.0f, 0.0f, 0.0f},
						glm::vec3{0.0f, 1.0f, 0.0f}
					);

					glm::mat4 m_matrix = glm::rotate(
						glm::radians(rotation),
						glm::vec3{0.0f, 1.0f, 0.0f}
					);

					parameters.u_mvp_matrix = p_matrix * v_matrix * m_matrix;
					parameters.u_positions = ctx.getBufferDevicePtr<float3>(position_buffer, cgpu::CommandRecorder::ResourceAccess::eReadonly);
					parameters.u_tex_coords = ctx.getBufferDevicePtr<float2>(tex_coord_buffer, cgpu::CommandRecorder::ResourceAccess::eReadonly);
					parameters.u_texture = ctx.getSampledImageDescriptor(*texture);
					parameters.u_sampler = sampler->getDescriptor();

					ctx.pushParameters(0, parameters);

					ctx.draw(12 * 3, 1, 0, 0);
				},
			});

			cmd_rec.submit();
		}

		cmd_ctx.finish();

		swapchain->presentImage();

		FrameMark;
	}

	return 0;
}
