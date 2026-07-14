#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContext.hpp>
#include <CyphGPU/Context.hpp>
#include <CyphGPU/ContextSession.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Sampler.hpp>
#include <CyphGPU/Surface.hpp>
#include <CyphGPU/Swapchain.hpp>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

CGPU_DECLARE_SHADER_BUNDLE(shaders)

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
	GLFWwindow* window = glfwCreateWindow(800, 600, "CyphGPU sample", nullptr, nullptr);
	auto destroy_glfw_window = boost::scope::make_scope_exit([&] { glfwDestroyWindow(window); });

	// Create surface
	VkSurfaceKHR surface_raw{};
	glfwCreateWindowSurface(context_session->getHandle(), window, nullptr, &surface_raw);

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

	cgpu::BufferPtr buffer = cgpu::Buffer::create(
		device_session,
		{
			.name = "Buffer",
			.size = 1024,
			.usages = vk::BufferUsageFlagBits2::eUniformTexelBuffer |
	                  vk::BufferUsageFlagBits2::eStorageTexelBuffer,
		}
	);

	std::ignore = buffer->getUniformTexelDescriptorIndirect(vk::Format::eR8G8B8A8Unorm);
	std::ignore = buffer->getStorageTexelDescriptorIndirect(vk::Format::eR8G8B8A8Unorm);

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
		}
	);

	cgpu::FragmentOutputStatePtr fragment_output_state = cgpu::FragmentOutputState::create(
		device_session,
		{
			.color_attachments = {
				{
					.format = vk::Format::eR8G8B8A8Unorm,
				},
			},
		}
	);

	cgpu::ImagePtr image = cgpu::Image::create(
		device_session,
		{
			.name = "Buffer",
			.format = vk::Format::eR8G8B8A8Unorm,
			.extent = {1024, 1024, 1},
			.usages = vk::ImageUsageFlagBits::eSampled |
	                  vk::ImageUsageFlagBits::eStorage |
	                  vk::ImageUsageFlagBits::eColorAttachment,
		}
	);

	std::ignore = image->getSampledDescriptorIndirect();
	std::ignore = image->getStorageDescriptorIndirect();
	std::ignore = image->getAttachmentView(vk::Format::eR8G8B8A8Srgb, 0, {0, 1}, vk::ImageAspectFlagBits::eColor, vk::ImageUsageFlagBits::eColorAttachment);

	cgpu::SamplerPtr sampler = cgpu::Sampler::create(
		device_session,
		{
			.min_filter = vk::Filter::eLinear,
			.mag_filter = vk::Filter::eLinear,
		}
	);

	glm::ivec2 extent;
	glfwGetFramebufferSize(window, &extent.x, &extent.y);

	cgpu::SwapchainPtr swapchain = cgpu::Swapchain::create(
		device_session,
		surface,
		{
			.format = {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
			.preferred_extent = extent,
			.usages = vk::ImageUsageFlagBits::eTransferDst,
		}
	);

	cgpu::CommandContext cmd_ctx{device_session};

	// Run render loop
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

			swapchain = cgpu::Swapchain::create(
				device_session,
				surface,
				{
					.format = {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
					.preferred_extent = extent,
					.usages = vk::ImageUsageFlagBits::eTransferDst,
					.old_swapchain = swapchain,
				}
			);
		}

		{
			cgpu::CommandRecorder cmd_rec = cmd_ctx.createRecorder(device_session->getMainQueue());

			cmd_rec.clearImage({
				.image = *swapchain->tryGetImage(),
				.color_value = glm::vec4{1.0f, 0.0f, 0.0f, 1.0f},
			});

			cmd_rec.submit();
		}

		cmd_ctx.finish();

		swapchain->presentImage();

		FrameMark;
	}

	return 0;
}
