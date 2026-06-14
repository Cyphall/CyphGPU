#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/Buffer.hpp>
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

int main()
{
	// Create context
	cgpu::ContextPtr context = cgpu::Context::create();

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

	// Create device session
	cgpu::DeviceSessionPtr device_session = cgpu::DeviceSession::create(
		*selected_device,
		{}
	);

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

	cgpu::BufferPtr buffer = cgpu::Buffer::create(
		device_session,
		{
			.name = "Buffer",
			.size = 1024,
			.usages = vk::BufferUsageFlagBits2::eUniformTexelBuffer |
	                  vk::BufferUsageFlagBits2::eStorageTexelBuffer,
		}
	);

	std::ignore = buffer->getUniformTexelDescriptor(vk::Format::eR8G8B8A8Unorm);
	std::ignore = buffer->getStorageTexelDescriptor(vk::Format::eR8G8B8A8Unorm);

	cgpu::VertexInputStatePtr vertex_input_state = cgpu::VertexInputState::create(
		device_session,
		{
			.topology = vk::PrimitiveTopology::eTriangleList,
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

	std::ignore = image->getSampledDescriptor();
	std::ignore = image->getStorageDescriptor();
	std::ignore = image->getAttachmentView(vk::Format::eR8G8B8A8Unorm, 0, {0, 1}, vk::ImageAspectFlagBits::eColor, true, vk::ImageUsageFlagBits::eColorAttachment);

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
			.format = (*selected_device)->getDefaultSurfaceFormat(surface),
			.preferred_extent = extent,
			.usages = vk::ImageUsageFlagBits::eTransferDst,
		}
	);

	// Run render loop
	while (glfwWindowShouldClose(window) == GLFW_FALSE)
	{
		glfwPollEvents();
	}

	return 0;
}
