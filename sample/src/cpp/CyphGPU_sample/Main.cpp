#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/CommandContext.hpp>
#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/Context.hpp>
#include <CyphGPU/ContextSession.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/GraphicsPassContext.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Surface.hpp>
#include <CyphGPU/Swapchain.hpp>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace
{
void recreateSwapchain(
	const cgpu::DeviceSessionPtr& device_session,
	const cgpu::SurfacePtr& surface,
	const vk::SurfaceFormatKHR& format,
	const glm::uvec2& extent,
	cgpu::SwapchainPtr& swapchain,
	cgpu::ImagePtr& multisampled_image,
	cgpu::ImagePtr& singlesampled_image,
	const std::optional<cgpu::SwapchainPtr>& old_swapchain
)
{
	swapchain = cgpu::Swapchain::create(
		device_session,
		surface,
		{
			.format = format,
			.preferred_extent = extent,
			.usages = vk::ImageUsageFlagBits::eColorAttachment |
	                  vk::ImageUsageFlagBits::eTransferSrc |
	                  vk::ImageUsageFlagBits::eTransferDst,
			.old_swapchain = old_swapchain,
		}
	);

	multisampled_image = cgpu::Image::create(
		device_session,
		{
			.name = "Multisampled image",
			.format = format.format,
			.extent = {extent, 1},
			.usages = vk::ImageUsageFlagBits::eColorAttachment |
	                  vk::ImageUsageFlagBits::eTransferSrc |
	                  vk::ImageUsageFlagBits::eTransferDst,
			.samples = vk::SampleCountFlagBits::e4,
		}
	);

	singlesampled_image = cgpu::Image::create(
		device_session,
		{
			.name = "Singlesampled image",
			.format = format.format,
			.extent = {extent, 1},
			.usages = vk::ImageUsageFlagBits::eColorAttachment |
	                  vk::ImageUsageFlagBits::eTransferSrc |
	                  vk::ImageUsageFlagBits::eTransferDst,
		}
	);
}
}

int main()
{
	// Create context
	cgpu::ContextPtr context = cgpu::Context::create({});

	if (!(context->getCapabilities() & cgpu::Context::Capability::eCore) ||
	    !(context->getCapabilities() & cgpu::Context::Capability::eSurfaceWin32))
	{
		spdlog::error("Context is not compatible.");
		return 1;
	}

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

	std::optional<vk::SurfaceFormatKHR> surface_format = selected_device.value()->selectBestSurfaceFormat(
		surface,
		{{
			{vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
		}}
	);

	if (!surface_format)
	{
		spdlog::error("Could not find a suitable surface format.");
		return 1;
	}

	// Create device session
	cgpu::DeviceSessionPtr device_session = cgpu::DeviceSession::create(
		*selected_device,
		{}
	);
	auto clean_device = boost::scope::make_scope_exit([&] { device_session->waitIdle(); });

	// Create swapchain
	glm::ivec2 extent;
	glfwGetFramebufferSize(window, &extent.x, &extent.y);

	cgpu::SwapchainPtr swapchain;
	cgpu::ImagePtr multisampled_image;
	cgpu::ImagePtr singlesampled_image;
	recreateSwapchain(device_session, surface, *surface_format, extent, swapchain, multisampled_image, singlesampled_image, std::nullopt);

	// Run render loop
	cgpu::CommandContext cmd_ctx{device_session};
	while (glfwWindowShouldClose(window) == GLFW_FALSE)
	{
		glfwPollEvents();

		std::optional<cgpu::ImagePtr> swapchain_image;
		while (!(swapchain_image = swapchain->tryGetImage()))
		{
			glfwGetFramebufferSize(window, &extent.x, &extent.y);
			if (extent.x == 0 || extent.y == 0)
			{
				glfwWaitEvents();
				continue;
			}

			recreateSwapchain(device_session, surface, *surface_format, extent, swapchain, multisampled_image, singlesampled_image, swapchain);
		}

		{
			cgpu::CommandRecorder cmd_rec = cmd_ctx.createRecorder(device_session->getMainQueue());

#define REPRO_MODE 0

#if REPRO_MODE == 0
			// 1) Graphics pass clear + resolve into singlesampled_image
			// 2) Copy singlesampled_image to swapchain_image
			// Broken

			cmd_rec.graphicsPass({
				.color_attachments = {{
					{
						.image = multisampled_image,
						.load_op = vk::AttachmentLoadOp::eClear,
						.store_op = vk::AttachmentStoreOp::eDontCare,
						.clear_color_value = glm::vec4{1, 0, 1, 1},
						.resolve = {{
							.image = singlesampled_image,
						}},
					},
				}},
				.callback = [&](cgpu::GraphicsPassContext&) {},
			});

			cmd_rec.copyImageToImage({
				.src_image = singlesampled_image,
				.dst_image = *swapchain_image,
			});
#elif REPRO_MODE == 1
			// 1) Standalone clear multisampled_image
			// 2) Standalone resolve multisampled_image into singlesampled_image
			// 3) Copy singlesampled_image to swapchain_image
			// Works

			cmd_rec.graphicsPass({
				.color_attachments = {{
					{
						.image = multisampled_image,
						.load_op = vk::AttachmentLoadOp::eClear,
						.store_op = vk::AttachmentStoreOp::eStore,
						.clear_color_value = glm::vec4{1, 0, 1, 1},
					},
				}},
				.callback = [&](cgpu::GraphicsPassContext&) {},
			});

			cmd_rec.resolve({
				.src_image = multisampled_image,
				.dst_image = singlesampled_image,
			});

			cmd_rec.copyImageToImage({
				.src_image = singlesampled_image,
				.dst_image = *swapchain_image,
			});
#elif REPRO_MODE == 2
			// 1) Graphics pass clear + resolve into swapchain_image
			// Works

			cmd_rec.graphicsPass({
				.color_attachments = {{
					{
						.image = multisampled_image,
						.load_op = vk::AttachmentLoadOp::eClear,
						.store_op = vk::AttachmentStoreOp::eDontCare,
						.clear_color_value = glm::vec4{1, 0, 1, 1},
						.resolve = {{
							.image = *swapchain_image,
						}},
					},
				}},
				.callback = [&](cgpu::GraphicsPassContext&) {},
			});
#elif REPRO_MODE == 3
			// 1) Standalone clear singlesampled_image
			// 2) Copy singlesampled_image to swapchain_image
			// Works

			cmd_rec.clearImage({
				.image = singlesampled_image,
				.color_value = glm::vec4{1, 0, 1, 1},
			});

			cmd_rec.copyImageToImage({
				.src_image = singlesampled_image,
				.dst_image = *swapchain_image,
			});
#endif

			cmd_rec.submit();
		}

		cmd_ctx.finish();

		swapchain->presentImage();
	}

	return 0;
}
