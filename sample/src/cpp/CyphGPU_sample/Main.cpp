#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/CommandContext.hpp>
#include <CyphGPU/ComputePassContext.hpp>
#include <CyphGPU/Context.hpp>
#include <CyphGPU/ContextSession.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/ShaderTypes.hpp>
#include <CyphGPU/Surface.hpp>
#include <CyphGPU/Swapchain.hpp>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

using namespace cgpu::shader_types;

CGPU_DECLARE_SHADER_BUNDLE(shaders)

namespace
{
void recreateSwapchain(
	const cgpu::DeviceSessionPtr& device_session,
	const cgpu::SurfacePtr& surface,
	const glm::uvec2& extent,
	cgpu::SwapchainPtr& swapchain,
	const std::optional<cgpu::SwapchainPtr>& old_swapchain
)
{
	swapchain = cgpu::Swapchain::create(
		device_session,
		surface,
		{
			.format = {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
			.preferred_extent = extent,
			.usages = vk::ImageUsageFlagBits::eTransferDst |
	                  vk::ImageUsageFlagBits::eStorage,
			.old_swapchain = old_swapchain,
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

	// Create pipeline states
	cgpu::ComputeShaderStatePtr compute_shader_state = cgpu::ComputeShaderState::create(
		device_session,
		{
			.compute_shader = {.source = "shader.slang"},
		}
	);

	// Create swapchain
	glm::ivec2 extent;
	glfwGetFramebufferSize(window, &extent.x, &extent.y);

	cgpu::SwapchainPtr swapchain;
	recreateSwapchain(device_session, surface, extent, swapchain, std::nullopt);

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

			recreateSwapchain(device_session, surface, extent, swapchain, swapchain);
		}

		cgpu::ImagePtr swapchain_image = *swapchain->tryGetImage();

		{
			cgpu::CommandRecorder cmd_rec = cmd_ctx.createRecorder(device_session->getMainQueue());

			cmd_rec.clearImage({
				.image = swapchain_image,
				.color_value = glm::vec4{0.0f, 0.0f, 0.0f, 1.0f},
			});

			cmd_rec.barrier({
				.src_stages = vk::PipelineStageFlagBits2::eClear,
				.src_accesses = vk::AccessFlagBits2::eTransferWrite,
				.dst_stages = vk::PipelineStageFlagBits2::eComputeShader,
				.dst_accesses = vk::AccessFlagBits2::eShaderStorageRead |
			                    vk::AccessFlagBits2::eShaderStorageWrite,
			});

			cmd_rec.computePass({
				.callback = [&](cgpu::ComputePassContext& ctx) {
					ctx.bindPipelineStates(compute_shader_state);

					struct
					{
						uint2 u_size{};
						WTexture2D<>::Handle u_image{};
						uint32_t u_loop_count{};
					} parameters{};

					parameters.u_size = swapchain->getExtent();
					parameters.u_image = ctx.getStorageImageDescriptor(swapchain_image, cgpu::CommandRecorder::ResourceAccess::eReadWrite);
					parameters.u_loop_count = 50000;

					ctx.pushParameters(0, parameters);

					ctx.dispatch({cgpu::alignUp(parameters.u_size.get(), glm::uvec2{8u}) / 8u, 1});
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
