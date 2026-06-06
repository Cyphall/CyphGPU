#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/Context/Context.hpp>
#include <CyphGPU/ContextSession/ContextSession.hpp>
#include <CyphGPU/Device/Device.hpp>
#include <CyphGPU/Surface/Surface.hpp>
#include <GLFW/glfw3.h>

int main()
{
	// Create context
	cgpu::ContextRef context = cgpu::Context::create();

	// Create context session
	cgpu::ContextSessionRef context_session = cgpu::ContextSession::create(
		context,
		{
			.application_name = "CyphGPU sample",
		}
	);

	// Select device
	std::optional<cgpu::DeviceRef> selected_device;
	for (const cgpu::DeviceRef& device : context_session->getDevices())
	{
		if (device->getCapabilities() & cgpu::Device::Capability::eCore &&
		    device->getCapabilities() & cgpu::Device::Capability::eSwapchain)
		{
			selected_device = device;
			break;
		}
	}

	// Create GLFW window
	glfwInitVulkanLoader(context_session->getDispatcher().vkGetInstanceProcAddr);
	glfwInit();
	auto terminate_glfw = boost::scope::make_scope_exit(
		[&]
		{
			glfwTerminate();
		}
	);

	glfwDefaultWindowHints();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
	GLFWwindow* window = glfwCreateWindow(800, 600, "CyphGPU sample", nullptr, nullptr);
	auto destroy_glfw_window = boost::scope::make_scope_exit(
		[&]
		{
			glfwDestroyWindow(window);
		}
	);

	// Create surface
	VkSurfaceKHR surface_raw{};
	glfwCreateWindowSurface(context_session->getHandle(), window, nullptr, &surface_raw);

	cgpu::SurfaceRef surface = cgpu::Surface::create(
		context_session,
		{
			.surface = surface_raw,
		}
	);

	// Run render loop
	while (glfwWindowShouldClose(window) == GLFW_FALSE)
	{
		glfwPollEvents();
	}

	return 0;
}
