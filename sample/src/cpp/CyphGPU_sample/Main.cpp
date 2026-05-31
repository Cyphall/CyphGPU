#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/Context/Context.hpp>
#include <CyphGPU/ContextSession/ContextSession.hpp>
#include <CyphGPU/Surface/Surface.hpp>
#include <GLFW/glfw3.h>

int main()
{
	// Create context
	cgpu::ContextRef context = cgpu::Context::create();

	// Create context session
	cgpu::ContextSessionRef contextSession = cgpu::ContextSession::create(
		context,
		{
			.applicationName = "CyphGPU sample",
		}
	);

	// Create GLFW window
	glfwInitVulkanLoader(contextSession->getDispatcher().vkGetInstanceProcAddr);
	glfwInit();
	auto terminateGLFW = boost::scope::make_scope_exit(
		[&]
		{
			glfwTerminate();
		}
	);

	glfwDefaultWindowHints();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
	GLFWwindow* window = glfwCreateWindow(800, 600, "CyphGPU sample", nullptr, nullptr);
	auto destroyGLFWWindow = boost::scope::make_scope_exit(
		[&]
		{
			glfwDestroyWindow(window);
		}
	);

	// Create surface
	VkSurfaceKHR surfaceRaw;
	glfwCreateWindowSurface(contextSession->getHandle(), window, nullptr, &surfaceRaw);

	cgpu::SurfaceRef surface = cgpu::Surface::create(
		contextSession,
		{
			.surface = surfaceRaw,
		}
	);

	// Run render loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}

	return 0;
}