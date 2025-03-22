#include <CyphGPU/Context/Context.hpp>
#include <CyphGPU/ContextSession/ContextSession.hpp>

int main()
{
	cgpu::ContextRef context = cgpu::Context::create();

	cgpu::ContextSessionRef contextSession = cgpu::ContextSession::create(
		context,
		{
			.applicationName = "CyphGPU sample",
		}
	);

	return 0;
}