#include <boost/scope/scope_exit.hpp>
#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContext.hpp>
#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/Context.hpp>
#include <CyphGPU/ContextSession.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/ComputePassContext.hpp>
#include <CyphGPU/ShaderBundle.hpp>
#include <CyphGPU/ShaderTypes.hpp>
#include <spdlog/spdlog.h>

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
		if (device->getCapabilities() & cgpu::Device::Capability::eCore)
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
	auto clean_device = boost::scope::make_scope_exit([&] { device_session->waitIdle(); });

	// Create sampler
	cgpu::SamplerPtr sampler = cgpu::Sampler::create(
		device_session,
		{
			.wrapping_u = vk::SamplerAddressMode::eClampToEdge,
			.wrapping_v = vk::SamplerAddressMode::eClampToEdge,
		}
	);

	// Create pipeline states
	cgpu::ComputeShaderStatePtr compute_shader_state = cgpu::ComputeShaderState::create(
		device_session,
		{
			.compute_shader = {.source = "shader.slang"},
		}
	);

	cgpu::CommandContext cmd_ctx{device_session};

	{
		cgpu::CommandRecorder cmd_rec = cmd_ctx.createRecorder(device_session->getMainQueue());

		cmd_rec.computePass({
			.callback = [&](cgpu::ComputePassContext& ctx) {
				cgpu::BufferPtr dst_buffer = cgpu::Buffer::create(
					device_session,
					{
						.name = "",
						.size = 4 * sizeof(uint32_t),
						.min_alignment = alignof(uint32_t),
					}
				);

				struct
				{
					uint32_t u_loop_count{};
					uint32_t* u_src_ptr{};
					uint32_t* u_dst_ptr{};
				} parameters{};

				parameters.u_loop_count = 0;
				parameters.u_src_ptr = nullptr;
				parameters.u_dst_ptr = ctx.getBufferDevicePtr<uint32_t>(dst_buffer, cgpu::StorageAccess::eWriteonly);

				ctx.dispatch(compute_shader_state, {1, 1, 1}, parameters);
			},
		});

		cmd_rec.submit();
	}

	cmd_ctx.finish();

	return 0;
}
