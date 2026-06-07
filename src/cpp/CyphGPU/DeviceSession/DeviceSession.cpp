#include "DeviceSession.hpp"

#include <CyphGPU/Device/Device.hpp>
#include <CyphGPU/Queue/Queue.hpp>
#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <unordered_set>

namespace
{
struct QueueFamilyInfo
{
	uint32_t available_queue_count{};
	vk::QueueFlags caps{};
	uint32_t num_caps{};
};

std::vector<QueueFamilyInfo> queryAvailableQueues(const cgpu::DevicePtr& device)
{
	std::vector<vk::QueueFamilyProperties> queue_family_properties = device->getHandle().getQueueFamilyProperties(device->getContextSession()->getDispatcher());

	std::vector<QueueFamilyInfo> infos;
	infos.reserve(queue_family_properties.size());
	for (const vk::QueueFamilyProperties& properties : device->getHandle().getQueueFamilyProperties(device->getContextSession()->getDispatcher()))
	{
		QueueFamilyInfo& info = infos.emplace_back();
		info.available_queue_count = properties.queueCount;
		info.caps = properties.queueFlags;
		info.num_caps = static_cast<vk::QueueFlags::MaskType>(properties.queueFlags);
	}

	return infos;
}

std::optional<uint32_t> tryReserveBestQueue(std::span<QueueFamilyInfo> available_queues, vk::QueueFlags required_caps)
{
	std::optional<uint32_t> best_queue_family;
	for (uint32_t i = 0; i < available_queues.size(); i++)
	{
		// Not compatible
		if ((available_queues[i].caps & required_caps) != required_caps)
		{
			continue;
		}

		// No queue available for this family
		if (available_queues[i].available_queue_count == 0)
		{
			continue;
		}

		// Current candidate is better
		if (best_queue_family && available_queues[i].num_caps > available_queues[*best_queue_family].num_caps)
		{
			continue;
		}

		best_queue_family = i;
	}

	if (best_queue_family)
	{
		available_queues[*best_queue_family].available_queue_count--;
	}

	return best_queue_family;
}
}

cgpu::DeviceSessionPtr cgpu::DeviceSession::create(const DevicePtr& device, Desc&& desc)
{
	return std::make_shared<cgpu::DeviceSession>(PrivateKey{}, device, std::move(desc));
}

cgpu::DeviceSession::DeviceSession(PrivateKey, const DevicePtr& device, Desc&& desc):
	m_device{device},
	m_desc{std::move(desc)},
	m_dispatcher{device->getContextSession()->getDispatcher()}
{
	createDevice();
}

cgpu::DeviceSession::~DeviceSession()
{
	m_async_transfer_queue.reset();
	m_async_compute_queue.reset();
	m_async_graphics_queue.reset();
	m_main_queue.reset();
	m_handle.destroy(nullptr, m_dispatcher);
}

const cgpu::DevicePtr& cgpu::DeviceSession::getDevice() const
{
	return m_device;
}

const cgpu::DeviceSession::Desc& cgpu::DeviceSession::getDesc() const
{
	return m_desc;
}

const vk::detail::DispatchLoaderDynamic& cgpu::DeviceSession::getDispatcher() const
{
	return m_dispatcher;
}

const vk::Device& cgpu::DeviceSession::getHandle() const
{
	return m_handle;
}

cgpu::QueuePtr cgpu::DeviceSession::getMainQueue() const
{
	return {shared_from_this(), m_main_queue.get()};
}

cgpu::QueuePtr cgpu::DeviceSession::getAsyncGraphicsQueue() const
{
	return {shared_from_this(), m_async_graphics_queue.get()};
}

cgpu::QueuePtr cgpu::DeviceSession::getAsyncComputeQueue() const
{
	return {shared_from_this(), m_async_compute_queue.get()};
}

cgpu::QueuePtr cgpu::DeviceSession::getAsyncTransferQueue() const
{
	return {shared_from_this(), m_async_transfer_queue.get()};
}

void cgpu::DeviceSession::createDevice()
{
	// Reserve queues

	std::vector<QueueFamilyInfo> available_queues = queryAvailableQueues(getDevice());

	std::optional<uint32_t> main_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer
	);

	std::optional<uint32_t> async_graphics_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eGraphics
	);

	std::optional<uint32_t> async_compute_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eCompute
	);

	std::optional<uint32_t> async_transfer_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eTransfer
	);

	// Main queue is mandatory, other queues are optional
	if (!main_queue_family)
	{
		throw std::runtime_error("Could not find a suitable main queue.");
	}

	std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
	std::vector<std::vector<float>> queue_priorities;

	auto try_register_queue = [&](const std::optional<uint32_t>& family, float priority) -> std::optional<uint32_t>
	{
		if (!family)
		{
			return std::nullopt;
		}

		std::optional<size_t> index;
		for (size_t i = 0; i < queue_create_infos.size(); i++)
		{
			if (queue_create_infos[i].queueFamilyIndex == *family)
			{
				index = i;
				break;
			}
		}

		if (!index)
		{
			vk::DeviceQueueCreateInfo& info = queue_create_infos.emplace_back();
			info.flags = vk::DeviceQueueCreateFlagBits::eInternallySynchronizedKHR;
			info.queueFamilyIndex = *family;
			info.queueCount = *family;
			info.pQueuePriorities = nullptr;

			queue_priorities.emplace_back();

			index = queue_create_infos.size() - 1;
		}

		queue_priorities[*index].emplace_back(priority);

		queue_create_infos[*index].queueCount = static_cast<uint32_t>(queue_priorities[*index].size());
		queue_create_infos[*index].pQueuePriorities = queue_priorities[*index].data();

		return queue_create_infos[*index].queueCount - 1;
	};

	std::optional<uint32_t> main_queue_index = try_register_queue(main_queue_family, 1.0f);
	std::optional<uint32_t> async_graphics_queue_index = try_register_queue(async_graphics_queue_family, 0.0f);
	std::optional<uint32_t> async_compute_queue_index = try_register_queue(async_compute_queue_family, 0.0f);
	std::optional<uint32_t> async_transfer_queue_index = try_register_queue(async_transfer_queue_family, 0.0f);

	// Prepare features & extensions

	std::unordered_set<const char*, cgpu::StringHash, cgpu::StringEqualTo> unique_extensions;
	DynamicFeatureChain feature_chain;
	for (Device::Capability capability : magic_enum::enum_values<Device::Capability>())
	{
		if (!(m_device->getCapabilities() & capability))
		{
			continue;
		}

		auto capability_data = cgpu::Device::getCapabilityData(capability);
		unique_extensions.insert(capability_data->extensions.begin(), capability_data->extensions.end());
		capability_data->feature_callback(feature_chain);
	}

	std::vector<const char*> extensions{unique_extensions.begin(), unique_extensions.end()};

	// Create device

	vk::DeviceCreateInfo device_create_info;
	device_create_info.pNext = feature_chain.getHead();
	device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	device_create_info.pQueueCreateInfos = queue_create_infos.data();
	device_create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	device_create_info.ppEnabledExtensionNames = extensions.data();

	m_handle = m_device->getHandle().createDevice(device_create_info, nullptr, m_dispatcher);

	m_dispatcher.init(m_handle);

	// Create queues

	auto try_create_queue = [&](const std::optional<uint32_t>& family, const std::optional<uint32_t>& index) -> std::shared_ptr<Queue>
	{
		if (!family || !index)
		{
			return {};
		}

		vk::DeviceQueueInfo2 info;
		info.flags = vk::DeviceQueueCreateFlagBits::eInternallySynchronizedKHR;
		info.queueFamilyIndex = *family;
		info.queueIndex = *index;

		vk::Queue queue = m_handle.getQueue2(info, m_dispatcher);

		return std::make_shared<Queue>(Queue::PrivateKey{}, *this, queue);
	};

	m_main_queue = try_create_queue(main_queue_family, main_queue_index);
	m_async_graphics_queue = try_create_queue(async_graphics_queue_family, async_graphics_queue_index);
	m_async_compute_queue = try_create_queue(async_compute_queue_family, async_compute_queue_index);
	m_async_transfer_queue = try_create_queue(async_transfer_queue_family, async_transfer_queue_index);

	// Some queue types may not have a dedicated queue.
	// In this case, use the main queue.
	if (!m_async_graphics_queue)
	{
		m_async_graphics_queue = m_main_queue;
	}
	if (!m_async_compute_queue)
	{
		m_async_compute_queue = m_main_queue;
	}
	if (!m_async_transfer_queue)
	{
		m_async_transfer_queue = m_main_queue;
	}
}
