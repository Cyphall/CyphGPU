#pragma once

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace cgpu::detail
{
class DynamicFeatureChain final
{
public:
	DynamicFeatureChain()
	{
		// Ensure vk::PhysicalDeviceFeatures2 is first in the chain
		get<vk::PhysicalDeviceFeatures2>();
	}

	template<class T>
	T& get()
	{
		auto it = std::ranges::find_if(m_structures, [&](const auto& structure) { return structure.type == T::structureType; });
		if (it == m_structures.end())
		{
			m_structures.push_back({
				.type = T::structureType,
				.feature_count = (sizeof(T) - sizeof(vk::BaseOutStructure)) / sizeof(vk::Bool32),
				.data = std::make_shared<T>(),
			});
			it = m_structures.end() - 1;
		}

		return *static_cast<T*>(it->data.get());
	}

	[[nodiscard]]
	size_t getCount() const
	{
		return m_structures.size();
	}

	[[nodiscard]]
	std::span<const vk::Bool32> getBoolArray(size_t i) const
	{
		const std::byte* ptr = reinterpret_cast<const std::byte*>(m_structures[i].data.get());
		ptr += sizeof(vk::BaseOutStructure);
		return {reinterpret_cast<const vk::Bool32*>(ptr), m_structures[i].feature_count};
	}

	[[nodiscard]]
	vk::PhysicalDeviceFeatures2& getHead() const
	{
		for (size_t i = 0; i < m_structures.size() - 1; i++)
		{
			vk::BaseOutStructure& structure = *reinterpret_cast<vk::BaseOutStructure*>(m_structures[i].data.get());
			structure.pNext = reinterpret_cast<vk::BaseOutStructure*>(m_structures[i + 1].data.get());
		}

		return *static_cast<vk::PhysicalDeviceFeatures2*>(m_structures[0].data.get());
	}

private:
	struct Structure
	{
		vk::StructureType type;
		size_t feature_count;
		std::shared_ptr<void> data;
	};

	std::vector<Structure> m_structures{};
};
}
