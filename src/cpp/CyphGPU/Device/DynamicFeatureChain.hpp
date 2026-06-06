#pragma once

#include <boost/optional/optional.hpp>
#include <memory>
#include <unordered_map>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class DynamicFeatureChain final
{
public:
	struct Structure
	{
		size_t feature_count;
		std::shared_ptr<void> data;
	};

	template<class T>
	T& get()
	{
		auto [it, inserted] = m_structures.try_emplace(T::structureType);
		if (inserted)
		{
			it->second.feature_count = (sizeof(T) - sizeof(vk::BaseOutStructure)) / sizeof(vk::Bool32);
			it->second.data = std::make_shared<T>();
			vk::BaseOutStructure& structure = *reinterpret_cast<vk::BaseOutStructure*>(it->second.data.get());
			if (m_head)
			{
				structure.pNext = m_head.get_ptr();
			}

			m_head = structure;
		}

		return *static_cast<T*>(it->second.data.get());
	}

	[[nodiscard]]
	const std::unordered_map<vk::StructureType, Structure>& getStructures() const
	{
		return m_structures;
	}

	[[nodiscard]]
	vk::BaseOutStructure& getHead()
	{
		return *m_head;
	}

private:
	std::unordered_map<vk::StructureType, Structure> m_structures{};
	boost::optional<vk::BaseOutStructure&> m_head{};
};
}
