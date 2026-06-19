#pragma once

#include <CyphGPU/Image.hpp>

namespace cgpu
{
class SwapchainImage : public Image
{
	class PrivateKey
	{};

public:
	explicit SwapchainImage(PrivateKey, Image::Desc&& desc, Swapchain& swapchain, uint32_t index);

	[[nodiscard]]
	SwapchainPtr getSwapchain() const;

	[[nodiscard]]
	const uint32_t& getIndex() const;

private:
	friend class Swapchain;

	Swapchain* m_swapchain;
	uint32_t m_index;
};
}
