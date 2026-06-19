#include "SwapchainImage.hpp"

#include <CyphGPU/Image.hpp>
#include <CyphGPU/Swapchain.hpp>

cgpu::SwapchainImage::SwapchainImage(PrivateKey, Image::Desc&& desc, Swapchain& swapchain, uint32_t index):
	Image{Image::PrivateKey{}, swapchain.getDeviceSession(), std::move(desc)},
	m_swapchain{&swapchain},
	m_index{index}
{
}

cgpu::SwapchainPtr cgpu::SwapchainImage::getSwapchain() const
{
	return m_swapchain->shared_from_this();
}

const uint32_t& cgpu::SwapchainImage::getIndex() const
{
	return m_index;
}
