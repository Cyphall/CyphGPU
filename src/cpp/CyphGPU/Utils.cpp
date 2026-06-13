#include "Utils.hpp"

vk::Format cgpu::getLinearEquivalent(vk::Format format)
{
	switch (format)
	{
	case vk::Format::eR8Srgb: return vk::Format::eR8Unorm;
	case vk::Format::eR8G8Srgb: return vk::Format::eR8G8Unorm;
	case vk::Format::eR8G8B8Srgb: return vk::Format::eR8G8B8Unorm;
	case vk::Format::eB8G8R8Srgb: return vk::Format::eB8G8R8Unorm;
	case vk::Format::eR8G8B8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
	case vk::Format::eB8G8R8A8Srgb: return vk::Format::eB8G8R8A8Unorm;
	case vk::Format::eA8B8G8R8SrgbPack32: return vk::Format::eA8B8G8R8UnormPack32;
	case vk::Format::eBc1RgbSrgbBlock: return vk::Format::eBc1RgbUnormBlock;
	case vk::Format::eBc1RgbaSrgbBlock: return vk::Format::eBc1RgbaUnormBlock;
	case vk::Format::eBc2SrgbBlock: return vk::Format::eBc2UnormBlock;
	case vk::Format::eBc3SrgbBlock: return vk::Format::eBc3UnormBlock;
	case vk::Format::eBc7SrgbBlock: return vk::Format::eBc7UnormBlock;
	case vk::Format::eEtc2R8G8B8SrgbBlock: return vk::Format::eEtc2R8G8B8UnormBlock;
	case vk::Format::eEtc2R8G8B8A1SrgbBlock: return vk::Format::eEtc2R8G8B8A1UnormBlock;
	case vk::Format::eEtc2R8G8B8A8SrgbBlock: return vk::Format::eEtc2R8G8B8A8UnormBlock;
	case vk::Format::eAstc4x4SrgbBlock: return vk::Format::eAstc4x4UnormBlock;
	case vk::Format::eAstc5x4SrgbBlock: return vk::Format::eAstc5x4UnormBlock;
	case vk::Format::eAstc5x5SrgbBlock: return vk::Format::eAstc5x5UnormBlock;
	case vk::Format::eAstc6x5SrgbBlock: return vk::Format::eAstc6x5UnormBlock;
	case vk::Format::eAstc6x6SrgbBlock: return vk::Format::eAstc6x6UnormBlock;
	case vk::Format::eAstc8x5SrgbBlock: return vk::Format::eAstc8x5UnormBlock;
	case vk::Format::eAstc8x6SrgbBlock: return vk::Format::eAstc8x6UnormBlock;
	case vk::Format::eAstc8x8SrgbBlock: return vk::Format::eAstc8x8UnormBlock;
	case vk::Format::eAstc10x5SrgbBlock: return vk::Format::eAstc10x5UnormBlock;
	case vk::Format::eAstc10x6SrgbBlock: return vk::Format::eAstc10x6UnormBlock;
	case vk::Format::eAstc10x8SrgbBlock: return vk::Format::eAstc10x8UnormBlock;
	case vk::Format::eAstc10x10SrgbBlock: return vk::Format::eAstc10x10UnormBlock;
	case vk::Format::eAstc12x10SrgbBlock: return vk::Format::eAstc12x10UnormBlock;
	case vk::Format::eAstc12x12SrgbBlock: return vk::Format::eAstc12x12UnormBlock;
	case vk::Format::ePvrtc12BppSrgbBlockIMG: return vk::Format::ePvrtc12BppUnormBlockIMG;
	case vk::Format::ePvrtc14BppSrgbBlockIMG: return vk::Format::ePvrtc14BppUnormBlockIMG;
	case vk::Format::ePvrtc22BppSrgbBlockIMG: return vk::Format::ePvrtc22BppUnormBlockIMG;
	case vk::Format::ePvrtc24BppSrgbBlockIMG: return vk::Format::ePvrtc24BppUnormBlockIMG;
	case vk::Format::eAstc3x3x3SrgbBlockEXT: return vk::Format::eAstc3x3x3UnormBlockEXT;
	case vk::Format::eAstc4x3x3SrgbBlockEXT: return vk::Format::eAstc4x3x3UnormBlockEXT;
	case vk::Format::eAstc4x4x3SrgbBlockEXT: return vk::Format::eAstc4x4x3UnormBlockEXT;
	case vk::Format::eAstc4x4x4SrgbBlockEXT: return vk::Format::eAstc4x4x4UnormBlockEXT;
	case vk::Format::eAstc5x4x4SrgbBlockEXT: return vk::Format::eAstc5x4x4UnormBlockEXT;
	case vk::Format::eAstc5x5x4SrgbBlockEXT: return vk::Format::eAstc5x5x4UnormBlockEXT;
	case vk::Format::eAstc5x5x5SrgbBlockEXT: return vk::Format::eAstc5x5x5UnormBlockEXT;
	case vk::Format::eAstc6x5x5SrgbBlockEXT: return vk::Format::eAstc6x5x5UnormBlockEXT;
	case vk::Format::eAstc6x6x5SrgbBlockEXT: return vk::Format::eAstc6x6x5UnormBlockEXT;
	case vk::Format::eAstc6x6x6SrgbBlockEXT: return vk::Format::eAstc6x6x6UnormBlockEXT;
	default: return format;
	}
}

vk::Format cgpu::getSrgbEquivalent(vk::Format format)
{
	switch (format)
	{
	case vk::Format::eR8Unorm: return vk::Format::eR8Srgb;
	case vk::Format::eR8G8Unorm: return vk::Format::eR8G8Srgb;
	case vk::Format::eR8G8B8Unorm: return vk::Format::eR8G8B8Srgb;
	case vk::Format::eB8G8R8Unorm: return vk::Format::eB8G8R8Srgb;
	case vk::Format::eR8G8B8A8Unorm: return vk::Format::eR8G8B8A8Srgb;
	case vk::Format::eB8G8R8A8Unorm: return vk::Format::eB8G8R8A8Srgb;
	case vk::Format::eA8B8G8R8UnormPack32: return vk::Format::eA8B8G8R8SrgbPack32;
	case vk::Format::eBc1RgbUnormBlock: return vk::Format::eBc1RgbSrgbBlock;
	case vk::Format::eBc1RgbaUnormBlock: return vk::Format::eBc1RgbaSrgbBlock;
	case vk::Format::eBc2UnormBlock: return vk::Format::eBc2SrgbBlock;
	case vk::Format::eBc3UnormBlock: return vk::Format::eBc3SrgbBlock;
	case vk::Format::eBc7UnormBlock: return vk::Format::eBc7SrgbBlock;
	case vk::Format::eEtc2R8G8B8UnormBlock: return vk::Format::eEtc2R8G8B8SrgbBlock;
	case vk::Format::eEtc2R8G8B8A1UnormBlock: return vk::Format::eEtc2R8G8B8A1SrgbBlock;
	case vk::Format::eEtc2R8G8B8A8UnormBlock: return vk::Format::eEtc2R8G8B8A8SrgbBlock;
	case vk::Format::eAstc4x4UnormBlock: return vk::Format::eAstc4x4SrgbBlock;
	case vk::Format::eAstc5x4UnormBlock: return vk::Format::eAstc5x4SrgbBlock;
	case vk::Format::eAstc5x5UnormBlock: return vk::Format::eAstc5x5SrgbBlock;
	case vk::Format::eAstc6x5UnormBlock: return vk::Format::eAstc6x5SrgbBlock;
	case vk::Format::eAstc6x6UnormBlock: return vk::Format::eAstc6x6SrgbBlock;
	case vk::Format::eAstc8x5UnormBlock: return vk::Format::eAstc8x5SrgbBlock;
	case vk::Format::eAstc8x6UnormBlock: return vk::Format::eAstc8x6SrgbBlock;
	case vk::Format::eAstc8x8UnormBlock: return vk::Format::eAstc8x8SrgbBlock;
	case vk::Format::eAstc10x5UnormBlock: return vk::Format::eAstc10x5SrgbBlock;
	case vk::Format::eAstc10x6UnormBlock: return vk::Format::eAstc10x6SrgbBlock;
	case vk::Format::eAstc10x8UnormBlock: return vk::Format::eAstc10x8SrgbBlock;
	case vk::Format::eAstc10x10UnormBlock: return vk::Format::eAstc10x10SrgbBlock;
	case vk::Format::eAstc12x10UnormBlock: return vk::Format::eAstc12x10SrgbBlock;
	case vk::Format::eAstc12x12UnormBlock: return vk::Format::eAstc12x12SrgbBlock;
	case vk::Format::ePvrtc12BppUnormBlockIMG: return vk::Format::ePvrtc12BppSrgbBlockIMG;
	case vk::Format::ePvrtc14BppUnormBlockIMG: return vk::Format::ePvrtc14BppSrgbBlockIMG;
	case vk::Format::ePvrtc22BppUnormBlockIMG: return vk::Format::ePvrtc22BppSrgbBlockIMG;
	case vk::Format::ePvrtc24BppUnormBlockIMG: return vk::Format::ePvrtc24BppSrgbBlockIMG;
	case vk::Format::eAstc3x3x3UnormBlockEXT: return vk::Format::eAstc3x3x3SrgbBlockEXT;
	case vk::Format::eAstc4x3x3UnormBlockEXT: return vk::Format::eAstc4x3x3SrgbBlockEXT;
	case vk::Format::eAstc4x4x3UnormBlockEXT: return vk::Format::eAstc4x4x3SrgbBlockEXT;
	case vk::Format::eAstc4x4x4UnormBlockEXT: return vk::Format::eAstc4x4x4SrgbBlockEXT;
	case vk::Format::eAstc5x4x4UnormBlockEXT: return vk::Format::eAstc5x4x4SrgbBlockEXT;
	case vk::Format::eAstc5x5x4UnormBlockEXT: return vk::Format::eAstc5x5x4SrgbBlockEXT;
	case vk::Format::eAstc5x5x5UnormBlockEXT: return vk::Format::eAstc5x5x5SrgbBlockEXT;
	case vk::Format::eAstc6x5x5UnormBlockEXT: return vk::Format::eAstc6x5x5SrgbBlockEXT;
	case vk::Format::eAstc6x6x5UnormBlockEXT: return vk::Format::eAstc6x6x5SrgbBlockEXT;
	case vk::Format::eAstc6x6x6UnormBlockEXT: return vk::Format::eAstc6x6x6SrgbBlockEXT;
	default: return format;
	}
}

vk::ImageAspectFlags cgpu::getAspects(vk::Format format)
{
	switch (format)
	{
	case vk::Format::eD16Unorm:
	case vk::Format::eX8D24UnormPack32:
	case vk::Format::eD32Sfloat:
		return vk::ImageAspectFlagBits::eDepth;
	case vk::Format::eS8Uint:
		return vk::ImageAspectFlagBits::eStencil;
	case vk::Format::eD16UnormS8Uint:
	case vk::Format::eD24UnormS8Uint:
	case vk::Format::eD32SfloatS8Uint:
		return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
	default:
		return vk::ImageAspectFlagBits::eColor;
	}
}
