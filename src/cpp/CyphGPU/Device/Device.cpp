#include "Device.hpp"

#include <CyphGPU/Context/Context.hpp>
#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <unordered_set>

namespace
{
std::span<const vk::Bool32> featureStructToBoolSpan(const cgpu::DynamicFeatureChain::Structure& structure)
{
	const std::byte* ptr = reinterpret_cast<const std::byte*>(structure.data.get());
	ptr += sizeof(vk::BaseOutStructure);
	return {reinterpret_cast<const vk::Bool32*>(ptr), structure.feature_count};
}
}

cgpu::Device::Device(PrivateKey, ContextSession& context_session, vk::PhysicalDevice physical_device):
	DependencyObject{context_session},
	m_context_session{context_session},
	m_physical_device{physical_device}
{
	checkCapabilitySupport();
}

const vk::PhysicalDevice& cgpu::Device::getHandle() const
{
	return m_physical_device;
}

cgpu::Device::Capabilities cgpu::Device::getCapabilities() const
{
	return m_capabilities;
}

boost::optional<const cgpu::Device::CapabilityData&> cgpu::Device::getCapabilityData(Capability capability)
{
	static CapabilityData core{
		{
			vk::EXTDescriptorHeapExtensionName,
			vk::KHRMaintenance7ExtensionName,
			vk::KHRMaintenance8ExtensionName,
			vk::KHRMaintenance9ExtensionName,
		},
		[](DynamicFeatureChain& chain)
		{
			{
				auto& features = chain.get<vk::PhysicalDeviceFeatures2>().features;
				// features.robustBufferAccess = true;
				features.fullDrawIndexUint32 = true;
				features.imageCubeArray = true;
				features.independentBlend = true;
				features.geometryShader = true;
				features.tessellationShader = true;
				features.sampleRateShading = true;
				features.dualSrcBlend = true;
				features.logicOp = true;
				features.multiDrawIndirect = true;
				features.drawIndirectFirstInstance = true;
				features.depthClamp = true;
				features.depthBiasClamp = true;
				features.fillModeNonSolid = true;
				// features.depthBounds = true;
				features.wideLines = true;
				features.largePoints = true;
				// features.alphaToOne = true;
			    // features.multiViewport = true;
				features.samplerAnisotropy = true;
				// features.textureCompressionETC2 = true;
			    // features.textureCompressionASTC_LDR = true;
				features.textureCompressionBC = true;
				// features.occlusionQueryPrecise = true;
			    // features.pipelineStatisticsQuery = true;
				features.vertexPipelineStoresAndAtomics = true;
				features.fragmentStoresAndAtomics = true;
				features.shaderTessellationAndGeometryPointSize = true;
				features.shaderImageGatherExtended = true;
				features.shaderStorageImageExtendedFormats = true;
				// features.shaderStorageImageMultisample = true;
				features.shaderStorageImageReadWithoutFormat = true;
				features.shaderStorageImageWriteWithoutFormat = true;
				// features.shaderUniformBufferArrayDynamicIndexing = true;
			    // features.shaderSampledImageArrayDynamicIndexing = true;
			    // features.shaderStorageBufferArrayDynamicIndexing = true;
			    // features.shaderStorageImageArrayDynamicIndexing = true;
				features.shaderClipDistance = true;
				features.shaderCullDistance = true;
				// features.shaderFloat64 = true;
				features.shaderInt64 = true;
				features.shaderInt16 = true;
				// features.shaderResourceResidency = true;
			    // features.shaderResourceMinLod = true;
			    // features.sparseBinding = true;
			    // features.sparseResidencyBuffer = true;
			    // features.sparseResidencyImage2D = true;
			    // features.sparseResidencyImage3D = true;
			    // features.sparseResidency2Samples = true;
			    // features.sparseResidency4Samples = true;
			    // features.sparseResidency8Samples = true;
			    // features.sparseResidency16Samples = true;
			    // features.sparseResidencyAliased = true;
			    // features.variableMultisampleRate = true;
			    // features.inheritedQueries = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan11Features>();
				features.storageBuffer16BitAccess = true;
				features.uniformAndStorageBuffer16BitAccess = true;
				// features.storagePushConstant16 = true;
			    // features.storageInputOutput16 = true;
				features.multiview = true;
				features.multiviewGeometryShader = true;
				features.multiviewTessellationShader = true;
				features.variablePointersStorageBuffer = true;
				features.variablePointers = true;
				// features.protectedMemory = true;
			    // features.samplerYcbcrConversion = true;
				features.shaderDrawParameters = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan12Features>();
				features.samplerMirrorClampToEdge = true;
				features.drawIndirectCount = true;
				features.storageBuffer8BitAccess = true;
				features.uniformAndStorageBuffer8BitAccess = true;
				// features.storagePushConstant8 = true;
				features.shaderBufferInt64Atomics = true;
				// features.shaderSharedInt64Atomics = true;
			    // features.shaderFloat16 = true;
				features.shaderInt8 = true;
				// features.descriptorIndexing = true;
			    // features.shaderInputAttachmentArrayDynamicIndexing = true;
			    // features.shaderUniformTexelBufferArrayDynamicIndexing = true;
			    // features.shaderStorageTexelBufferArrayDynamicIndexing = true;
			    // features.shaderUniformBufferArrayNonUniformIndexing = true;
			    // features.shaderSampledImageArrayNonUniformIndexing = true;
			    // features.shaderStorageBufferArrayNonUniformIndexing = true;
			    // features.shaderStorageImageArrayNonUniformIndexing = true;
			    // features.shaderInputAttachmentArrayNonUniformIndexing = true;
			    // features.shaderUniformTexelBufferArrayNonUniformIndexing = true;
			    // features.shaderStorageTexelBufferArrayNonUniformIndexing = true;
			    // features.descriptorBindingUniformBufferUpdateAfterBind = true;
			    // features.descriptorBindingSampledImageUpdateAfterBind = true;
			    // features.descriptorBindingStorageImageUpdateAfterBind = true;
			    // features.descriptorBindingStorageBufferUpdateAfterBind = true;
			    // features.descriptorBindingUniformTexelBufferUpdateAfterBind = true;
			    // features.descriptorBindingStorageTexelBufferUpdateAfterBind = true;
			    // features.descriptorBindingUpdateUnusedWhilePending = true;
			    // features.descriptorBindingPartiallyBound = true;
			    // features.descriptorBindingVariableDescriptorCount = true;
			    // features.runtimeDescriptorArray = true;
				features.samplerFilterMinmax = true;
				features.scalarBlockLayout = true;
				// features.imagelessFramebuffer = true;
				features.uniformBufferStandardLayout = true;
				features.shaderSubgroupExtendedTypes = true;
				features.separateDepthStencilLayouts = true;
				features.hostQueryReset = true;
				features.timelineSemaphore = true;
				features.bufferDeviceAddress = true;
				// features.bufferDeviceAddressCaptureReplay = true;
			    // features.bufferDeviceAddressMultiDevice = true;
				features.vulkanMemoryModel = true;
				features.vulkanMemoryModelDeviceScope = true;
				// features.vulkanMemoryModelAvailabilityVisibilityChains = true;
				features.shaderOutputViewportIndex = true;
				features.shaderOutputLayer = true;
				features.subgroupBroadcastDynamicId = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan13Features>();
				// features.robustImageAccess = true;
			    // features.inlineUniformBlock = true;
			    // features.descriptorBindingInlineUniformBlockUpdateAfterBind = true;
				features.pipelineCreationCacheControl = true;
				// features.privateData = true;
				features.shaderDemoteToHelperInvocation = true;
				features.shaderTerminateInvocation = true;
				features.subgroupSizeControl = true;
				features.computeFullSubgroups = true;
				features.synchronization2 = true;
				// features.textureCompressionASTC_HDR = true;
				features.shaderZeroInitializeWorkgroupMemory = true;
				features.dynamicRendering = true;
				features.shaderIntegerDotProduct = true;
				features.maintenance4 = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan14Features>();
				// features.globalPriorityQuery = true;
				features.shaderSubgroupRotate = true;
				features.shaderSubgroupRotateClustered = true;
				features.shaderFloatControls2 = true;
				features.shaderExpectAssume = true;
				// features.rectangularLines = true;
			    // features.bresenhamLines = true;
			    // features.smoothLines = true;
			    // features.stippledRectangularLines = true;
			    // features.stippledBresenhamLines = true;
			    // features.stippledSmoothLines = true;
			    // features.vertexAttributeInstanceRateDivisor = true;
			    // features.vertexAttributeInstanceRateZeroDivisor = true;
				features.indexTypeUint8 = true;
				features.dynamicRenderingLocalRead = true;
				features.maintenance5 = true;
				features.maintenance6 = true;
				// features.pipelineProtectedAccess = true;
			    // features.pipelineRobustness = true;
			    // features.hostImageCopy = true;
			    // features.pushDescriptor = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>();
				features.descriptorHeap = true;
				// features.descriptorHeapCaptureReplay = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceMaintenance7FeaturesKHR>();
				features.maintenance7 = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
				features.maintenance8 = true;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceMaintenance9FeaturesKHR>();
				features.maintenance9 = true;
			}
		}
	};

	static CapabilityData swapchain{
		{
			vk::KHRSwapchainExtensionName,
			vk::KHRSwapchainMutableFormatExtensionName,
			vk::KHRSwapchainMaintenance1ExtensionName,
		},
		[](DynamicFeatureChain& chain)
		{
			{
				auto& features = chain.get<vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>();
				features.swapchainMaintenance1 = true;
			}
		}
	};

	switch (capability)
	{
	case Capability::eCore: return core;
	case Capability::eSwapchain: return swapchain;
	default: std::terminate();
	}
}

void cgpu::Device::checkCapabilitySupport()
{
	// If Vulkan version is not supported, nothing is considered supported
	uint32_t api_version = getProperties<vk::PhysicalDeviceProperties2>().properties.apiVersion;
	if (api_version < Context::VULKAN_API_VERSION)
	{
		return;
	}

	std::unordered_set<std::string, cgpu::StringHash, cgpu::StringEqualTo> supported_extensions;
	for (const vk::ExtensionProperties& extension_properties : m_physical_device.enumerateDeviceExtensionProperties(nullptr, m_context_session->getDispatcher()))
	{
		supported_extensions.emplace(extension_properties.extensionName.data());
	}

	auto is_capability_supported = [&](Capability capability) -> bool
	{
		boost::optional<const CapabilityData&> capability_data = getCapabilityData(capability);
		if (!capability_data)
		{
			return false;
		}

		for (const char* required_extension : capability_data->extensions)
		{
			if (!supported_extensions.contains(required_extension))
			{
				return false;
			}
		}

		// This will set some features to true but the driver should overwrite that so it's fine
		DynamicFeatureChain supported_structures;
		capability_data->feature_callback(supported_structures);
		m_physical_device.getFeatures2(&supported_structures.get<vk::PhysicalDeviceFeatures2>(), m_context_session->getDispatcher());

		DynamicFeatureChain required_structures;
		capability_data->feature_callback(required_structures);

		for (const auto& [type, required_structure] : required_structures.getStructures())
		{
			std::span<const vk::Bool32> required_features = featureStructToBoolSpan(required_structure);
			std::span<const vk::Bool32> supported_features = featureStructToBoolSpan(supported_structures.getStructures().at(type));
			for (size_t i = 0; i < required_features.size(); i++)
			{
				if (required_features[i] && !supported_features[i])
				{
					return false;
				}
			}
		}

		return true;
	};

	for (Capability capability : magic_enum::enum_values<Capability>())
	{
		if (is_capability_supported(capability))
		{
			m_capabilities |= capability;
		}
		// If core is not supported, nothing is considered supported
		else if (capability == Capability::eCore)
		{
			return;
		}
	}
}
