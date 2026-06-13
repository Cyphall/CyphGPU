#include "Device.hpp"

#include <CyphGPU/Context/Context.hpp>
#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <unordered_set>
#include <utility>

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
	m_context_session{&context_session},
	m_handle{physical_device}
{
	checkCapabilitySupport();
}

cgpu::ContextSessionPtr cgpu::Device::getContextSession() const
{
	return m_context_session->shared_from_this();
}

const vk::PhysicalDevice& cgpu::Device::getHandle()
{
	return m_handle;
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
			vk::KHRInternallySynchronizedQueuesExtensionName,
			vk::KHRPipelineLibraryExtensionName,
			vk::EXTGraphicsPipelineLibraryExtensionName,
			vk::KHRMaintenance7ExtensionName,
			vk::KHRMaintenance8ExtensionName,
			vk::KHRMaintenance9ExtensionName,
		},
		[](DynamicFeatureChain& chain) {
			{
				auto& features = chain.get<vk::PhysicalDeviceFeatures2>().features;
				// features.robustBufferAccess = vk::True;
				features.fullDrawIndexUint32 = vk::True;
				features.imageCubeArray = vk::True;
				features.independentBlend = vk::True;
				features.geometryShader = vk::True;
				features.tessellationShader = vk::True;
				features.sampleRateShading = vk::True;
				features.dualSrcBlend = vk::True;
				features.logicOp = vk::True;
				features.multiDrawIndirect = vk::True;
				features.drawIndirectFirstInstance = vk::True;
				features.depthClamp = vk::True;
				features.depthBiasClamp = vk::True;
				features.fillModeNonSolid = vk::True;
				// features.depthBounds = vk::True;
				features.wideLines = vk::True;
				features.largePoints = vk::True;
				// features.alphaToOne = vk::True;
			    // features.multiViewport = vk::True;
				features.samplerAnisotropy = vk::True;
				// features.textureCompressionETC2 = vk::True;
			    // features.textureCompressionASTC_LDR = vk::True;
				features.textureCompressionBC = vk::True;
				// features.occlusionQueryPrecise = vk::True;
			    // features.pipelineStatisticsQuery = vk::True;
				features.vertexPipelineStoresAndAtomics = vk::True;
				features.fragmentStoresAndAtomics = vk::True;
				features.shaderTessellationAndGeometryPointSize = vk::True;
				features.shaderImageGatherExtended = vk::True;
				features.shaderStorageImageExtendedFormats = vk::True;
				// features.shaderStorageImageMultisample = vk::True;
				features.shaderStorageImageReadWithoutFormat = vk::True;
				features.shaderStorageImageWriteWithoutFormat = vk::True;
				// features.shaderUniformBufferArrayDynamicIndexing = vk::True;
			    // features.shaderSampledImageArrayDynamicIndexing = vk::True;
			    // features.shaderStorageBufferArrayDynamicIndexing = vk::True;
			    // features.shaderStorageImageArrayDynamicIndexing = vk::True;
				features.shaderClipDistance = vk::True;
				features.shaderCullDistance = vk::True;
				// features.shaderFloat64 = vk::True;
				features.shaderInt64 = vk::True;
				features.shaderInt16 = vk::True;
				// features.shaderResourceResidency = vk::True;
			    // features.shaderResourceMinLod = vk::True;
			    // features.sparseBinding = vk::True;
			    // features.sparseResidencyBuffer = vk::True;
			    // features.sparseResidencyImage2D = vk::True;
			    // features.sparseResidencyImage3D = vk::True;
			    // features.sparseResidency2Samples = vk::True;
			    // features.sparseResidency4Samples = vk::True;
			    // features.sparseResidency8Samples = vk::True;
			    // features.sparseResidency16Samples = vk::True;
			    // features.sparseResidencyAliased = vk::True;
			    // features.variableMultisampleRate = vk::True;
			    // features.inheritedQueries = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan11Features>();
				features.storageBuffer16BitAccess = vk::True;
				features.uniformAndStorageBuffer16BitAccess = vk::True;
				// features.storagePushConstant16 = vk::True;
			    // features.storageInputOutput16 = vk::True;
				features.multiview = vk::True;
				features.multiviewGeometryShader = vk::True;
				features.multiviewTessellationShader = vk::True;
				features.variablePointersStorageBuffer = vk::True;
				features.variablePointers = vk::True;
				// features.protectedMemory = vk::True;
			    // features.samplerYcbcrConversion = vk::True;
				features.shaderDrawParameters = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan12Features>();
				features.samplerMirrorClampToEdge = vk::True;
				features.drawIndirectCount = vk::True;
				features.storageBuffer8BitAccess = vk::True;
				features.uniformAndStorageBuffer8BitAccess = vk::True;
				// features.storagePushConstant8 = vk::True;
				features.shaderBufferInt64Atomics = vk::True;
				// features.shaderSharedInt64Atomics = vk::True;
			    // features.shaderFloat16 = vk::True;
				features.shaderInt8 = vk::True;
				// features.descriptorIndexing = vk::True;
			    // features.shaderInputAttachmentArrayDynamicIndexing = vk::True;
			    // features.shaderUniformTexelBufferArrayDynamicIndexing = vk::True;
			    // features.shaderStorageTexelBufferArrayDynamicIndexing = vk::True;
			    // features.shaderUniformBufferArrayNonUniformIndexing = vk::True;
			    // features.shaderSampledImageArrayNonUniformIndexing = vk::True;
			    // features.shaderStorageBufferArrayNonUniformIndexing = vk::True;
			    // features.shaderStorageImageArrayNonUniformIndexing = vk::True;
			    // features.shaderInputAttachmentArrayNonUniformIndexing = vk::True;
			    // features.shaderUniformTexelBufferArrayNonUniformIndexing = vk::True;
			    // features.shaderStorageTexelBufferArrayNonUniformIndexing = vk::True;
			    // features.descriptorBindingUniformBufferUpdateAfterBind = vk::True;
			    // features.descriptorBindingSampledImageUpdateAfterBind = vk::True;
			    // features.descriptorBindingStorageImageUpdateAfterBind = vk::True;
			    // features.descriptorBindingStorageBufferUpdateAfterBind = vk::True;
			    // features.descriptorBindingUniformTexelBufferUpdateAfterBind = vk::True;
			    // features.descriptorBindingStorageTexelBufferUpdateAfterBind = vk::True;
			    // features.descriptorBindingUpdateUnusedWhilePending = vk::True;
			    // features.descriptorBindingPartiallyBound = vk::True;
			    // features.descriptorBindingVariableDescriptorCount = vk::True;
			    // features.runtimeDescriptorArray = vk::True;
				features.samplerFilterMinmax = vk::True;
				features.scalarBlockLayout = vk::True;
				// features.imagelessFramebuffer = vk::True;
				features.uniformBufferStandardLayout = vk::True;
				features.shaderSubgroupExtendedTypes = vk::True;
				features.separateDepthStencilLayouts = vk::True;
				features.hostQueryReset = vk::True;
				features.timelineSemaphore = vk::True;
				features.bufferDeviceAddress = vk::True;
				// features.bufferDeviceAddressCaptureReplay = vk::True;
			    // features.bufferDeviceAddressMultiDevice = vk::True;
				features.vulkanMemoryModel = vk::True;
				features.vulkanMemoryModelDeviceScope = vk::True;
				// features.vulkanMemoryModelAvailabilityVisibilityChains = vk::True;
				features.shaderOutputViewportIndex = vk::True;
				features.shaderOutputLayer = vk::True;
				features.subgroupBroadcastDynamicId = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan13Features>();
				// features.robustImageAccess = vk::True;
			    // features.inlineUniformBlock = vk::True;
			    // features.descriptorBindingInlineUniformBlockUpdateAfterBind = vk::True;
				features.pipelineCreationCacheControl = vk::True;
				// features.privateData = vk::True;
				features.shaderDemoteToHelperInvocation = vk::True;
				features.shaderTerminateInvocation = vk::True;
				features.subgroupSizeControl = vk::True;
				features.computeFullSubgroups = vk::True;
				features.synchronization2 = vk::True;
				// features.textureCompressionASTC_HDR = vk::True;
				features.shaderZeroInitializeWorkgroupMemory = vk::True;
				features.dynamicRendering = vk::True;
				features.shaderIntegerDotProduct = vk::True;
				features.maintenance4 = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceVulkan14Features>();
				// features.globalPriorityQuery = vk::True;
				features.shaderSubgroupRotate = vk::True;
				features.shaderSubgroupRotateClustered = vk::True;
				features.shaderFloatControls2 = vk::True;
				features.shaderExpectAssume = vk::True;
				// features.rectangularLines = vk::True;
			    // features.bresenhamLines = vk::True;
			    // features.smoothLines = vk::True;
			    // features.stippledRectangularLines = vk::True;
			    // features.stippledBresenhamLines = vk::True;
			    // features.stippledSmoothLines = vk::True;
			    // features.vertexAttributeInstanceRateDivisor = vk::True;
			    // features.vertexAttributeInstanceRateZeroDivisor = vk::True;
				features.indexTypeUint8 = vk::True;
				features.dynamicRenderingLocalRead = vk::True;
				features.maintenance5 = vk::True;
				features.maintenance6 = vk::True;
				// features.pipelineProtectedAccess = vk::True;
			    // features.pipelineRobustness = vk::True;
			    // features.hostImageCopy = vk::True;
			    // features.pushDescriptor = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>();
				features.descriptorHeap = vk::True;
				// features.descriptorHeapCaptureReplay = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceInternallySynchronizedQueuesFeaturesKHR>();
				features.internallySynchronizedQueues = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>();
				features.graphicsPipelineLibrary = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceMaintenance7FeaturesKHR>();
				features.maintenance7 = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
				features.maintenance8 = vk::True;
			}

			{
				auto& features = chain.get<vk::PhysicalDeviceMaintenance9FeaturesKHR>();
				features.maintenance9 = vk::True;
			}
		}
	};

	static CapabilityData swapchain{
		{
			vk::KHRSwapchainExtensionName,
			vk::KHRSwapchainMutableFormatExtensionName,
			vk::KHRSwapchainMaintenance1ExtensionName,
		},
		[](DynamicFeatureChain& chain) {
			{
				auto& features = chain.get<vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>();
				features.swapchainMaintenance1 = vk::True;
			}
		}
	};

	static CapabilityData memory_budget{
		{
			vk::EXTMemoryBudgetExtensionName,
		},
		[](DynamicFeatureChain&) {}
	};

	static CapabilityData memory_priority{
		{
			vk::EXTMemoryPriorityExtensionName,
		},
		[](DynamicFeatureChain& chain) {
			{
				auto& features = chain.get<vk::PhysicalDeviceMemoryPriorityFeaturesEXT>();
				features.memoryPriority = vk::True;
			}
		}
	};

	switch (capability)
	{
	case Capability::eCore: return core;
	case Capability::eSwapchain: return swapchain;
	case Capability::eMemoryBudget: return memory_budget;
	case Capability::eMemoryPriority: return memory_priority;
	default: std::unreachable();
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
	for (const vk::ExtensionProperties& extension_properties : m_handle.enumerateDeviceExtensionProperties(nullptr, m_context_session->getDispatcher()))
	{
		supported_extensions.emplace(extension_properties.extensionName.data());
	}

	auto is_capability_supported = [&](Capability capability) -> bool {
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
		m_handle.getFeatures2(&supported_structures.get<vk::PhysicalDeviceFeatures2>(), m_context_session->getDispatcher());

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
