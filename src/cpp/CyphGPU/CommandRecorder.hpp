#pragma once

#include <CyphGPU/CommandContext.hpp>
#include <CyphGPU/detail/BumpAllocator.hpp>
#include <CyphGPU/detail/Resource.hpp>
#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Utils.hpp>

#include <array>
#include <boost/container/static_vector.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <type_traits>
#include <variant>

namespace cgpu
{
class ScopedDebugRegion;

class CommandRecorder
{
public:
	class SubmitHandle
	{
	public:
		void waitFinished() const;

		[[nodiscard]]
		bool isFinished() const;

	private:
		friend class CommandRecorder;

		DeviceSessionPtr m_device_session;
		Queue::Signal m_signal;

		SubmitHandle(const DeviceSessionPtr& device_session, Queue::Signal signal);

		[[nodiscard]]
		vk::Result waitSemaphore(uint64_t timeout) const;
	};

	CommandRecorder(const CommandRecorder&) = delete;
	CommandRecorder(CommandRecorder&&) = delete;

	CommandRecorder& operator=(const CommandRecorder&) = delete;
	CommandRecorder& operator=(CommandRecorder&&) = delete;

	SubmitHandle submit();

	// ----- Common structs -----

	struct ImageLevelsLayersRange
	{
		/// Optional. Default: All levels.
		std::optional<Range<uint32_t>> levels{};
		/// Optional. Default: All layers.
		std::optional<Range<uint32_t>> layers{};
	};

	struct ImageLevelLayersAspectsPixelsRange
	{
		/// Optional. Default: Level 0.
		uint32_t level{0};
		/// Optional. Default: All layers.
		std::optional<Range<uint32_t>> layers{};
		/// Optional. Default: All aspects.
		std::optional<vk::ImageAspectFlags> aspects{};
		/// Optional. Default: All pixels.
		std::optional<Range<glm::uvec3>> pixels{};
	};

	struct ImageLevelLayersAspectPixelsRange
	{
		/// Optional. Default: Level 0.
		uint32_t level{0};
		/// Optional. Default: All layers.
		std::optional<Range<uint32_t>> layers{};
		/// Optional. Default: Main aspect. For depth-stencil formats, the default aspect is depth.
		std::optional<vk::ImageAspectFlagBits> aspect{};
		/// Optional. Default: All pixels.
		std::optional<Range<glm::uvec3>> pixels{};
	};

	struct ImageLevelLayersAspectsRectRange
	{
		/// Optional. Default: Level 0.
		uint32_t level{0};
		/// Optional. Default: All layers.
		std::optional<Range<uint32_t>> layers{};
		/// Optional. Default: All aspects.
		std::optional<vk::ImageAspectFlags> aspects{};
		/// Optional. Default: [0, 0, 0].
		glm::uvec3 top_left{0, 0, 0};
		/// Optional. Default: Image extent.
		std::optional<glm::uvec3> bottom_right{};
	};

	struct BufferRange
	{
		/// Optional. Default: All bytes.
		std::optional<Range<vk::DeviceSize>> byte_range{};
	};

	using ColorValue = std::variant<glm::vec4, glm::ivec4, glm::uvec4>;
	using DepthValue = float;
	using StencilValue = uint32_t;

	// ----- Commands -----

	struct ClearImageParams
	{
		static const std::array<ImageLevelsLayersRange, 1> DEFAULT_RANGE;

		/// Required.
		ImagePtr image CGPU_REQUIRED;
		/// Optional. Default: One default-initialized range.
		std::span<const ImageLevelsLayersRange> ranges{DEFAULT_RANGE};
		/// Optional. Default: Empty (no color clear).
		std::optional<ColorValue> color_value{};
		/// Optional. Default: Empty (no depth clear).
		std::optional<DepthValue> depth_value{};
		/// Optional. Default: Empty (no stencil clear).
		std::optional<StencilValue> stencil_value{};
	};

	void clearImage(ClearImageParams&& params);

	struct CopyImageToImageParams
	{
		struct Range
		{
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectsPixelsRange src{};
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectsPixelsRange dst{};
		};

		static const std::array<Range, 1> DEFAULT_RANGE;

		/// Required.
		ImagePtr src_image CGPU_REQUIRED;
		/// Required.
		ImagePtr dst_image CGPU_REQUIRED;
		/// Optional. Default: One default-initialized range.
		std::span<const Range> ranges{DEFAULT_RANGE};
	};

	void copyImageToImage(CopyImageToImageParams&& params);

	struct CopyBufferToImageParams
	{
		struct Range
		{
			/// Optional. Default: Default-initialized range.
			BufferRange src{};
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectPixelsRange dst{};
		};

		static const std::array<Range, 1> DEFAULT_RANGE;

		/// Required.
		BufferPtr src_buffer CGPU_REQUIRED;
		/// Required.
		ImagePtr dst_image CGPU_REQUIRED;
		/// Optional. Default: One default-initialized range.
		std::span<const Range> ranges{DEFAULT_RANGE};
	};

	void copyBufferToImage(CopyBufferToImageParams&& params);

	struct CopyImageToBufferParams
	{
		struct Range
		{
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectPixelsRange src{};
			/// Optional. Default: Default-initialized range.
			BufferRange dst{};
		};

		static const std::array<Range, 1> DEFAULT_RANGE;

		/// Required.
		ImagePtr src_image CGPU_REQUIRED;
		/// Required.
		BufferPtr dst_buffer CGPU_REQUIRED;
		/// Optional. Default: One default-initialized range.
		std::span<const Range> ranges{DEFAULT_RANGE};
	};

	void copyImageToBuffer(CopyImageToBufferParams&& params);

	struct CopyBufferToBufferParams
	{
		struct Range
		{
			/// Optional. Default: Default-initialized range.
			BufferRange src{};
			/// Optional. Default: Default-initialized range.
			BufferRange dst{};
		};

		static const std::array<Range, 1> DEFAULT_RANGE;

		/// Required.
		BufferPtr src_buffer CGPU_REQUIRED;
		/// Required.
		BufferPtr dst_buffer CGPU_REQUIRED;
		/// Optional. Default: One default-initialized range.
		std::span<const Range> ranges{DEFAULT_RANGE};
	};

	void copyBufferToBuffer(CopyBufferToBufferParams&& params);

	struct BlitParams
	{
		struct Range
		{
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectsRectRange src{};
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectsRectRange dst{};
		};

		static const std::array<Range, 1> DEFAULT_RANGE;

		/// Required.
		ImagePtr src_image CGPU_REQUIRED;
		/// Required.
		ImagePtr dst_image CGPU_REQUIRED;
		/// Optional. Default: Nearest.
		vk::Filter filter{vk::Filter::eNearest};
		/// Optional. Default: One default-initialized range.
		std::span<const Range> ranges{DEFAULT_RANGE};
	};

	void blit(BlitParams&& params);

	struct GraphicsPassParams
	{
		struct ColorAttachment
		{
			struct Resolve
			{
				/// Required.
				ImagePtr image CGPU_REQUIRED;
				/// Optional. Default: Level 0.
				uint32_t level{0};
				/// Optional. Default: Layer 0.
				uint32_t first_layer{0};
			};

			/// Required.
			ImagePtr image CGPU_REQUIRED;
			/// Optional. Default: Image format.
			std::optional<vk::Format> format{};
			/// Optional. Default: Level 0.
			uint32_t level{0};
			/// Optional. Default: Layer 0.
			uint32_t first_layer{0};
			/// Required.
			vk::AttachmentLoadOp load_op CGPU_REQUIRED;
			/// Required.
			vk::AttachmentStoreOp store_op CGPU_REQUIRED;
			/// Optional. Default: Empty. Must be set if load_op == eClear.
			std::optional<ColorValue> clear_color_value{};
			/// Optional. Default: No resolve.
			std::optional<Resolve> resolve{};
		};

		struct DepthStencilAttachment
		{
			struct Resolve
			{
				/// Required.
				ImagePtr image CGPU_REQUIRED;
				/// Optional. Default: SampleZero. Only used when depth is enabled.
				vk::ResolveModeFlagBits depth_mode{vk::ResolveModeFlagBits::eSampleZero};
				/// Optional. Default: SampleZero. Only used when stencil is enabled.
				vk::ResolveModeFlagBits stencil_mode{vk::ResolveModeFlagBits::eSampleZero};
				/// Optional. Default: Level 0.
				uint32_t level{0};
				/// Optional. Default: Layer 0.
				uint32_t first_layer{0};
			};

			/// Required.
			ImagePtr image CGPU_REQUIRED;
			/// Optional. Default: Level 0.
			uint32_t level{0};
			/// Optional. Default: Layer 0.
			uint32_t first_layer{0};
			/// Optional. Default: True if the format has a depth aspect.
			std::optional<bool> enable_depth{};
			/// Optional. Default: True if the format has a stencil aspect.
			std::optional<bool> enable_stencil{};
			/// Required.
			vk::AttachmentLoadOp load_op CGPU_REQUIRED;
			/// Required.
			vk::AttachmentStoreOp store_op CGPU_REQUIRED;
			/// Optional. Default: Empty. Must be set if load_op == eClear and depth is enabled.
			std::optional<DepthValue> clear_depth_value{};
			/// Optional. Default: Empty. Must be set if load_op == eClear and stencil is enabled.
			std::optional<StencilValue> clear_stencil_value{};
			/// Optional. Default: No resolve.
			std::optional<Resolve> resolve{};
		};

		struct LayerCount
		{
			/// Required.
			uint32_t value CGPU_REQUIRED;
		};

		struct MultiviewMask
		{
			/// Required.
			uint32_t value CGPU_REQUIRED;
		};

		/// Optional. Default: Attachment images extent.
		///
		/// Must be set if there is no attachment or if attachments have different extents.
		std::optional<Range<glm::uvec2>> render_area{};
		/// Optional. Default: Single layer.
		std::variant<LayerCount, MultiviewMask> layer_mode{LayerCount{1}};
		/// Optional. Default: No color attachment.
		boost::container::static_vector<ColorAttachment, 8> color_attachments{};
		/// Optional. Default: No depth-stencil attachment.
		std::optional<DepthStencilAttachment> depth_stencil_attachment{};
		/// Required.
		std::function<void(GraphicsPassContext& ctx)> callback CGPU_REQUIRED;
	};

	void graphicsPass(GraphicsPassParams&& params);

	struct ComputePassParams
	{
		/// Required.
		std::function<void(ComputePassContext& ctx)> callback CGPU_REQUIRED;
	};

	void computePass(ComputePassParams&& params);

	struct BLASParams
	{
		struct VertexBuffer
		{
			/// Required.
			BufferPtr buffer CGPU_REQUIRED;
			/// Optional. Default: Default-initialized range.
			BufferRange range{};
		};

		struct IndexBuffer
		{
			/// Required.
			BufferPtr buffer CGPU_REQUIRED;
			/// Optional. Default: Default-initialized range.
			BufferRange range{};
		};

		/// Must be aligned to minAccelerationStructureScratchOffsetAlignment bytes.
		struct ScratchBuffer
		{
			/// Required.
			BufferPtr buffer CGPU_REQUIRED;
			/// Optional. Default: Default-initialized range.
			BufferRange range{};
		};

		/// Required.
		BLASPtr blas CGPU_REQUIRED;
		/// Required.
		VertexBuffer vertex_buffer CGPU_REQUIRED;
		/// Optional. Default: No index buffer.
		std::optional<IndexBuffer> index_buffer{};
		/// Optional. Default: No scratch buffer.
		std::optional<ScratchBuffer> scratch_buffer{};
	};

	void buildBLAS(BLASParams&& params);

	struct TLASParams
	{
		struct Instance
		{
			/// Required.
			BLASPtr blas CGPU_REQUIRED;
			/// Required.
			glm::mat4x3 local_to_world CGPU_REQUIRED;
			/// Optional. Default: 0.
			uint32_t custom_index{0};
			/// Optional. Default: 0xFF.
			uint8_t mask{0xFF};
			/// Optional. Default: 0.
			uint32_t sbt_record_offset{0};
			/// Optional. Default: No flag.
			vk::GeometryInstanceFlagsKHR flags{};
		};

		/// Must have a size of N * sizeof(vk::AccelerationStructureInstanceKHR) structs, where N being the number of instances.
		///
		/// Must be aligned to 16 bytes.
		struct InstancesBuffer
		{
			/// Required.
			BufferPtr buffer CGPU_REQUIRED;
			/// Optional. Default: Default-initialized range.
			BufferRange range{};
		};

		struct InstanceInfo
		{
			/// Required.
			std::span<const Instance> data CGPU_REQUIRED;
			/// Required.
			InstancesBuffer buffer CGPU_REQUIRED;
		};

		/// Must be aligned to minAccelerationStructureScratchOffsetAlignment bytes.
		struct ScratchBuffer
		{
			/// Required.
			BufferPtr buffer CGPU_REQUIRED;
			/// Optional. Default: Default-initialized range.
			BufferRange range{};
		};

		/// Required.
		TLASPtr tlas CGPU_REQUIRED;
		/// Optional. Default: No instance.
		std::optional<InstanceInfo> instance_info{};
		/// Optional. Default: No scratch buffer.
		std::optional<ScratchBuffer> scratch_buffer{};
	};

	//TODO: All BLASes referenced in the build should be kept alive by the TLAS until the next TLAS build
	void buildTLAS(TLASParams&& params);

	struct DebugBarrierParams
	{
		/// Optional. Default: All commands.
		vk::PipelineStageFlags2 src_stages{vk::PipelineStageFlagBits2::eAllCommands};
		/// Optional. Default: All accesses.
		vk::AccessFlags2 src_accesses{vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
		/// Optional. Default: All commands.
		vk::PipelineStageFlags2 dst_stages{vk::PipelineStageFlagBits2::eAllCommands};
		/// Optional. Default: All accesses.
		vk::AccessFlags2 dst_accesses{vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
	};

	void debugBarrier(DebugBarrierParams&& params);

	struct ResolveParams
	{
		struct Range
		{
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectsPixelsRange src{};
			/// Optional. Default: Default-initialized range.
			ImageLevelLayersAspectsPixelsRange dst{};
		};

		static const std::array<Range, 1> DEFAULT_RANGE;

		/// Required.
		ImagePtr src_image CGPU_REQUIRED;
		/// Required.
		ImagePtr dst_image CGPU_REQUIRED;
		/// Optional. Default: One default-initialized range.
		std::span<const Range> ranges{DEFAULT_RANGE};
		/// Optional. Default: SampleZero.
		vk::ResolveModeFlagBits depth_mode{vk::ResolveModeFlagBits::eSampleZero};
		/// Optional. Default: SampleZero.
		vk::ResolveModeFlagBits stencil_mode{vk::ResolveModeFlagBits::eSampleZero};
	};

	void resolve(ResolveParams&& params);

private:
	friend class CommandContext::Slot;
	friend class PassContext;
	friend class GraphicsPassContext;
	friend class ComputePassContext;
	friend class ScopedDebugRegion;

	struct AccessPoints
	{
		vk::PipelineStageFlags2 stages{};
		vk::AccessFlags2 accesses{};

		[[nodiscard]]
		AccessPoints operator|(AccessPoints other) const
		{
			return {stages | other.stages, accesses | other.accesses};
		}

		AccessPoints& operator|=(AccessPoints other)
		{
			stages |= other.stages;
			accesses |= other.accesses;
			return *this;
		}
	};

	struct CmdCallbackBase
	{
		virtual ~CmdCallbackBase() = default;

		virtual void operator()(const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) = 0;
	};

	struct Cmd
	{
		detail::BumpUniquePtr<CmdCallbackBase> callback;

		bool is_stageful{};

		detail::BumpDenseUnorderedMap<detail::Resource*, AccessPoints> referenced_resources;

		explicit Cmd(detail::BumpMemoryResource& bump_memory):
			referenced_resources{detail::BumpAllocator{bump_memory}}
		{}
	};

	struct Containers
	{
		// bool: true if it is a resource and it is written
		detail::BumpSegmentedUnorderedMap<std::shared_ptr<void>, bool> referenced_objects;

		detail::BumpList<Cmd> cmd_list;

		explicit Containers(detail::BumpMemoryResource& bump_memory):
			referenced_objects{detail::BumpAllocator{bump_memory}},
			cmd_list{detail::BumpAllocator{bump_memory}}
		{}
	};

	std::shared_ptr<CommandContext::Slot> m_slot;
	const vk::detail::DispatchLoaderDynamic* m_dispatcher;
	detail::BumpMemoryResource* m_bump_memory;

	QueuePtr m_queue;

	uint32_t m_num_resources{0};
	uint32_t m_num_resource_barriers{0};

	std::optional<Containers> m_containers;

#if !defined(NDEBUG)
	bool m_submitted{false};
#endif

	explicit CommandRecorder(
		std::shared_ptr<CommandContext::Slot>&& slot,
		detail::BumpMemoryResource& bump_memory,
		const QueuePtr& queue
	);

	template<class T>
	requires(!std::is_base_of_v<cgpu::detail::Resource, T>)
	void addReferencedObject(const std::shared_ptr<T>& object)
	{
		m_containers->referenced_objects.try_emplace(object, false);
	}

	template<class T>
	requires(std::is_base_of_v<cgpu::detail::Resource, T>)
	void addCmdResource(const std::shared_ptr<T>& resource, AccessPoints access_point);

	/// If is_stageful is false, the cmd will not be taken into account when deciding between event vs barrier
	template<class TCallback, class... TArgs>
	requires(std::derived_from<TCallback, cgpu::CommandRecorder::CmdCallbackBase>)
	TCallback& addCmd(bool is_stageful, TArgs&&... args);

	vk::DeviceAddress writeParameters(
		const void* data,
		size_t size,
		size_t alignment
	);

	void beginDebugRegion(std::string_view name, glm::vec4 color);

	void endDebugRegion();
};

class ScopedDebugRegion
{
public:
	explicit ScopedDebugRegion(CommandRecorder& rec, std::string_view name, glm::vec4 color = glm::vec4{0.0f});
	~ScopedDebugRegion();

	ScopedDebugRegion(const ScopedDebugRegion&) = delete;
	ScopedDebugRegion(ScopedDebugRegion&&) = delete;

	ScopedDebugRegion& operator=(const ScopedDebugRegion&) = delete;
	ScopedDebugRegion& operator=(ScopedDebugRegion&&) = delete;

private:
	CommandRecorder* m_rec;
};
}
