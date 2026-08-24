#pragma once

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContext.hpp>
#include <CyphGPU/detail/BumpAllocator.hpp>
#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Utils.hpp>

#include <boost/container/static_vector.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <variant>

namespace cgpu
{
class ScopedDebugRegion;

class CommandRecorder
{
public:
	template<class T>
	struct Req
	{
		T value;

		Req() = delete;

		template<class... TArgs>
		requires(std::constructible_from<T, TArgs...>)
		Req(TArgs&&... args):
			value{std::forward<TArgs>(args)...}
		{}

		Req(const T& value):
			value{value}
		{}

		Req(T&& value):
			value{std::move(value)}
		{}

		T& operator*()
		{
			return value;
		}

		std::add_const_t<T>& operator*() const
		{
			return value;
		}

		T* operator->()
		{
			return std::addressof(value);
		}

		std::add_const_t<T>* operator->() const
		{
			return std::addressof(value);
		}
	};

	template<class T>
	using Opt = std::optional<T>;

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
		/// Default: All levels.
		Opt<Range<uint32_t>> levels{};
		/// Default: All layers.
		Opt<Range<uint32_t>> layers{};
	};

	struct ImageLevelLayersAspectsPixelsRange
	{
		/// Default: Level 0.
		Opt<uint32_t> level{};
		/// Default: All layers.
		Opt<Range<uint32_t>> layers{};
		/// Default: All aspects.
		Opt<vk::ImageAspectFlags> aspects{};
		/// Default: All pixels.
		Opt<Range<glm::uvec3>> pixels{};
	};

	struct ImageLevelLayersAspectsRectRange
	{
		/// Default: Level 0.
		Opt<uint32_t> level{};
		/// Default: All layers.
		Opt<Range<uint32_t>> layers{};
		/// Default: All aspects.
		Opt<vk::ImageAspectFlags> aspects{};
		/// Default: [0, 0, 0].
		Opt<glm::uvec3> top_left{};
		/// Default: Image extent.
		Opt<glm::uvec3> bottom_right{};
	};

	struct BufferRange
	{
		/// Default: All bytes.
		Opt<Range<vk::DeviceSize>> byte_range{};
	};

	// ----- Commands -----

	struct ClearImageParams
	{
		Req<ImagePtr> image;
		/// Default: One default-initialized range.
		Opt<std::vector<ImageLevelsLayersRange>> ranges{};
		/// Default: Empty (no color clear).
		Opt<std::variant<glm::vec4, glm::ivec4, glm::uvec4>> color_value{};
		/// Default: Empty (no depth clear).
		Opt<float> depth_value{};
		/// Default: Empty (no stencil clear).
		Opt<uint32_t> stencil_value{};
	};

	void clearImage(ClearImageParams&& params);

	struct CopyImageToImageParams
	{
		struct Range
		{
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsPixelsRange> src{};
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsPixelsRange> dst{};
		};

		Req<ImagePtr> src_image;
		Req<ImagePtr> dst_image;
		/// Default: One default-initialized range.
		Opt<std::vector<Range>> ranges{};
	};

	void copyImageToImage(CopyImageToImageParams&& params);

	struct CopyBufferToImageParams
	{
		struct Range
		{
			/// Default: Default-initialized range.
			Opt<BufferRange> src{};
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsPixelsRange> dst{};
		};

		Req<BufferPtr> src_buffer;
		Req<ImagePtr> dst_image;
		/// Default: One default-initialized range.
		Opt<std::vector<Range>> ranges{};
	};

	void copyBufferToImage(CopyBufferToImageParams&& params);

	struct CopyImageToBufferParams
	{
		struct Range
		{
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsPixelsRange> src{};
			/// Default: Default-initialized range.
			Opt<BufferRange> dst{};
		};

		Req<ImagePtr> src_image;
		Req<BufferPtr> dst_buffer;
		/// Default: One default-initialized range.
		Opt<std::vector<Range>> ranges{};
	};

	void copyImageToBuffer(CopyImageToBufferParams&& params);

	struct CopyBufferToBufferParams
	{
		struct Range
		{
			/// Default: Default-initialized range.
			Opt<BufferRange> src{};
			/// Default: Default-initialized range.
			Opt<BufferRange> dst{};
		};

		Req<BufferPtr> src_buffer;
		Req<BufferPtr> dst_buffer;
		/// Default: One default-initialized range.
		Opt<std::vector<Range>> ranges{};
	};

	void copyBufferToBuffer(CopyBufferToBufferParams&& params);

	struct BlitParams
	{
		struct Range
		{
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsRectRange> src{};
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsRectRange> dst{};
		};

		Req<ImagePtr> src_image;
		Req<ImagePtr> dst_image;
		/// Default: Nearest.
		Opt<vk::Filter> filter{};
		/// Default: One default-initialized range.
		Opt<std::vector<Range>> ranges{};
	};

	void blit(BlitParams&& params);

	struct GraphicsPassParams
	{
		struct ColorAttachment
		{
			struct Resolve
			{
				Req<ImagePtr> image;
				/// Default: Average.
				Opt<vk::ResolveModeFlagBits> mode{};
				/// Default: Level 0.
				Opt<uint32_t> level{};
				/// Default: Layer 0.
				Opt<uint32_t> first_layer{};
			};

			Req<ImagePtr> image;
			/// Default: Image format.
			Opt<vk::Format> format{};
			/// Default: Level 0.
			Opt<uint32_t> level{};
			/// Default: Layer 0.
			Opt<uint32_t> first_layer{};
			Req<vk::AttachmentLoadOp> load_op;
			Req<vk::AttachmentStoreOp> store_op;
			/// Default: Empty. Must be set if load_op == eClear.
			Opt<std::variant<glm::vec4, glm::ivec4, glm::uvec4>> clear_color_value{};
			/// Default: No resolve.
			Opt<Resolve> resolve{};
		};

		struct DepthStencilAttachment
		{
			struct Resolve
			{
				Req<ImagePtr> image;
				/// Default: SampleZero. Only used when depth is enabled.
				Opt<vk::ResolveModeFlagBits> depth_mode{};
				/// Default: SampleZero. Only used when stencil is enabled.
				Opt<vk::ResolveModeFlagBits> stencil_mode{};
				/// Default: Level 0.
				Opt<uint32_t> level{};
				/// Default: Layer 0.
				Opt<uint32_t> first_layer{};
			};

			Req<ImagePtr> image;
			/// Default: Level 0.
			Opt<uint32_t> level{};
			/// Defaul: Layer 0.
			Opt<uint32_t> first_layer{};
			/// Default: True if the format has a depth aspect.
			Opt<bool> enable_depth{};
			/// Default: True if the format has a stencil aspect.
			Opt<bool> enable_stencil{};
			Req<vk::AttachmentLoadOp> load_op;
			Req<vk::AttachmentStoreOp> store_op;
			/// Default: Empty. Must be set if load_op == eClear and depth is enabled.
			Opt<float> clear_depth_value{};
			/// Default: Empty. Must be set if load_op == eClear and stencil is enabled.
			Opt<uint32_t> clear_stencil_value{};
			/// Default: No resolve.
			Opt<Resolve> resolve{};
		};

		struct LayerCount
		{
			Req<uint32_t> value;
		};

		struct MultiviewMask
		{
			Req<uint32_t> value;
		};

		/// Default: Attachment images extent.
		///
		/// Must be set if there is no attachment or if attachments have different extents.
		Opt<Range<glm::uvec2>> render_area{};
		/// Default: Single layer.
		Opt<std::variant<LayerCount, MultiviewMask>> layer_mode{};
		/// Default: No color attachment.
		Opt<boost::container::static_vector<ColorAttachment, 8>> color_attachments{};
		/// Default: No depth-stencil attachment.
		Opt<DepthStencilAttachment> depth_stencil_attachment{};
		Req<std::function<void(GraphicsPassContext& ctx)>> callback;
	};

	void graphicsPass(GraphicsPassParams&& params);

	struct ComputePassParams
	{
		Req<std::function<void(ComputePassContext& ctx)>> callback;
	};

	void computePass(ComputePassParams&& params);

	struct BLASParams
	{
		struct VertexBuffer
		{
			Req<BufferPtr> buffer;
			/// Default: Default-initialized range.
			Opt<BufferRange> range{};
		};

		struct IndexBuffer
		{
			Req<BufferPtr> buffer;
			/// Default: Default-initialized range.
			Opt<BufferRange> range{};
		};

		/// Must be aligned to minAccelerationStructureScratchOffsetAlignment bytes.
		struct ScratchBuffer
		{
			Req<BufferPtr> buffer;
			/// Default: Default-initialized range.
			Opt<BufferRange> range{};
		};

		Req<BLASPtr> blas;
		Req<VertexBuffer> vertex_buffer;
		/// Default: No index buffer.
		Opt<IndexBuffer> index_buffer{};
		/// Default: No scratch buffer.
		Opt<ScratchBuffer> scratch_buffer{};
	};

	void buildBLAS(BLASParams&& params);

	struct TLASParams
	{
		struct Instance
		{
			Req<BLASPtr> blas;
			Req<glm::mat4x3> local_to_world;
			/// Default: 0.
			Opt<uint32_t> custom_index{};
			/// Default: 0xFF.
			Opt<uint8_t> mask{};
			/// Default: 0.
			Opt<uint32_t> sbt_record_offset{};
			/// Default: 0.
			Opt<vk::GeometryInstanceFlagsKHR> flags{};
		};

		/// Must have a size of N * sizeof(vk::AccelerationStructureInstanceKHR) structs, where N being the number of instances.
		///
		/// Must be aligned to 16 bytes.
		struct InstancesBuffer
		{
			Req<BufferPtr> buffer;
			/// Default: Default-initialized range.
			Opt<BufferRange> range{};
		};

		struct InstanceInfo
		{
			Req<std::vector<Instance>> data;
			Req<InstancesBuffer> buffer;
		};

		/// Must be aligned to minAccelerationStructureScratchOffsetAlignment bytes.
		struct ScratchBuffer
		{
			Req<BufferPtr> buffer;
			/// Default: Default-initialized range.
			Opt<BufferRange> range{};
		};

		Req<TLASPtr> tlas;
		/// Default: No instance.
		Opt<InstanceInfo> instance_info;
		/// Default: No scratch buffer.
		Opt<ScratchBuffer> scratch_buffer{};
	};

	void buildTLAS(TLASParams&& params);

	struct DebugBarrierParams
	{
		/// Default: All commands.
		Opt<vk::PipelineStageFlags2> src_stages;
		/// Default: All accesses.
		Opt<vk::AccessFlags2> src_accesses;
		/// Default: All commands.
		Opt<vk::PipelineStageFlags2> dst_stages;
		/// Default: All accesses.
		Opt<vk::AccessFlags2> dst_accesses;
	};

	void debugBarrier(DebugBarrierParams&& params);

	struct ResolveParams
	{
		struct Range
		{
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsPixelsRange> src{};
			/// Default: Default-initialized range.
			Opt<ImageLevelLayersAspectsPixelsRange> dst{};
		};

		Req<ImagePtr> src_image;
		Req<ImagePtr> dst_image;
		/// Default: One default-initialized range.
		Opt<std::vector<Range>> ranges{};
		/// Default: Average.
		Opt<vk::ResolveModeFlagBits> color_mode{};
		/// Default: SampleZero.
		Opt<vk::ResolveModeFlagBits> depth_mode{};
		/// Default: SampleZero.
		Opt<vk::ResolveModeFlagBits> stencil_mode{};
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

	struct Barrier
	{
		AccessPoints src;
		AccessPoints dst;
	};

	struct Event
	{
		vk::Event vk_event;
		detail::BumpVector<vk::ImageMemoryBarrier2> vk_image_barriers;
		detail::BumpVector<vk::MemoryRangeBarrierKHR> vk_buffer_barriers;
		vk::StructureChain<vk::DependencyInfo, vk::MemoryRangeBarriersInfoKHR> vk_chain;
		vk::PipelineStageFlags2 reset_stages;

		explicit Event(detail::BumpMemoryResource& bump_memory):
			vk_image_barriers{detail::BumpAllocator{bump_memory}},
			vk_buffer_barriers{detail::BumpAllocator{bump_memory}}
		{}
	};

	struct SignalPoint
	{
		detail::BumpFlatMap<Image*, Barrier> image_barriers;
		detail::BumpFlatMap<Buffer*, Barrier> buffer_barriers;

		// Storage for submit-time recording
		detail::BumpUniquePtr<Event> event; // Nullable, nullptr = empty

		explicit SignalPoint(detail::BumpMemoryResource& bump_memory):
			image_barriers{detail::BumpAllocator{bump_memory}},
			buffer_barriers{detail::BumpAllocator{bump_memory}}
		{}
	};

	struct CmdBase
	{
		virtual ~CmdBase() = default;

		virtual void execute(const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) = 0;

		detail::BumpFlatMap<Image*, AccessPoints> referenced_images;
		detail::BumpFlatMap<Buffer*, AccessPoints> referenced_buffers;

		uint64_t stageful_index{};

		// Storage for submit-time recording
		detail::BumpFlatSet<CmdBase*> wait_cmds;
		std::optional<SignalPoint> signal_point;

		explicit CmdBase(detail::BumpMemoryResource& bump_memory):
			referenced_images{detail::BumpAllocator{bump_memory}},
			referenced_buffers{detail::BumpAllocator{bump_memory}},
			wait_cmds{detail::BumpAllocator{bump_memory}}
		{}
	};

	struct SyncPoint
	{
		CmdBase* cmd;
		AccessPoints access_points;
	};

	struct ResourceAccess
	{
		std::optional<SyncPoint> last_write;
		std::optional<SyncPoint> reads_since_last_write;
	};

	struct ReferencedContainers
	{
		detail::BumpSegmentedUnorderedSet<std::shared_ptr<void>> objects;

		detail::BumpList<detail::BumpUniquePtr<CmdBase>> cmd_list;

		explicit ReferencedContainers(detail::BumpMemoryResource& bump_memory):
			objects{detail::BumpAllocator{bump_memory}},
			cmd_list{detail::BumpAllocator{bump_memory}}
		{}
	};

	std::shared_ptr<CommandContext::Slot> m_slot;
	const vk::detail::DispatchLoaderDynamic* m_dispatcher;
	detail::BumpMemoryResource* m_bump_memory;

	QueuePtr m_queue;

	std::optional<ReferencedContainers> m_referenced_containers;

	uint64_t m_next_stageful_index{0};

#if !defined(NDEBUG)
	bool m_submitted{false};
#endif

	explicit CommandRecorder(
		std::shared_ptr<CommandContext::Slot>&& slot,
		detail::BumpMemoryResource& bump_memory,
		const QueuePtr& queue
	);

	template<class T>
	requires(!std::derived_from<T, cgpu::Resource>)
	void addReferencedObject(const std::shared_ptr<T>& object)
	{
		m_referenced_containers->objects.emplace(object);
	}

	template<class T>
	requires(std::derived_from<T, cgpu::Resource>)
	void addCmdResource(const std::shared_ptr<T>& resource, AccessPoints access_point);

	/// If is_stageful is false, the cmd will not be taken into account when deciding between event vs barrier
	template<class T, class... TArgs>
	requires(std::derived_from<T, cgpu::CommandRecorder::CmdBase>)
	T& addCmd(bool is_stageful, TArgs&&... args);

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
