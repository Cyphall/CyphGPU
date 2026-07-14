#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Resource.hpp>
#include <CyphGPU/Utils.hpp>

#include <glm/glm.hpp>
#include <map>
#include <variant>

namespace cgpu
{
class CommandContextSlot;
class PassContext;
class GraphicsPassContext;
class ComputePassContext;

class CommandRecorder
{
public:
	template<class T>
	struct Req
	{
		T value;

		template<class... TArgs>
		requires(std::constructible_from<T, TArgs...>)
		Req(TArgs&&... args):
			value{std::forward<TArgs>(args)...}
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

	enum class ResourceAccess : uint8_t
	{
		eReadonly,
		eReadWrite,
	};

	CommandRecorder(const CommandRecorder&) = delete;
	CommandRecorder(CommandRecorder&&) = delete;

	CommandRecorder& operator=(const CommandRecorder&) = delete;
	CommandRecorder& operator=(CommandRecorder&&) = delete;

	void submit();

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

	void clearImage(const ClearImageParams& params);

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

	void copyImageToImage(const CopyImageToImageParams& params);

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

	void copyBufferToImage(const CopyBufferToImageParams& params);

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

	void copyImageToBuffer(const CopyImageToBufferParams& params);

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

	void copyBufferToBuffer(const CopyBufferToBufferParams& params);

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

	void blit(const BlitParams& params);

	struct BarrierParams
	{
		Req<vk::PipelineStageFlags2> src_stages;
		Req<vk::AccessFlags2> src_accesses;
		Req<vk::PipelineStageFlags2> dst_stages;
		Req<vk::AccessFlags2> dst_accesses;
	};

	void barrier(const BarrierParams& params);

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
				/// Default: SampleZero.
				Opt<vk::ResolveModeFlagBits> mode{};
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
		Opt<std::vector<ColorAttachment>> color_attachments{};
		/// Default: No depth-stencil attachment.
		Opt<DepthStencilAttachment> depth_stencil_attachment{};
		Req<std::function<void(GraphicsPassContext& ctx)>> callback;
	};

	void graphicsPass(const GraphicsPassParams& params);

	struct ComputePassParams
	{
		Req<std::function<void(ComputePassContext& ctx)>> callback;
	};

	void computePass(const ComputePassParams& params);

private:
	friend class CommandContextSlot;
	friend class PassContext;
	friend class GraphicsPassContext;
	friend class ComputePassContext;

	std::shared_ptr<CommandContextSlot> m_slot;
	const vk::detail::DispatchLoaderDynamic* m_dispatcher;

	vk::DeviceSize m_minUniformBufferAlignment;

	QueuePtr m_queue;
	vk::CommandBuffer m_cmdbuf;

	std::vector<std::shared_ptr<void>> m_referenced_objects;
	std::map<std::shared_ptr<Resource>, ResourceAccess> m_referenced_resources;

#if !defined(NDEBUG)
	bool m_submitted{false};
#endif

	explicit CommandRecorder(
		std::shared_ptr<CommandContextSlot>&& slot,
		const QueuePtr& queue,
		vk::CommandBuffer cmdbuf
	);

	template<class T>
	requires(!std::derived_from<T, cgpu::Resource>)
	void addReferencedObject(const std::shared_ptr<T>& object);

	void addReferencedObject(const std::shared_ptr<Resource>& resource, ResourceAccess access);

	// ----- Pass sub-commands -----

	void bindPipelineStates(
		const VertexInputStatePtr& vertex_input_state,
		const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
		const FragmentShaderStatePtr& fragment_shader_state,
		const FragmentOutputStatePtr& fragment_output_state
	);

	void bindPipelineStates(
		const ComputeShaderStatePtr& compute_shader_state
	);

	void pushParameters(
		uint32_t slot,
		const void* data,
		size_t size,
		size_t alignment
	);

	void draw(
		uint32_t vertex_count,
		uint32_t instance_count,
		uint32_t first_vertex,
		uint32_t first_instance
	);

	void drawIndexed(
		uint32_t index_count,
		uint32_t instance_count,
		uint32_t first_index,
		int32_t vertex_offset,
		uint32_t first_instance
	);

	void setViewport(
		const vk::Viewport& viewport
	);

	void setScissor(
		const vk::Rect2D& scissor
	);

	void dispatch(
		const glm::uvec3& group_count
	);
};
}
