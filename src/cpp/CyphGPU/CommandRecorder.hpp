#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Resource.hpp>
#include <CyphGPU/Utils.hpp>

#include <map>
#include <glm/glm.hpp>
#include <variant>

namespace cgpu
{
class CommandContextSlot;
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

	enum class ResourceAccess
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
		Opt<cgpu::Range<uint32_t>> levels{};
		/// Default: All layers.
		Opt<cgpu::Range<uint32_t>> layers{};
	};

	struct ImageLevelLayersAspectsPixelsRange
	{
		/// Default: Level 0.
		Opt<uint32_t> level{};
		/// Default: All layers.
		Opt<cgpu::Range<uint32_t>> layers{};
		/// Default: All aspects.
		Opt<vk::ImageAspectFlags> aspects{};
		/// Default: All pixels.
		Opt<cgpu::Range<glm::uvec3>> pixels{};
	};

	struct BufferRange
	{
		/// Default: All bytes.
		Opt<cgpu::Range<vk::DeviceSize>> byte_range{};
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

		Req<ImagePtr> srcImage;
		Req<ImagePtr> dstImage;
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

		Req<BufferPtr> srcBuffer;
		Req<ImagePtr> dstImage;
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

		Req<ImagePtr> srcImage;
		Req<BufferPtr> dstBuffer;
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

		Req<BufferPtr> srcBuffer;
		Req<BufferPtr> dstBuffer;
		/// Default: One default-initialized range.
		Opt<std::vector<Range>> ranges{};
	};

	void copyBufferToBuffer(const CopyBufferToBufferParams& params);

	struct BarrierParams
	{
		Req<vk::PipelineStageFlags2> src_stages;
		Req<vk::AccessFlags2> src_accesses;
		Req<vk::PipelineStageFlags2> dst_stages;
		Req<vk::AccessFlags2> dst_accesses;
	};

	void barrier(const BarrierParams& params);

	struct ComputePassParams
	{
		using Callback = void(ComputePassContext& ctx);
		Req<Callback*> callback;
	};

	void computePass(const ComputePassParams& params);

private:
	friend class CommandContextSlot;
	friend class ComputePassContext;

	std::shared_ptr<CommandContextSlot> m_slot;
	const vk::detail::DispatchLoaderDynamic* m_dispatcher;

	cgpu::QueuePtr m_queue;
	vk::CommandBuffer m_cmdbuf;

	std::vector<std::shared_ptr<void>> m_referenced_objects;
	std::map<std::shared_ptr<Resource>, ResourceAccess> m_referenced_resources;

#if !defined(NDEBUG)
	bool m_submitted{false};
#endif

	explicit CommandRecorder(
		std::shared_ptr<CommandContextSlot>&& slot,
		const cgpu::QueuePtr& queue,
		vk::CommandBuffer cmdbuf
	);

	template<class T>
	requires(!std::derived_from<T, cgpu::Resource>)
	void addReferencedObject(const std::shared_ptr<T>& object);

	void addReferencedObject(const std::shared_ptr<Resource>& resource, ResourceAccess access);

	// ----- Pass sub-commands -----

	void bindPipelineStates(
		const cgpu::ComputeShaderStatePtr& compute_shader_state
	);

	void pushParameters(
		uint32_t slot,
		const void* data,
		size_t size,
		size_t alignment
	);

	void dispatch(
		const glm::uvec3& group_count
	);
};
}
