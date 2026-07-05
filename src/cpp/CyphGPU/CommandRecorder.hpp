#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Utils.hpp>

#include <glm/glm.hpp>
#include <variant>

namespace cgpu
{
class CommandContextSlot;

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

		const T& operator*() const
		{
			return value;
		}

		T* operator->()
		{
			return std::addressof(value);
		}

		const T* operator->() const
		{
			return std::addressof(value);
		}
	};

	template<class T>
	using Opt = std::optional<T>;

	CommandRecorder(const CommandRecorder&) = delete;
	CommandRecorder(CommandRecorder&&) = delete;

	CommandRecorder& operator=(const CommandRecorder&) = delete;
	CommandRecorder& operator=(CommandRecorder&&) = delete;

	void submit();

	// ----- Commands -----

	struct ImageLevelLayerRange
	{
		/// Default: All levels.
		Opt<cgpu::Range<uint32_t>> levels{};
		/// Default: All layers.
		Opt<cgpu::Range<uint32_t>> layers{};
	};

	struct ClearColorImageParams
	{
		Req<ImagePtr> image;
		/// Default: One default-initialized range.
		Opt<std::vector<ImageLevelLayerRange>> ranges{};
		Req<std::variant<glm::vec4, glm::ivec4, glm::uvec4>> clear_value;
	};

	void clearColorImage(const ClearColorImageParams& params);

private:
	friend class CommandContextSlot;

	std::shared_ptr<CommandContextSlot> m_slot;
	const vk::detail::DispatchLoaderDynamic* m_dispatcher;

	cgpu::QueuePtr m_queue;
	vk::CommandBuffer m_cmdbuf;

	std::vector<std::shared_ptr<void>> m_referenced_objects;
	std::vector<ImagePtr> m_referenced_images;
	std::vector<BufferPtr> m_referenced_buffers;

	std::vector<vk::Semaphore> m_wait_semaphores;
	std::vector<uint64_t> m_wait_values;

#if !defined(NDEBUG)
	bool m_submitted{false};
#endif

	explicit CommandRecorder(
		std::shared_ptr<CommandContextSlot>&& slot,
		const cgpu::QueuePtr& queue,
		vk::CommandBuffer cmdbuf
	);

	template<class T>
	void addReferencedObject(const std::shared_ptr<T>& object);
};
}
