#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <half.hpp>

namespace cgpu
{
#define CGPU_DECLARE_DESCRIPTOR_HANDLE(name, parent_name) \
	class parent_name;                                    \
	struct name                                           \
	{                                                     \
		uint32_t index{0};                                \
                                                          \
		name() = default;                                 \
                                                          \
	private:                                              \
		friend class parent_name;                         \
                                                          \
		[[maybe_unused]]                                  \
		uint32_t m_padding{};                             \
                                                          \
		name(uint32_t index):                             \
			index{index}                                  \
		{}                                                \
	};

CGPU_DECLARE_DESCRIPTOR_HANDLE(SamplerHandle, Sampler)
CGPU_DECLARE_DESCRIPTOR_HANDLE(SampledImageHandle, Image)
CGPU_DECLARE_DESCRIPTOR_HANDLE(StorageImageHandle, Image)
CGPU_DECLARE_DESCRIPTOR_HANDLE(UniformTexelBufferHandle, Buffer)
CGPU_DECLARE_DESCRIPTOR_HANDLE(StorageTexelBufferHandle, Buffer)

#undef CGPU_DECLARE_DESCRIPTOR_HANDLE

namespace shader_types
{
namespace detail
{
template<class T>
class Scalar
{
public:
	Scalar() = default;

	Scalar(const T& value):
		m_value{value}
	{}

	[[nodiscard]]
	const T& get() const
	{
		return m_value;
	}

private:
	T m_value{};
};

template<class T, size_t N>
class Vector
{
public:
	Vector() = default;

	Vector(const glm::vec<N, T>& value)
	{
		std::memcpy(&m_value, glm::value_ptr(value), sizeof(m_value));
	}

	[[nodiscard]]
	const glm::vec<N, T>& get() const
	{
		glm::vec<N, T> value;
		std::memcpy(glm::value_ptr(value), &m_value, sizeof(m_value));
		return value;
	}

private:
	std::array<T, N> m_value{};
};

template<class T, size_t R, size_t C>
class Matrix
{
public:
	Matrix() = default;

	Matrix(const glm::mat<C, R, T>& value)
	{
		glm::mat<R, C, T> row_major_value = glm::transpose(value);
		std::memcpy(&m_value, glm::value_ptr(row_major_value), sizeof(m_value));
	}

	[[nodiscard]]
	const glm::mat<C, R, T>& get() const
	{
		glm::mat<R, C, T> row_major_value;
		std::memcpy(glm::value_ptr(row_major_value), &m_value, sizeof(m_value));
		return glm::transpose(row_major_value);
	}

private:
	std::array<T, R * C> m_value{};
};
}

// Scalars

using float16_t = half_float::half;
using float32_t = float;
using float64_t = double;

using uint = uint32_t;
using half = float16_t;

// Vectors & matrices

#define CGPU_DECLARE_STRUCTURED_TYPES(scalar)         \
	using scalar##2 = detail::Vector<scalar, 2>;      \
	using scalar##3 = detail::Vector<scalar, 3>;      \
	using scalar##4 = detail::Vector<scalar, 4>;      \
	using scalar##2x1 = detail::Matrix<scalar, 2, 1>; \
	using scalar##3x1 = detail::Matrix<scalar, 3, 1>; \
	using scalar##4x1 = detail::Matrix<scalar, 4, 1>; \
	using scalar##2x2 = detail::Matrix<scalar, 2, 2>; \
	using scalar##3x2 = detail::Matrix<scalar, 3, 2>; \
	using scalar##4x2 = detail::Matrix<scalar, 4, 2>; \
	using scalar##2x3 = detail::Matrix<scalar, 2, 3>; \
	using scalar##3x3 = detail::Matrix<scalar, 3, 3>; \
	using scalar##4x3 = detail::Matrix<scalar, 4, 3>; \
	using scalar##2x4 = detail::Matrix<scalar, 2, 4>; \
	using scalar##3x4 = detail::Matrix<scalar, 3, 4>; \
	using scalar##4x4 = detail::Matrix<scalar, 4, 4>;

CGPU_DECLARE_STRUCTURED_TYPES(int8_t)
CGPU_DECLARE_STRUCTURED_TYPES(int16_t)
CGPU_DECLARE_STRUCTURED_TYPES(int32_t)
CGPU_DECLARE_STRUCTURED_TYPES(int64_t)

CGPU_DECLARE_STRUCTURED_TYPES(uint8_t)
CGPU_DECLARE_STRUCTURED_TYPES(uint16_t)
CGPU_DECLARE_STRUCTURED_TYPES(uint32_t)
CGPU_DECLARE_STRUCTURED_TYPES(uint64_t)

CGPU_DECLARE_STRUCTURED_TYPES(float16_t)
CGPU_DECLARE_STRUCTURED_TYPES(float32_t)
CGPU_DECLARE_STRUCTURED_TYPES(float64_t)

#undef CGPU_DECLARE_STRUCTURED_TYPES

#define CGPU_DECLARE_STRUCTURED_ALIAS(alias, scalar) \
	using alias##2 = scalar##2;                      \
	using alias##3 = scalar##3;                      \
	using alias##4 = scalar##4;                      \
	using alias##2x1 = scalar##2x1;                  \
	using alias##3x1 = scalar##3x1;                  \
	using alias##4x1 = scalar##4x1;                  \
	using alias##2x2 = scalar##2x2;                  \
	using alias##3x2 = scalar##3x2;                  \
	using alias##4x2 = scalar##4x2;                  \
	using alias##2x3 = scalar##2x3;                  \
	using alias##3x3 = scalar##3x3;                  \
	using alias##4x3 = scalar##4x3;                  \
	using alias##2x4 = scalar##2x4;                  \
	using alias##3x4 = scalar##3x4;                  \
	using alias##4x4 = scalar##4x4;

CGPU_DECLARE_STRUCTURED_ALIAS(int, int32_t)
CGPU_DECLARE_STRUCTURED_ALIAS(uint, uint32_t)
CGPU_DECLARE_STRUCTURED_ALIAS(half, float16_t)
CGPU_DECLARE_STRUCTURED_ALIAS(float, float32_t)
CGPU_DECLARE_STRUCTURED_ALIAS(double, float64_t)

#undef CGPU_DECLARE_STRUCTURED_TYPES

// Descriptor handles

using SamplerState = cgpu::SamplerHandle;

#define CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(descriptor_handle_type, name) \
	template<class = float4, int = 0, int = 0>                              \
	struct name                                                             \
	{                                                                       \
		using Handle = descriptor_handle_type;                              \
	};

CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture1D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTexture1D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTexture1D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture1DArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTexture1DArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTexture1DArray)

CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture2D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture2DMS)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTexture2D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTexture2D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture2DArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture2DMSArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTexture2DArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTexture2DArray)

CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, TextureCube)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTextureCube)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTextureCube)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, TextureCubeArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTextureCubeArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTextureCubeArray)

CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture3D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTexture3D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTexture3D)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::SampledImageHandle, Texture3DArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, RWTexture3DArray)
CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS(cgpu::StorageImageHandle, WTexture3DArray)

#undef CGPU_DECLARE_TEXTURE_DESCRIPTOR_ALIAS

#define CGPU_DECLARE_BUFFER_DESCRIPTOR_ALIAS(descriptor_handle_type, name) \
	template<class = float4, int = 0>                                      \
	struct name                                                            \
	{                                                                      \
		using Handle = descriptor_handle_type;                             \
	};

CGPU_DECLARE_BUFFER_DESCRIPTOR_ALIAS(cgpu::UniformTexelBufferHandle, Buffer)
CGPU_DECLARE_BUFFER_DESCRIPTOR_ALIAS(cgpu::StorageTexelBufferHandle, RWBuffer)

#undef CGPU_DECLARE_BUFFER_DESCRIPTOR_ALIAS
}
}
