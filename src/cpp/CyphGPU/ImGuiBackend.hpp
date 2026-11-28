#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Image.hpp>

#include <imgui.h>

// NOLINTBEGIN(*-identifier-naming)
void ImGui_ImplCyphGPU_Init(const cgpu::DeviceSessionPtr& device_session, vk::Format output_image_format);
void ImGui_ImplCyphGPU_NewFrame();
void ImGui_ImplCyphGPU_RenderDrawData(const ImDrawData& draw_data, cgpu::CommandRecorder& cmd_rec, const cgpu::ImagePtr& output_image);
void ImGui_ImplCyphGPU_Shutdown();

ImTextureID ImGui_ImplCyphGPU_ToTextureID(const cgpu::ImagePtr& image, const cgpu::Image::SampledDescriptorOverrides& overrides = {});
// NOLINTEND(*-identifier-naming)
