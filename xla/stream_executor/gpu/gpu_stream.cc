/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/gpu/gpu_stream.h"

#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace gpu {


Stream::PlatformSpecificHandle GpuStream::platform_specific_handle() const {
  PlatformSpecificHandle handle;
  handle.stream = gpu_stream_;
  return handle;
}

GpuStream::~GpuStream() {
  GpuDriver::DestroyStream(parent_->gpu_context(), gpu_stream_);
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
GpuStream::CreateEventBasedTimer(bool use_delay_kernel) {
  return parent_->CreateEventBasedTimer(this, use_delay_kernel);
}

absl::Status GpuStream::Launch(const ThreadDim& thread_dims,
                               const BlockDim& block_dims, const Kernel& kernel,
                               const KernelArgs& args) {
  return Launch(thread_dims, block_dims, std::nullopt, kernel, args);
}

absl::Status GpuStream::Launch(const ThreadDim& thread_dims,
                               const BlockDim& block_dims,
                               const ClusterDim& cluster_dims,
                               const Kernel& kernel, const KernelArgs& args) {
  return Launch(thread_dims, block_dims, std::make_optional(cluster_dims),
                kernel, args);
}

GpuStream* AsGpuStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return static_cast<GpuStream*>(stream);
}

GpuStreamHandle AsGpuStreamValue(Stream* stream) {
  DCHECK(stream != nullptr);
  return AsGpuStream(stream)->gpu_stream();
}

}  // namespace gpu
}  // namespace stream_executor
