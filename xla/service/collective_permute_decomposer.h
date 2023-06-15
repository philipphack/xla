/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_COLLECTIVE_PERMUTE_DECOMPOSER_H_
#define XLA_SERVICE_COLLECTIVE_PERMUTE_DECOMPOSER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// CollectivePermuteDecomposer is a pass that converts asynchronous
// CollectivePermuteStart operations without any cycle in (source, target)
// relationship to send/recv. We currently restrict this transformation to
// CollectivePermuteStart with one input and with a result type the same as the
// input type.
//
// before transformation:
//     start = <result-type> collective-permute-start(data),
//       source_target_pairs={...}
//     done = <result-type> collective-permute-done(start)
//
// after transformation:
//    after-all = token[] after-all()
//    recv = (u32[2], token[]) recv(after-all), channel_id=0,
//     frontend_attributes={_xla_send_recv_source_target_pairs="{...}"}
//    send = (u32[2], token[]) send(data, after-all), channel_id=0,
//      control-predecessors={recv}, frontend_attributes={
//      _xla_send_recv_source_target_pairs="{...}"}
//    recv-done = (u32[2], token[]) recv-done(recv), channel_id=0
//    done = u32[2] get-tuple-element(recv-done), index=0
//    send-done = token[] send-done(send), channel_id=0,
//      control-predecessors={recv}
//
class CollectivePermuteDecomposer : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "collective_permute_decomposer";
  }

  using HloPassInterface::Run;
  // Runs CollectivePermuteDecomposer pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_PERMUTE_DECOMPOSER_H_
