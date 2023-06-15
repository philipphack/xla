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

#include "xla/service/collective_permute_decomposer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/graphcycles/graphcycles.h"

namespace xla {

namespace impl {
// Returns true if the (source, target) pairs relationship has a cycle.
//
bool hasCycles(const std::vector<std::pair<int64_t, int64_t>>& pairs) {
  // Build a direct graph to check for cycles in (source, target) relationship.
  tensorflow::GraphCycles graph;

  // Map replica numbers to graph node ids.
  absl::flat_hash_map<int64_t, int32_t> replica_to_node_id;
  auto get_node_id = [&](int64_t replica) {
    auto it_and_inserted = replica_to_node_id.emplace(replica, -1);
    auto it = it_and_inserted.first;
    auto inserted = it_and_inserted.second;
    if (inserted) {
      // First time to see the replica, create a node for it.
      it->second = graph.NewNode();
    }
    return it->second;
  };

  for (auto pair : pairs) {
    auto source = get_node_id(pair.first);
    auto target = get_node_id(pair.second);
    VLOG(3) << "See source target " << source << " " << target;
    if (!graph.InsertEdge(source, target)) {
      VLOG(3) << "Detect cycles";
      return true;
    }
  }
  return false;
}

// Returns true if the CollectivePermuteStart instruction should be transformed
// send/recv. We currently limit the transformation to is asynchronous
// CollectivePermuteStart without any cycle in (source, target) relationship,
// with only one input and with a result type the same as the input type.
bool ShouldDecompose(
    const HloCollectivePermuteInstruction& collective_permute) {
  VLOG(3) << collective_permute.ToString();
  auto backend_config =
      collective_permute.backend_config<xla::gpu::CollectiveBackendConfig>()
          .value();
  if (backend_config.is_sync()) return false;
  if (collective_permute.operand_count() != 1 ||
      collective_permute.shape() != collective_permute.operand(0)->shape())
    return false;
  return !hasCycles(collective_permute.source_target_pairs());
}

Status DecomposeCollectivePermute(
    HloCollectivePermuteInstruction* collective_permute,
    HloComputation* computation) {
  // The HLO verifier ensures that CollectivePermuteStart's single user is
  // CollectivePermuteDone.
  HloInstruction* collective_permute_done = collective_permute->users().front();
  auto optional_channel_id = collective_permute->channel_id();
  int64_t channel_id =
      optional_channel_id.has_value() ? optional_channel_id.value() : 0;
  HloInstruction* data = collective_permute->mutable_operand(0);
  const Shape& data_shape = data->shape();

  xla::FrontendAttributes attributes;
  std::string source_target_pairs_string =
      "{" +
      absl::StrJoin(collective_permute->source_target_pairs(), ",",
                    absl::PairFormatter(
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, "{", value);
                        },
                        ",",
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, value, "}");
                        })) +
      "}";

  (*attributes.mutable_map())[kSendRecvSourceTargetPairsAttr] =
      source_target_pairs_string;

  HloInstruction* after_all =
      computation->AddInstruction(HloInstruction::CreateToken());
  HloInstruction* recv = computation->AddInstruction(
      HloInstruction::CreateRecv(data_shape, after_all, channel_id));
  recv->set_frontend_attributes(attributes);

  HloInstruction* send = computation->AddInstruction(
      HloInstruction::CreateSend(data, after_all, channel_id));
  send->set_frontend_attributes(attributes);
  TF_RETURN_IF_ERROR(recv->AddControlDependencyTo(send));

  HloInstruction* recv_done = computation->AddInstruction(
      HloInstruction::CreateRecvDone(recv, channel_id));
  HloInstruction* send_done = computation->AddInstruction(
      HloInstruction::CreateSendDone(send, channel_id));

  HloInstruction* recv_data = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(recv_done, 0));
  TF_RETURN_IF_ERROR(recv_done->AddControlDependencyTo(send_done));
  TF_RETURN_IF_ERROR(collective_permute_done->ReplaceAllUsesWith(recv_data));
  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(collective_permute_done));
  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(collective_permute));

  return OkStatus();
}
}  // namespace impl

StatusOr<bool> CollectivePermuteDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto comp : module->computations(execution_threads)) {
    for (auto hlo : comp->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kCollectivePermuteStart) {
        continue;
      }
      auto collective_permute = Cast<HloCollectivePermuteInstruction>(hlo);
      if (impl::ShouldDecompose(*collective_permute)) {
        TF_RETURN_IF_ERROR(
            impl::DecomposeCollectivePermute(collective_permute, comp));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
