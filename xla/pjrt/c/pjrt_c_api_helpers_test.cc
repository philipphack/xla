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
#include "xla/pjrt/c/pjrt_c_api_helpers.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"

namespace pjrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

TEST(PjRtCApiHelperTest, ConvertValidPjRtValueType) {
  std::vector<int64_t> int64_list = {static_cast<int64_t>(1),
                                     static_cast<int64_t>(2)};
  absl::flat_hash_map<std::string, xla::PjRtValueType> original_cpp_map = {
      {"string", "v1"},
      {"int64", static_cast<int64_t>(1)},
      {"int64_list", int64_list},
      {"float", static_cast<float>(1.0)}};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_map,
                          ConvertToPjRtNamedValueList(original_cpp_map));
  auto converted_back_cpp_map =
      ConvertFromPjRtNamedValueList(c_map.data(), c_map.size());

  EXPECT_THAT(converted_back_cpp_map,
              testing::UnorderedElementsAreArray(original_cpp_map));
}

TEST(PjRtCApiHelperTest, ValidOptionNameAndPjRtValueTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> valid_map = {
      {"string", "v1"}, {"int64", static_cast<int64_t>(1)}};

  TF_EXPECT_OK(ValidateCreateOptions(valid_map, expected));
}

TEST(PjRtCApiHelperTest, InvalidOptionName) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"invalid", "v1"}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, tsl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Unexpected option name passed to PJRT_Client_Create"));
}

TEST(PjRtCApiHelperTest, InvalidOptionTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"string", static_cast<int64_t>(1)}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, tsl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Option passed to PJRT_Client_Create with name string "
                        "has type index 1 but expected type index is 0"));
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutBothLayoutAndStrides) {
  xla::Layout layout;
  std::optional<absl::Span<const int64_t>> strides;
  strides = {4, 8};
  auto c_layout = ConvertToCLayout(&layout, strides);

  EXPECT_FALSE(c_layout.ok());
  EXPECT_THAT(c_layout.status().message(),
              HasSubstr("Both xla::Layout and byte_strides are set when "
                        "converting to PJRT_Layout"));
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutNoLayoutOrStrides) {
  auto c_layout = ConvertToCLayout(nullptr, std::nullopt);

  EXPECT_FALSE(c_layout.ok());
  EXPECT_THAT(c_layout.status().message(),
              HasSubstr("Both xla::Layout and byte_strides are not set when "
                        "converting to PJRT_Layout"));
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromStrides) {
  std::optional<absl::Span<const int64_t>> strides;
  strides = {4, 8};
  auto c_layout = ConvertToCLayout(nullptr, strides);

  EXPECT_TRUE(c_layout.ok());
  EXPECT_EQ(c_layout->type, PJRT_Layout_Type::PJRT_Layout_Strides);
  EXPECT_EQ(c_layout->strides.num_byte_strides, 2);
  EXPECT_EQ(c_layout->strides.byte_strides[0], strides.value()[0]);
  EXPECT_EQ(c_layout->strides.byte_strides[1], strides.value()[1]);
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromLayoutNoTiles) {
  std::vector<int64_t> minor_to_major = {1, 0};
  xla::Layout layout(minor_to_major);

  auto c_layout = ConvertToCLayout(&layout, std::nullopt);

  EXPECT_TRUE(c_layout.ok());
  EXPECT_EQ(c_layout->type, PJRT_Layout_Type::PJRT_Layout_TiledLayout);
  EXPECT_EQ(c_layout->tiled_layout.tiles_size, 0);
  auto c_minor_to_major = c_layout->tiled_layout.minor_to_major;
  EXPECT_EQ(c_minor_to_major.size, 2);
  EXPECT_EQ(c_minor_to_major.inlined[0], minor_to_major[0]);
  EXPECT_EQ(c_minor_to_major.inlined[1], minor_to_major[1]);
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromLayoutWithTiles) {
  std::vector<int64_t> minor_to_major = {1, 0};
  xla::Layout layout(minor_to_major);
  std::vector<int64_t> tile_dims = {1, 128};
  layout.mutable_tiles()->push_back(xla::Tile(tile_dims));

  auto c_layout = ConvertToCLayout(&layout, std::nullopt);

  EXPECT_TRUE(c_layout.ok());
  EXPECT_EQ(c_layout->type, PJRT_Layout_Type::PJRT_Layout_TiledLayout);
  auto c_minor_to_major = c_layout->tiled_layout.minor_to_major;
  EXPECT_EQ(c_minor_to_major.size, 2);
  EXPECT_EQ(c_minor_to_major.inlined[0], minor_to_major[0]);
  EXPECT_EQ(c_minor_to_major.inlined[1], minor_to_major[1]);
  EXPECT_EQ(c_layout->tiled_layout.tiles_size, 1);
  auto c_tile = c_layout->tiled_layout.tiles_inlined[0];
  EXPECT_EQ(c_tile.size, 2);
  EXPECT_EQ(c_tile.inlined[0], tile_dims[0]);
  EXPECT_EQ(c_tile.inlined[1], tile_dims[1]);
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToStrides) {
  std::vector<int64_t> strides = {4, 8};
  PJRT_Layout c_layout;
  c_layout.type = PJRT_Layout_Type::PJRT_Layout_Strides;
  c_layout.strides.num_byte_strides = strides.size();
  c_layout.strides.byte_strides = strides.data();

  auto layout_or_stride = ConvertFromCLayout(&c_layout);

  EXPECT_TRUE(layout_or_stride.ok());
  EXPECT_TRUE(
      std::holds_alternative<absl::Span<int64_t const>>(*layout_or_stride));
  auto result = std::get<absl::Span<int64_t const>>(*layout_or_stride);
  EXPECT_THAT(result, ElementsAreArray(strides));
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToLayout) {
  PJRT_Layout c_layout;
  c_layout.type = PJRT_Layout_Type::PJRT_Layout_TiledLayout;
  PJRT_Int64List minor_to_major;
  minor_to_major.size = 2;
  minor_to_major.inlined[0] = 1;
  minor_to_major.inlined[1] = 0;
  c_layout.tiled_layout.minor_to_major = minor_to_major;
  PJRT_Int64List tile;
  tile.size = 2;
  tile.inlined[0] = 1;
  tile.inlined[1] = 128;
  c_layout.tiled_layout.tiles_size = 1;
  c_layout.tiled_layout.tiles_inlined[0] = tile;

  auto layout_or_stride = ConvertFromCLayout(&c_layout);

  EXPECT_TRUE(layout_or_stride.ok());
  EXPECT_TRUE(std::holds_alternative<xla::Layout>(*layout_or_stride));
  auto layout = std::get<xla::Layout>(*layout_or_stride);
  EXPECT_EQ(layout.ToString(), "{1,0:T(1,128)}");
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToLayoutHeap) {
  std::vector<int64_t> cpp_minor_to_major{0, 1, 2, 3, 4, 5, 6};
  PJRT_Layout c_layout;
  c_layout.type = PJRT_Layout_Type::PJRT_Layout_TiledLayout;
  PJRT_Int64List minor_to_major;
  minor_to_major.size = cpp_minor_to_major.size();
  minor_to_major.heap = new int64_t[cpp_minor_to_major.size()];
  std::copy(cpp_minor_to_major.begin(), cpp_minor_to_major.end(),
            minor_to_major.heap);
  c_layout.tiled_layout.minor_to_major = minor_to_major;
  c_layout.tiled_layout.tiles_size = PJRT_C_API_MAX_INLINED + 1;
  c_layout.tiled_layout.tiles_heap =
      new PJRT_Int64List[PJRT_C_API_MAX_INLINED + 1];
  std::vector<int64_t> cpp_tile{0, 1, 2, 3, 4, 5, 6};
  for (int i = 0; i < PJRT_C_API_MAX_INLINED + 1; i++) {
    PJRT_Int64List tile;
    tile.size = cpp_tile.size();
    tile.heap = new int64_t[cpp_tile.size()];
    std::copy(cpp_tile.begin(), cpp_tile.end(), tile.heap);
    c_layout.tiled_layout.tiles_heap[i] = tile;
  }

  auto layout_or_stride = ConvertFromCLayout(&c_layout);

  EXPECT_TRUE(layout_or_stride.ok());
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToLayoutNoTile) {
  PJRT_Layout c_layout;
  c_layout.type = PJRT_Layout_Type::PJRT_Layout_TiledLayout;
  c_layout.tiled_layout.tiles_size = 0;
  PJRT_Int64List minor_to_major;
  minor_to_major.size = 2;
  minor_to_major.inlined[0] = 1;
  minor_to_major.inlined[1] = 0;
  c_layout.tiled_layout.minor_to_major = minor_to_major;

  auto layout_or_stride = ConvertFromCLayout(&c_layout);

  EXPECT_TRUE(layout_or_stride.ok());
  EXPECT_TRUE(std::holds_alternative<xla::Layout>(*layout_or_stride));
  auto layout = std::get<xla::Layout>(*layout_or_stride);
  EXPECT_EQ(layout.ToString(), "{1,0}");
}

}  // namespace
}  // namespace pjrt
