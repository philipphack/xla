/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include "xla/python/pytree.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "xla/python/exceptions.h"
#include "tsl/platform/logging.h"

namespace xla {

namespace py = pybind11;

PyTreeRegistry::PyTreeRegistry(bool enable_none, bool enable_tuple,
                               bool enable_namedtuple, bool enable_list,
                               bool enable_dict) {
  auto add_builtin_type = [&](PyTypeObject* type_obj, PyTreeKind kind) {
    py::object type = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject*>(type_obj));
    auto registration = std::make_unique<Registration>();
    registration->kind = kind;
    registration->type = type;
    CHECK(registrations_.emplace(type, std::move(registration)).second);
  };
  if (enable_none) {
    add_builtin_type(Py_TYPE(Py_None), PyTreeKind::kNone);
  }
  if (enable_tuple) {
    add_builtin_type(&PyTuple_Type, PyTreeKind::kTuple);
  }
  enable_namedtuple_ = enable_namedtuple;
  if (enable_list) {
    add_builtin_type(&PyList_Type, PyTreeKind::kList);
  }
  if (enable_dict) {
    add_builtin_type(&PyDict_Type, PyTreeKind::kDict);
  }
}

void PyTreeRegistry::Register(py::object type, py::function to_iterable,
                              py::function from_iterable) {
  auto registration = std::make_unique<Registration>();
  registration->kind = PyTreeKind::kCustom;
  registration->type = type;
  registration->to_iterable = std::move(to_iterable);
  registration->from_iterable = std::move(from_iterable);
  auto it = registrations_.emplace(type, std::move(registration));
  if (!it.second) {
    throw std::invalid_argument(
        absl::StrFormat("Duplicate custom PyTreeDef type registration for %s.",
                        py::repr(type)));
  }
}

// Computes the node kind of a given Python object.
PyTreeKind PyTreeRegistry::KindOfObject(
    py::handle obj, PyTreeRegistry::Registration const** custom) const {
  const PyTreeRegistry::Registration* registration = Lookup(obj.get_type());
  if (registration) {
    if (registration->kind == PyTreeKind::kCustom) {
      *custom = registration;
    } else {
      *custom = nullptr;
    }
    return registration->kind;
  } else if (py::isinstance<py::tuple>(obj) && py::hasattr(obj, "_fields")) {
    // We can only identify namedtuples heuristically, here by the presence of
    // a _fields attribute.
    return PyTreeKind::kNamedTuple;
  } else {
    return PyTreeKind::kLeaf;
  }
}

/*static*/ const PyTreeRegistry::Registration* PyTreeRegistry::Lookup(
    py::handle type) const {
  auto it = registrations_.find(type);
  return it == registrations_.end() ? nullptr : it->second.get();
}

py::dict PyTreeRegistry::ToPickle() const {
  py::dict out;
  py::list registrations;
  bool enable_none = false, enable_tuple = false, enable_list = false,
       enable_dict = false;
  for (const auto& entry : registrations_) {
    const auto& registration = *entry.second;
    switch (registration.kind) {
      case PyTreeKind::kNone:
        enable_none = true;
        break;
      case PyTreeKind::kTuple:
        enable_tuple = true;
        break;
      case PyTreeKind::kList:
        enable_list = true;
        break;
      case PyTreeKind::kDict:
        enable_dict = true;
        break;
      case PyTreeKind::kCustom:
        registrations.append(py::make_tuple(registration.type,
                                            registration.to_iterable,
                                            registration.from_iterable));
        break;

      case PyTreeKind::kLeaf:
      case PyTreeKind::kNamedTuple:
        throw XlaRuntimeError(
            "Invalid PyTreeKind in PyTreeRegistry::ToPickle()");
    }
  }
  out["registrations"] = registrations;
  out["enable_none"] = py::bool_(enable_none);
  out["enable_tuple"] = py::bool_(enable_tuple);
  out["enable_namedtuple"] = py::bool_(enable_namedtuple_);
  out["enable_list"] = py::bool_(enable_list);
  out["enable_dict"] = py::bool_(enable_dict);
  return out;
}

/*static*/ std::unique_ptr<PyTreeRegistry> PyTreeRegistry::FromPickle(
    py::dict pickle) {
  auto registry = std::make_unique<PyTreeRegistry>(
      pickle["enable_none"].cast<bool>(), pickle["enable_tuple"].cast<bool>(),
      pickle["enable_namedtuple"].cast<bool>(),
      pickle["enable_list"].cast<bool>(), pickle["enable_dict"].cast<bool>());
  for (const auto& registration : pickle["registrations"].cast<py::list>()) {
    auto r = registration.cast<py::tuple>();
    if (r.size() != 3) {
      throw XlaRuntimeError(
          "Pickled PyTreeRegistry registrations must be 3-tuples");
    }
    registry->Register(r[0], r[1], r[2]);
  }
  return registry;
}

/*static*/ std::vector<py::object> GetSortedPyDictKeys(PyObject* py_dict) {
  std::vector<py::object> keys;
  keys.reserve(PyDict_Size(py_dict));
  PyObject* key;
  Py_ssize_t pos = 0;
  while (PyDict_Next(py_dict, &pos, &key, /*value=*/nullptr)) {
    keys.push_back(py::reinterpret_borrow<py::object>(key));
  }

  int ret = 0;
  std::stable_sort(keys.begin(), keys.end(),
                   [&ret](const py::object& a, const py::object& b) {
                     int cmp =
                         PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_LT);
                     if (cmp == -1) ret = -1;
                     return cmp;
                   });
  if (ret == -1) {
    throw py::error_already_set();
  }
  return keys;
}

/*static*/ bool IsSortedPyDictKeysEqual(absl::Span<const py::object> lhs,
                                        absl::Span<const py::object> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs[i].not_equal(rhs[i])) {
      return false;
    }
  }
  return true;
}

bool PyTreeDef::operator==(const PyTreeDef& other) const {
  if (traversal_.size() != other.traversal_.size()) {
    return false;
  }
  for (size_t i = 0; i < traversal_.size(); ++i) {
    const Node& a = traversal_[i];
    const Node& b = other.traversal_[i];
    if (a.kind != b.kind || a.arity != b.arity ||
        (a.node_data.ptr() == nullptr) != (b.node_data.ptr() == nullptr) ||
        (a.sorted_dict_keys.size() != b.sorted_dict_keys.size()) ||
        a.custom != b.custom) {
      return false;
    }
    if (a.node_data && a.node_data.not_equal(b.node_data)) {
      return false;
    }
    if (!IsSortedPyDictKeysEqual(a.sorted_dict_keys, b.sorted_dict_keys)) {
      return false;
    }
    // We don't need to test equality of num_leaves and num_nodes since they
    // are derivable from the other node data.
  }
  return true;
}

template <typename T>
void PyTreeDef::FlattenImpl(py::handle handle, T& leaves,
                            const std::optional<py::function>& leaf_predicate) {
  Node node;
  const int start_num_nodes = traversal_.size();
  const int start_num_leaves = leaves.size();
  if (leaf_predicate && (*leaf_predicate)(handle).cast<bool>()) {
    leaves.push_back(py::reinterpret_borrow<py::object>(handle));
  } else {
    node.kind = registry_->KindOfObject(handle, &node.custom);
    auto recurse = [this, &leaf_predicate, &leaves](py::handle child) {
      Flatten(child, leaves, leaf_predicate);
    };
    switch (node.kind) {
      case PyTreeKind::kNone:
        // Nothing to do.
        break;
      case PyTreeKind::kTuple: {
        node.arity = PyTuple_GET_SIZE(handle.ptr());
        for (int i = 0; i < node.arity; ++i) {
          recurse(PyTuple_GET_ITEM(handle.ptr(), i));
        }
        break;
      }
      case PyTreeKind::kList: {
        node.arity = PyList_GET_SIZE(handle.ptr());
        for (int i = 0; i < node.arity; ++i) {
          recurse(PyList_GET_ITEM(handle.ptr(), i));
        }
        break;
      }
      case PyTreeKind::kDict: {
        py::dict dict = py::reinterpret_borrow<py::dict>(handle);

        std::vector<py::object> keys = GetSortedPyDictKeys(dict.ptr());
        for (py::handle key : keys) {
          recurse(dict[key]);
        }
        node.arity = dict.size();
        node.sorted_dict_keys = std::move(keys);
        break;
      }
      case PyTreeKind::kCustom: {
        py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
        if (out.size() != 2) {
          throw xla::XlaRuntimeError(
              "PyTree custom to_iterable function should return a pair");
        }
        node.node_data = out[1];
        node.arity = 0;
        for (py::handle entry : py::cast<py::iterable>(out[0])) {
          ++node.arity;
          recurse(entry);
        }
        break;
      }
      case PyTreeKind::kNamedTuple: {
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
        node.arity = tuple.size();
        node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
        for (py::handle entry : tuple) {
          recurse(entry);
        }
        break;
      }
      default:
        DCHECK(node.kind == PyTreeKind::kLeaf);
        leaves.push_back(py::reinterpret_borrow<py::object>(handle));
    }
  }
  node.num_nodes = traversal_.size() - start_num_nodes + 1;
  node.num_leaves = leaves.size() - start_num_leaves;
  traversal_.push_back(std::move(node));
}

void PyTreeDef::Flatten(py::handle handle,
                        absl::InlinedVector<py::object, 2>& leaves,
                        std::optional<py::function> leaf_predicate) {
  FlattenImpl(handle, leaves, leaf_predicate);
}

void PyTreeDef::Flatten(py::handle handle, std::vector<py::object>& leaves,
                        std::optional<py::function> leaf_predicate) {
  FlattenImpl(handle, leaves, leaf_predicate);
}

/*static*/ bool PyTreeDef::AllLeaves(PyTreeRegistry* registry,
                                     const py::iterable& x) {
  const PyTreeRegistry::Registration* custom;
  for (const py::handle& h : x) {
    if (registry->KindOfObject(h, &custom) != PyTreeKind::kLeaf) return false;
  }
  return true;
}

template <typename T>
py::object PyTreeDef::UnflattenImpl(T leaves) const {
  absl::InlinedVector<py::object, 4> agenda;
  auto it = leaves.begin();
  int leaf_count = 0;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for TreeDef node.");
    }
    switch (node.kind) {
      case PyTreeKind::kLeaf:
        if (it == leaves.end()) {
          throw std::invalid_argument(absl::StrFormat(
              "Too few leaves for PyTreeDef; expected %d, got %d", num_leaves(),
              leaf_count));
        }
        agenda.push_back(py::reinterpret_borrow<py::object>(*it));
        ++it;
        ++leaf_count;
        break;

      case PyTreeKind::kNone:
      case PyTreeKind::kTuple:
      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kList:
      case PyTreeKind::kDict:
      case PyTreeKind::kCustom: {
        const int size = agenda.size();
        absl::Span<py::object> span;
        if (node.arity > 0) {
          span = absl::Span<py::object>(&agenda[size - node.arity], node.arity);
        }
        py::object o = MakeNode(node, span);
        agenda.resize(size - node.arity);
        agenda.push_back(o);
        break;
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument(absl::StrFormat(
        "Too many leaves for PyTreeDef; expected %d.", num_leaves()));
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

py::object PyTreeDef::Unflatten(py::iterable leaves) const {
  return UnflattenImpl(leaves);
}

py::object PyTreeDef::Unflatten(absl::Span<const py::object> leaves) const {
  return UnflattenImpl(leaves);
}

/*static*/ py::object PyTreeDef::MakeNode(const PyTreeDef::Node& node,
                                          absl::Span<py::object> children) {
  if (children.size() != node.arity) {
    throw std::logic_error("Node arity mismatch.");
  }
  switch (node.kind) {
    case PyTreeKind::kLeaf:
      throw std::logic_error("MakeNode not implemented for leaves.");

    case PyTreeKind::kNone:
      return py::none();

    case PyTreeKind::kTuple:
    case PyTreeKind::kNamedTuple: {
      py::tuple tuple(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        tuple[i] = std::move(children[i]);
      }
      if (node.kind == PyTreeKind::kNamedTuple) {
        return node.node_data(*tuple);
      } else {
        return std::move(tuple);
      }
    }

    case PyTreeKind::kList: {
      py::list list(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        list[i] = std::move(children[i]);
      }
      return std::move(list);
    }

    case PyTreeKind::kDict: {
      py::dict dict;
      for (int i = 0; i < node.arity; ++i) {
        dict[node.sorted_dict_keys[i]] = std::move(children[i]);
      }
      return std::move(dict);
      break;
    }
    case PyTreeKind::kCustom: {
      py::tuple tuple(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        tuple[i] = std::move(children[i]);
      }
      return node.custom->from_iterable(node.node_data, tuple);
    }
  }
  throw std::logic_error("Unreachable code.");
}

py::list PyTreeDef::FlattenUpTo(py::handle xs) const {
  py::list leaves(num_leaves());
  std::vector<py::object> agenda;
  agenda.push_back(py::reinterpret_borrow<py::object>(xs));
  auto it = traversal_.rbegin();
  int leaf = num_leaves() - 1;
  while (!agenda.empty()) {
    if (it == traversal_.rend()) {
      throw std::invalid_argument(absl::StrFormat(
          "Tree structures did not match: %s vs %s", py::repr(xs), ToString()));
    }
    const Node& node = *it;
    py::object object = agenda.back();
    agenda.pop_back();
    ++it;

    switch (node.kind) {
      case PyTreeKind::kLeaf:
        if (leaf < 0) {
          throw std::logic_error("Leaf count mismatch.");
        }
        leaves[leaf] = py::reinterpret_borrow<py::object>(object);
        --leaf;
        break;

      case PyTreeKind::kNone:
        break;

      case PyTreeKind::kTuple: {
        if (!PyTuple_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected tuple, got %s.", py::repr(object)));
        }
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(
              absl::StrFormat("Tuple arity mismatch: %d != %d; tuple: %s.",
                              tuple.size(), node.arity, py::repr(object)));
        }
        for (py::handle entry : tuple) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case PyTreeKind::kList: {
        if (!PyList_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected list, got %s.", py::repr(object)));
        }
        py::list list = py::reinterpret_borrow<py::list>(object);
        if (list.size() != node.arity) {
          throw std::invalid_argument(
              absl::StrFormat("List arity mismatch: %d != %d; list: %s.",
                              list.size(), node.arity, py::repr(object)));
        }
        for (py::handle entry : list) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case PyTreeKind::kDict: {
        if (!PyDict_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected dict, got %s.", py::repr(object)));
        }
        py::dict dict = py::reinterpret_borrow<py::dict>(object);
        std::vector<py::object> keys = GetSortedPyDictKeys(dict.ptr());
        if (!IsSortedPyDictKeysEqual(keys, node.sorted_dict_keys)) {
          // Convert to a py::list for py::repr to avoid having to stringify a
          // vector. This is error path so it is fine to pay conversion cost.
          throw std::invalid_argument(absl::StrFormat(
              "Dict key mismatch; expected keys: %s; dict: %s.",
              py::repr(py::cast(node.sorted_dict_keys)), py::repr(object)));
        }
        for (py::handle key : keys) {
          agenda.push_back(dict[key]);
        }
        break;
      }

      case PyTreeKind::kNamedTuple: {
        if (!py::isinstance<py::tuple>(object) ||
            !py::hasattr(object, "_fields")) {
          throw std::invalid_argument(absl::StrFormat(
              "Expected named tuple, got %s.", py::repr(object)));
        }
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple arity mismatch: %d != %d; tuple: %s.", tuple.size(),
              node.arity, py::repr(object)));
        }
        if (tuple.get_type().not_equal(node.node_data)) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple type mismatch: expected type: %s, tuple: %s.",
              py::repr(node.node_data), py::repr(object)));
        }
        for (py::handle entry : tuple) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case PyTreeKind::kCustom: {
        auto* registration = registry_->Lookup(object.get_type());
        if (registration != node.custom) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom node type mismatch: expected type: %s, value: %s.",
              py::repr(node.custom->type), py::repr(object)));
        }
        py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(object));
        if (out.size() != 2) {
          throw xla::XlaRuntimeError(
              "PyTree custom to_iterable function should return a pair");
        }
        if (node.node_data.not_equal(out[1])) {
          throw std::invalid_argument(absl::StrFormat(
              "Mismatch custom node data: %s != %s; value: %s.",
              py::repr(node.node_data), py::repr(out[1]), py::repr(object)));
        }
        int arity = 0;
        for (py::handle entry : py::cast<py::iterable>(out[0])) {
          ++arity;
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        if (arity != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom type arity mismatch: %d != %d; value: %s.", arity,
              node.arity, py::repr(object)));
        }
        break;
      }
    }
  }
  if (it != traversal_.rend() || leaf != -1) {
    throw std::invalid_argument(absl::StrFormat(
        "Tree structures did not match: %s vs %s", py::repr(xs), ToString()));
  }
  return leaves;
}

py::object PyTreeDef::Walk(const py::function& f_node, py::handle f_leaf,
                           py::iterable leaves) const {
  std::vector<py::object> agenda;
  auto it = leaves.begin();
  for (const Node& node : traversal_) {
    switch (node.kind) {
      case PyTreeKind::kLeaf: {
        if (it == leaves.end()) {
          throw std::invalid_argument("Too few leaves for PyTreeDef");
        }

        py::object leaf = py::reinterpret_borrow<py::object>(*it);
        agenda.push_back(f_leaf.is_none() ? std::move(leaf)
                                          : f_leaf(std::move(leaf)));
        ++it;
        break;
      }

      case PyTreeKind::kNone:
      case PyTreeKind::kTuple:
      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kList:
      case PyTreeKind::kDict:
      case PyTreeKind::kCustom: {
        if (agenda.size() < node.arity) {
          throw std::logic_error("Too few elements for custom type.");
        }
        py::tuple tuple(node.arity);
        for (int i = node.arity - 1; i >= 0; --i) {
          tuple[i] = agenda.back();
          agenda.pop_back();
        }
        py::object node_data = node.node_data;
        if (node.kind == PyTreeKind::kDict) {
          // Convert to a py::list for f_node invocation.
          node_data = py::cast(node.sorted_dict_keys);
        }
        agenda.push_back(f_node(tuple, node_data ? node_data : py::none()));
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument("Too many leaves for PyTreeDef");
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

py::object PyTreeDef::FromIterableTreeHelper(
    py::handle xs,
    absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it) const {
  if (*it == traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  const Node& node = **it;
  ++*it;
  if (node.kind == PyTreeKind::kLeaf) {
    return py::reinterpret_borrow<py::object>(xs);
  }
  py::iterable iterable = py::reinterpret_borrow<py::iterable>(xs);
  std::vector<py::object> ys;
  ys.reserve(node.arity);
  for (py::handle x : iterable) {
    ys.push_back(py::reinterpret_borrow<py::object>(x));
  }
  if (ys.size() != node.arity) {
    throw std::invalid_argument("Arity mismatch between trees");
  }
  for (int j = node.arity - 1; j >= 0; --j) {
    ys[j] = FromIterableTreeHelper(ys[j], it);
  }

  return MakeNode(node, absl::MakeSpan(ys));
}

py::object PyTreeDef::FromIterableTree(py::handle xs) const {
  auto it = traversal_.rbegin();
  py::object out = FromIterableTreeHelper(xs, &it);
  if (it != traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  return out;
}

std::unique_ptr<PyTreeDef> PyTreeDef::Compose(const PyTreeDef& inner) const {
  if (inner.registry_ != registry_) {
    throw std::invalid_argument(
        "PyTree registries of PyTreeDefs passed to Compose() must match.");
  }
  auto out = std::make_unique<PyTreeDef>(registry_->shared_from_this());
  for (const Node& n : traversal_) {
    if (n.kind == PyTreeKind::kLeaf) {
      absl::c_copy(inner.traversal_, std::back_inserter(out->traversal_));
    } else {
      out->traversal_.push_back(n);
    }
  }
  const auto& root = traversal_.back();
  const auto& inner_root = inner.traversal_.back();
  // TODO(tomhennigan): This should update all nodes in the traversal.
  auto& out_root = out->traversal_.back();
  out_root.num_nodes = (root.num_nodes - root.num_leaves) +
                       (inner_root.num_nodes * root.num_leaves);
  out_root.num_leaves *= inner_root.num_leaves;
  return out;
}

/*static*/ std::unique_ptr<PyTreeDef> PyTreeDef::Tuple(
    std::shared_ptr<PyTreeRegistry> registry,
    absl::Span<PyTreeDef* const> defs) {
  auto out = std::make_unique<PyTreeDef>(std::move(registry));
  int num_leaves = 0;
  for (const PyTreeDef* def : defs) {
    if (def->registry() != out->registry()) {
      throw std::invalid_argument(
          "PyTree registries of PyTreeDefs passed to Tuple() must match.");
    }
    absl::c_copy(def->traversal_, std::back_inserter(out->traversal_));
    num_leaves += def->num_leaves();
  }
  Node node;
  node.kind = PyTreeKind::kTuple;
  node.arity = defs.size();
  node.num_leaves = num_leaves;
  node.num_nodes = out->traversal_.size() + 1;
  out->traversal_.push_back(node);
  return out;
}

std::vector<std::unique_ptr<PyTreeDef>> PyTreeDef::Children() const {
  std::vector<std::unique_ptr<PyTreeDef>> children;
  if (traversal_.empty()) {
    return children;
  }
  Node const& root = traversal_.back();
  children.resize(root.arity);
  int pos = traversal_.size() - 1;
  for (int i = root.arity - 1; i >= 0; --i) {
    children[i] = std::make_unique<PyTreeDef>(registry_->shared_from_this());
    const Node& node = traversal_.at(pos - 1);
    if (pos < node.num_nodes) {
      throw std::logic_error("children() walked off start of array");
    }
    std::copy(traversal_.begin() + pos - node.num_nodes,
              traversal_.begin() + pos,
              std::back_inserter(children[i]->traversal_));
    pos -= node.num_nodes;
  }
  if (pos != 0) {
    throw std::logic_error("pos != 0 at end of PyTreeDef::Children");
  }
  return children;
}

std::string PyTreeDef::ToString() const {
  std::vector<std::string> agenda;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for container.");
    }

    std::string children =
        absl::StrJoin(agenda.end() - node.arity, agenda.end(), ", ");
    std::string representation;
    switch (node.kind) {
      case PyTreeKind::kLeaf:
        agenda.push_back("*");
        continue;
      case PyTreeKind::kNone:
        representation = "None";
        break;
      case PyTreeKind::kTuple:
        // Tuples with only one element must have a trailing comma.
        if (node.arity == 1) children += ",";
        representation = absl::StrCat("(", children, ")");
        break;
      case PyTreeKind::kList:
        representation = absl::StrCat("[", children, "]");
        break;
      case PyTreeKind::kDict: {
        if (node.sorted_dict_keys.size() != node.arity) {
          throw std::logic_error("Number of keys and entries does not match.");
        }
        representation = "{";
        std::string separator;
        auto child_iter = agenda.end() - node.arity;
        for (const py::handle& key : node.sorted_dict_keys) {
          absl::StrAppendFormat(&representation, "%s%s: %s", separator,
                                py::repr(key), *child_iter);
          child_iter++;
          separator = ", ";
        }
        representation += "}";
        break;
      }

      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kCustom: {
        std::string kind;
        std::string data;
        if (node.kind == PyTreeKind::kNamedTuple) {
          kind = "namedtuple";
          if (node.node_data) {
            // Node data for named tuples is the type.
            data = absl::StrFormat(
                "[%s]", py::str(py::getattr(node.node_data, "__name__")));
          }
        } else {
          kind = static_cast<std::string>(
              py::str(py::getattr(node.custom->type, "__name__")));
          if (node.node_data) {
            data = absl::StrFormat("[%s]", py::str(node.node_data));
          }
        }

        representation =
            absl::StrFormat("CustomNode(%s%s, [%s])", kind, data, children);
        break;
      }
    }
    agenda.erase(agenda.end() - node.arity, agenda.end());
    agenda.push_back(std::move(representation));
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return absl::StrCat("PyTreeDef(", agenda.back(), ")");
}

py::object PyTreeDef::ToPickle() const {
  py::list traversal;
  for (const auto& node : traversal_) {
    py::object node_data = node.node_data;
    if (node.kind == PyTreeKind::kDict) {
      // Convert to a py::list for pickling to avoid having to pickle a vector.
      // Pickle should be a rare operation so this conversion cost is hopefully
      // on non-critical path.
      node_data = py::cast(node.sorted_dict_keys);
    }
    traversal.append(
        py::make_tuple(static_cast<int>(node.kind), node.arity,
                       node_data ? node_data : py::none(),
                       node.custom != nullptr ? node.custom->type : py::none(),
                       node.num_leaves, node.num_nodes));
  }
  return py::make_tuple(py::cast(registry_->shared_from_this()), traversal);
}

PyTreeDef PyTreeDef::FromPickle(py::object pickleable) {
  py::tuple pickle = pickleable.cast<py::tuple>();
  if (pickle.size() != 2) {
    throw xla::XlaRuntimeError("Malformed pickled PyTreeDef, expected 2-tuple");
  }
  auto registry = py::cast<std::shared_ptr<PyTreeRegistry>>(pickle[0]);
  PyTreeDef tree(registry);
  for (const auto& item : pickle[1].cast<py::list>()) {
    auto t = item.cast<py::tuple>();
    if (t.size() != 6) {
      throw xla::XlaRuntimeError("Malformed pickled PyTreeDef");
    }
    Node& node = tree.traversal_.emplace_back();
    node.kind = static_cast<PyTreeKind>(t[0].cast<int>());
    node.arity = t[1].cast<int>();
    switch (node.kind) {
      case PyTreeKind::kNamedTuple:
        node.node_data = t[2].cast<py::type>();
        break;
      case PyTreeKind::kDict:
        node.sorted_dict_keys = t[2].cast<std::vector<py::object>>();
        break;
      case PyTreeKind::kCustom:
        node.node_data = t[2];
        break;
      default:
        if (!t[2].is_none()) {
          throw xla::XlaRuntimeError("Malformed pickled PyTreeDef");
        }
        break;
    }
    if (node.kind == PyTreeKind::kCustom) {
      node.custom = t[3].is_none() ? nullptr : registry->Lookup(t[3]);
      if (node.custom == nullptr) {
        throw xla::XlaRuntimeError(
            absl::StrCat("Unknown custom type in pickled PyTreeDef: ",
                         static_cast<std::string>(py::repr(t[3]))));
      }
    } else {
      if (!t[3].is_none()) {
        throw xla::XlaRuntimeError("Malformed pickled PyTreeDef");
      }
    }
    node.num_leaves = t[4].cast<int>();
    node.num_nodes = t[5].cast<int>();
  }
  return tree;
}

void BuildPytreeSubmodule(py::module& m) {
  py::module pytree = m.def_submodule("pytree", "Python tree library");
  pytree.attr("version") = py::int_(3);

  py::class_<PyTreeDef> treedef(pytree, "PyTreeDef");

  py::class_<PyTreeRegistry, std::shared_ptr<PyTreeRegistry>> registry(
      m, "PyTreeRegistry", py::dynamic_attr());
  registry.def(py::init<bool, bool, bool, bool, bool>(), py::kw_only(),
               py::arg("enable_none") = true, py::arg("enable_tuple") = true,
               py::arg("enable_namedtuple") = true,
               py::arg("enable_list") = true, py::arg("enable_dict") = true);
  registry.def(
      "flatten",
      [](std::shared_ptr<PyTreeRegistry> registry, pybind11::handle x,
         std::optional<pybind11::function> leaf_predicate) {
        std::vector<py::object> leaves;
        PyTreeDef def(std::move(registry));
        def.Flatten(x, leaves, leaf_predicate);
        return std::make_pair(std::move(leaves), std::move(def));
      },
      py::arg("tree"), py::arg("leaf_predicate") = std::nullopt);
  registry.def("register_node", &PyTreeRegistry::Register);
  registry.def(py::pickle([](const PyTreeRegistry& r) { return r.ToPickle(); },
                          &PyTreeRegistry::FromPickle));

  pytree.attr("PyTreeRegistry") = m.attr("PyTreeRegistry");
  pytree.def("tuple", &PyTreeDef::Tuple);
  pytree.def("all_leaves", &PyTreeDef::AllLeaves);

  treedef.def("unflatten",
              static_cast<pybind11::object (PyTreeDef::*)(
                  pybind11::iterable leaves) const>(&PyTreeDef::Unflatten));
  treedef.def("flatten_up_to", &PyTreeDef::FlattenUpTo);
  treedef.def("compose", &PyTreeDef::Compose);
  treedef.def("walk", &PyTreeDef::Walk,
              "Walk pytree, calling f_node(node, node_data) at nodes, and "
              "f_leaf at leaves ",
              py::arg("f_node"), py::arg("f_leaf"), py::arg("leaves"));
  treedef.def("from_iterable_tree", &PyTreeDef::FromIterableTree);
  treedef.def("children", &PyTreeDef::Children);
  treedef.def_property_readonly("num_leaves", &PyTreeDef::num_leaves);
  treedef.def_property_readonly("num_nodes", &PyTreeDef::num_nodes);
  treedef.def("__repr__", &PyTreeDef::ToString);
  treedef.def("__eq__",
              [](const PyTreeDef& a, const PyTreeDef& b) { return a == b; });
  treedef.def("__ne__",
              [](const PyTreeDef& a, const PyTreeDef& b) { return a != b; });
  treedef.def("__hash__", [](const PyTreeDef& t) { return absl::HashOf(t); });
  treedef.def(py::pickle([](const PyTreeDef& t) { return t.ToPickle(); },
                         &PyTreeDef::FromPickle));
}

}  // namespace xla
