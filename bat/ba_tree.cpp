#include "ba_tree.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include "owned_array.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

KdNode::KdNode(float split_pos, AXIS split_axis)
    : split_pos(split_pos), right_child(static_cast<uint32_t>(split_axis))
{
}

KdNode::KdNode(uint32_t nprims, uint32_t prim_offset)
    : prim_indices_offset(prim_offset), num_prims(3 | (nprims << 2))
{
}

KdNode::KdNode(const KdNode &n)
{
    if (n.is_leaf()) {
        prim_indices_offset = n.prim_indices_offset;
        num_prims = n.num_prims;
    } else {
        split_pos = n.split_pos;
        right_child = n.right_child;
    }
}

KdNode::KdNode() : prim_indices_offset(-1), num_prims(-1) {}

KdNode &KdNode::operator=(const KdNode &n)
{
    if (n.is_leaf()) {
        prim_indices_offset = n.prim_indices_offset;
        num_prims = n.num_prims;
    } else {
        split_pos = n.split_pos;
        right_child = n.right_child;
    }
    return *this;
}

void KdNode::set_right_child(uint32_t r)
{
    // Clear the previous right child bits before setting the value
    right_child = right_child & 0x3;
    right_child |= (r << 2);
}

uint32_t KdNode::get_num_prims() const
{
    return num_prims >> 2;
}

uint32_t KdNode::right_child_offset() const
{
    return right_child >> 2;
}

AXIS KdNode::split_axis() const
{
    return static_cast<AXIS>(num_prims & 3);
}

bool KdNode::is_leaf() const
{
    return (num_prims & 3) == 3;
}

BATree::BATree(Box tree_bounds,
               const std::shared_ptr<AbstractArray<glm::vec3>> &points,
               const std::shared_ptr<AbstractArray<KdNode>> &nodes,
               const std::shared_ptr<AbstractArray<glm::vec2>> &node_attrib_ranges,
               std::vector<Attribute> &attribs)
    : tree_bounds(tree_bounds),
      points(points),
      nodes(nodes),
      node_attrib_ranges(node_attrib_ranges),
      attribs(attribs)
{
    for (size_t i = 0; i < attribs.size(); ++i) {
        attrib_ids[attribs[i].name] = i;
    }
}

void BATree::query_box(const Box &b,
                       std::vector<glm::vec3> &query,
                       std::vector<AttributeQuery> *attrib_queries) const
{
    std::array<size_t, 64> node_stack = {0};
    size_t stack_idx = 0;
    size_t current_node = 0;

    std::vector<size_t> query_indices;
    if (attrib_queries) {
        for (auto &a : *attrib_queries) {
            auto fnd = attrib_ids.find(a.name);
            if (fnd == attrib_ids.end()) {
                throw std::runtime_error("Request for attribute " + a.name +
                                         " which does not exist");
            }
            query_indices.push_back(fnd->second);
            a.data_type = attribs[fnd->second].data_type;

            if (a.range.x == a.range.y) {
                std::cout << "Not querying empty range\n";
                return;
            }
        }
    }

    while (true) {
        const KdNode &node = nodes->at(current_node);
        if (!node.is_leaf()) {
            // Interior node, descend into children if they're contained in the box
            bool left_overlaps = b.lower[node.split_axis()] <= node.split_pos;
            bool right_overlaps = b.upper[node.split_axis()] >= node.split_pos;
            if (attrib_queries) {
                left_overlaps =
                    left_overlaps &&
                    node_overlaps_query(current_node + 1, query_indices, *attrib_queries);
                right_overlaps =
                    right_overlaps &&
                    node_overlaps_query(node.right_child_offset(), query_indices, *attrib_queries);
            }

            // If both overlap, descend both children following the left first
            if (left_overlaps && right_overlaps) {
                node_stack[stack_idx++] = node.right_child_offset();
                current_node = current_node + 1;
            } else if (left_overlaps) {
                current_node = current_node + 1;
            } else if (right_overlaps) {
                current_node = node.right_child_offset();
            } else {
                // When filtering by attributes we can have a case where neither child overlaps
                // the query range, even though the parent did (e.g., the child ranges are
                // disjoint) Pop the stack to get the next node to traverse
                if (stack_idx > 0) {
                    current_node = node_stack[--stack_idx];
                } else {
                    break;
                }
            }
        } else {
            // Leaf node, collect points contained in the box
            for (size_t i = 0; i < node.get_num_prims(); ++i) {
                const auto &pt = points->at(node.prim_indices_offset + i);
                if (b.contains_point(pt)) {
                    query.push_back(pt);

                    // Copy in the point's attributes for the query
                    // TODO: If we have attrib queries we should check if it's in the range
                    // queried first
                    if (attrib_queries) {
                        for (size_t j = 0; j < attrib_queries->size(); ++j) {
                            AttributeQuery &q = (*attrib_queries)[j];
                            q.push_back(attribs[query_indices[j]].at(node.prim_indices_offset + i));
                        }
                    }
                }
            }

            // Pop the stack to get the next node to traverse
            if (stack_idx > 0) {
                current_node = node_stack[--stack_idx];
            } else {
                break;
            }
        }
    }
}

void BATree::get_bounding_boxes(std::vector<Box> &boxes, const Box &b) const
{
    std::array<size_t, 64> node_stack = {0};
    std::array<Box, 64> bounds_stack;
    size_t stack_idx = 0;
    size_t current_node = 0;
    Box current_bounds = tree_bounds;

    while (true) {
        const KdNode &node = nodes->at(current_node);
        boxes.push_back(current_bounds);
        if (!node.is_leaf()) {
            // Interior node, descend into children if they're contained in the box
            const bool left_overlaps = b.lower[node.split_axis()] <= node.split_pos;
            const bool right_overlaps = b.upper[node.split_axis()] >= node.split_pos;

            if (left_overlaps && right_overlaps) {
                node_stack[stack_idx] = node.right_child_offset();
                bounds_stack[stack_idx] = current_bounds;
                bounds_stack[stack_idx].lower[node.split_axis()] = node.split_pos;
                ++stack_idx;

                current_node = current_node + 1;
                current_bounds.upper[node.split_axis()] = node.split_pos;
            } else if (left_overlaps) {
                current_node = current_node + 1;
                current_bounds.upper[node.split_axis()] = node.split_pos;
            } else {
                current_node = node.right_child_offset();
                current_bounds.lower[node.split_axis()] = node.split_pos;
            }
        } else {
            // Pop the stack to get the next node to traverse
            if (stack_idx > 0) {
                --stack_idx;
                current_node = node_stack[stack_idx];
                current_bounds = bounds_stack[stack_idx];
            } else {
                break;
            }
        }
    }
}

void BATree::get_splitting_planes(std::vector<Plane> &planes,
                                  const Box &b,
                                  std::vector<AttributeQuery> *attrib_queries) const
{
    std::array<size_t, 64> node_stack = {0};
    std::array<Box, 64> bounds_stack;
    size_t stack_idx = 0;
    size_t current_node = 0;
    Box current_bounds = tree_bounds;

    std::vector<size_t> query_indices;
    if (attrib_queries) {
        for (auto &a : *attrib_queries) {
            auto fnd = attrib_ids.find(a.name);
            if (fnd == attrib_ids.end()) {
                throw std::runtime_error("Request for attribute " + a.name +
                                         " which does not exist");
            }
            query_indices.push_back(fnd->second);
            a.data_type = attribs[fnd->second].data_type;
            std::cout << "attrib '" << a.name << "', query range: " << glm::to_string(a.range)
                      << "\n";

            if (a.range.x == a.range.y) {
                std::cout << "Not querying empty range\n";
                return;
            }
        }
    }

    while (true) {
        const KdNode &node = nodes->at(current_node);
        if (!node.is_leaf()) {
            glm::vec3 plane_origin = current_bounds.center();
            plane_origin[node.split_axis()] = node.split_pos;

            glm::vec3 plane_half_vecs = current_bounds.diagonal() / 2.f;

            plane_half_vecs[node.split_axis()] = 0.f;

            planes.emplace_back(plane_origin, plane_half_vecs);

            // Interior node, descend into children if they're contained in the box
            bool left_overlaps = b.lower[node.split_axis()] <= node.split_pos;
            bool right_overlaps = b.upper[node.split_axis()] >= node.split_pos;
            if (attrib_queries) {
                left_overlaps =
                    left_overlaps &&
                    node_overlaps_query(current_node + 1, query_indices, *attrib_queries);
                right_overlaps =
                    right_overlaps &&
                    node_overlaps_query(node.right_child_offset(), query_indices, *attrib_queries);
            }

            // If both overlap, descend both children following the left first
            if (left_overlaps && right_overlaps) {
                node_stack[stack_idx] = node.right_child_offset();
                bounds_stack[stack_idx] = current_bounds;
                bounds_stack[stack_idx].lower[node.split_axis()] = node.split_pos;
                ++stack_idx;

                current_node = current_node + 1;
                current_bounds.upper[node.split_axis()] = node.split_pos;
            } else if (left_overlaps) {
                current_node = current_node + 1;
                current_bounds.upper[node.split_axis()] = node.split_pos;
            } else if (right_overlaps) {
                current_node = node.right_child_offset();
                current_bounds.lower[node.split_axis()] = node.split_pos;
            } else {
                // When filtering by attributes we can have a case where neither child overlaps
                // the query range, even though the parent did (e.g., the child ranges are
                // disjoint) Pop the stack to get the next node to traverse
                if (stack_idx > 0) {
                    current_node = node_stack[--stack_idx];
                } else {
                    break;
                }
            }
        } else {
            // Pop the stack to get the next node to traverse
            if (stack_idx > 0) {
                --stack_idx;
                current_node = node_stack[stack_idx];
                current_bounds = bounds_stack[stack_idx];
            } else {
                break;
            }
        }
    }
}

bool BATree::node_overlaps_query(uint32_t n,
                                 const std::vector<size_t> &query_indices,
                                 const std::vector<AttributeQuery> &query) const
{
    for (size_t i = 0; i < query.size(); ++i) {
        const size_t attr = query_indices[i];
        const glm::vec2 node_range = node_attrib_ranges->at(n * attribs.size() + attr);
        if (node_range.x > query[i].range.y || node_range.y < query[i].range.x) {
            return false;
        }
    }
    return true;
}

