#pragma once

#include <memory>
#include <mutex>
#include <ostream>
#include <unordered_map>
#include <vector>
#include "abstract_array.h"
#include "attribute.h"
#include "box.h"
#include "owned_array.h"
#include "plane.h"

#pragma pack(push, 1)
struct KdNode {
    union {
        // Interior node, splitting position along the axis
        float split_pos;
        // Leaf node, offset in 'primitive_indices' to its contained prims
        uint32_t prim_indices_offset;
    };
    // Used by inner and leaf, lower 2 bits used by both inner and leaf
    // nodes, for inner nodes the lower bits track the split axis,
    // for leaf nodes they indicate it's a leaf
    union {
        // Interior node, offset to its right child (with elements above
        // the splitting plane)
        uint32_t right_child;
        // Leaf node, number of primitives in the leaf
        uint32_t num_prims;
    };

    // Interior node
    KdNode(float split_pos, AXIS split_axis);
    // Leaf node
    KdNode(uint32_t nprims, uint32_t prim_offset);

    KdNode(const KdNode &n);

    KdNode();

    KdNode& operator=(const KdNode &n);

    void set_right_child(uint32_t right_child);

    uint32_t get_num_prims() const;

    uint32_t right_child_offset() const;

    AXIS split_axis() const;

    bool is_leaf() const;
};
#pragma pack(pop)

struct BATree {
    Box tree_bounds;
    std::shared_ptr<AbstractArray<glm::vec3>> points;
    std::shared_ptr<AbstractArray<KdNode>> nodes;
    // TODO: May be better to keep these ranges stored in separate arrays instead of interleaved
    std::shared_ptr<AbstractArray<glm::vec2>> node_attrib_ranges;
    std::vector<Attribute> attribs;
    std::unordered_map<std::string, size_t> attrib_ids;

    BATree() = default;

    BATree(Box tree_bounds,
           const std::shared_ptr<AbstractArray<glm::vec3>> &points,
           const std::shared_ptr<AbstractArray<KdNode>> &nodes,
           const std::shared_ptr<AbstractArray<glm::vec2>> &node_attrib_ranges,
           std::vector<Attribute> &attribs);

    // Query all particles contained in some bounding box
    void query_box(const Box &b,
                   std::vector<glm::vec3> &query,
                   std::vector<AttributeQuery> *attrib_queries = nullptr) const;

    // For debugging/testing: query the bounding boxes of each node within the query box
    void get_bounding_boxes(std::vector<Box> &boxes, const Box &query_box) const;

    // For debugging/testing: query the splitting planes of the tree
    void get_splitting_planes(std::vector<Plane> &planes,
                              const Box &query_box,
                              std::vector<AttributeQuery> *attrib_queries = nullptr) const;

private:
    bool node_overlaps_query(uint32_t n,
                             const std::vector<size_t> &query_indices,
                             const std::vector<AttributeQuery> &query) const;
};

