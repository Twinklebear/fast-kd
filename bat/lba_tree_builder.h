#pragma once

#include <memory>
#include <mutex>
#include <ostream>
#include <vector>
#include "abstract_array.h"
#include "attribute.h"
#include "ba_tree.h"
#include "box.h"
#include "owned_array.h"

/* We track the original ID of the particle so that we can re-order the attribute arrays
 * to match the kd tree once the build is done.
 */
struct LBuildPoint {
    glm::vec3 pos = glm::vec3(0);
    size_t id = -1;
    uint32_t morton_code = 0;

    // Debugging: keep the quantized point too
    glm::uvec3 quantized_pt = glm::uvec3(0);

    LBuildPoint(const glm::vec3 &pos,
                size_t id,
                uint32_t morton_code,
                const glm::uvec3 &quantized_pt);

    LBuildPoint() = default;

    inline operator uint32_t() const
    {
        return morton_code;
    }
};

struct RadixTreeNode {
    // technically we only need gamma, and if the child is a leaf or not
    size_t left_child = std::numeric_limits<size_t>::max();
    bool left_leaf = false;

    size_t right_child = std::numeric_limits<size_t>::max();
    bool right_leaf = false;

    AXIS split_axis = X;
    float split_pos = 0.f;
};

/* Within each leaf of the coarse bottom-up built tree, we construct
 * a better kd-tree using a standard median-split approach. Each treelet
 * is built serially, however the builds are run in parallel
 */
struct LBATreelet {
    LBuildPoint *points;
    const std::vector<Attribute> *in_attribs = nullptr;
    std::vector<Attribute> attributes;
    std::vector<KdNode> nodes;

    uint32_t n_prims = 0;
    uint32_t min_prims = 64;
    uint32_t max_depth;

    LBATreelet(LBuildPoint *points, uint32_t n_prims, std::vector<Attribute> *attributes = nullptr);
    LBATreelet() = default;

    void compact(std::shared_ptr<OwnedArray<KdNode>> &nodes,
                 std::shared_ptr<OwnedArray<glm::vec3>> &points) const;

private:
    uint32_t build(const size_t lo, const size_t hi, const uint32_t depth);
    uint32_t build_leaf(const size_t lo, const size_t hi);
};

/* A hybrid bottom-up/treelet tree. A coarse top-level tree is built
 * bottom-up based on `kd_morton_bits` to produce a coarse hierarchy.
 * Within each leaf cell of this tree we build a spatial median-split
 * k-d tree.
 */
struct LBATreeBuilder {
    Box tree_bounds;
    std::vector<Attribute> attribs;
    std::vector<LBuildPoint> build_points;
    // The inner nodes of the coarse bottom-up k-d tree
    std::vector<RadixTreeNode> inner_nodes;
    // Morton codes for each leaf
    std::vector<uint32_t> morton_codes;
    // The treelets built within each leaf of the coarse k-d tree
    std::vector<LBATreelet> treelets;

    uint32_t kd_morton_bits = 4;
    uint32_t kd_morton_mask = 0;
    uint32_t morton_bits = 10;

    LBATreeBuilder(std::vector<glm::vec3> points,
                   std::vector<Attribute> attributes = std::vector<Attribute>{});

    BATree compact();

private:
    uint32_t compact_tree(uint32_t n,
                          std::shared_ptr<OwnedArray<KdNode>> &nodes,
                          std::shared_ptr<OwnedArray<glm::vec3>> &points);

    uint32_t compact_leaf(uint32_t n,
                          std::shared_ptr<OwnedArray<KdNode>> &nodes,
                          std::shared_ptr<OwnedArray<glm::vec3>> &points);

    void build_treelets();
};

