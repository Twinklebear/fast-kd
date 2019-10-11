#include "lba_tree_builder.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_group.h>
#include "owned_array.h"
#include "util.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

LBuildPoint::LBuildPoint(const glm::vec3 &pos,
                         size_t id,
                         uint32_t morton_code,
                         const glm::uvec3 &quantized_pt)
    : pos(pos), id(id), morton_code(morton_code), quantized_pt(quantized_pt)
{
}

std::ostream &operator<<(std::ostream &os, const RadixTreeNode &n)
{
    os << "{left: " << n.left_child << (n.left_leaf ? " (leaf) " : " (inner) ")
       << ", right: " << n.right_child << (n.right_leaf ? " (leaf) " : " (inner) ")
       << ", split axis: " << n.split_axis << ", split pos: " << n.split_pos << "}";
    return os;
}

LBATreelet::LBATreelet(LBuildPoint *points, uint32_t nprims, std::vector<Attribute> *attributes)
    : points(points),
      in_attribs(attributes),
      n_prims(nprims),
      max_depth(8 + 1.3 * std::log2(n_prims))
{
    build(0, n_prims, 0);
}

uint32_t LBATreelet::build(const size_t lo, const size_t hi, const uint32_t depth)
{
    const size_t num_prims = hi - lo;
    if (num_prims <= min_prims || depth >= max_depth) {
        return build_leaf(lo, hi);
    }

    Box centroid_bounds;
    for (auto it = points + lo; it != points + hi; ++it) {
        centroid_bounds.extend(it->pos);
    }

    // Find the median-split position, retrying if we have some weird particle configuration.
    // It seems like in very rare cases this can happen (e.g., the Uintah data set), where a lot
    // particles are all clumped right on the boundary so the median is actually at the lower/upper
    // edge of the box. Try the next longest axis, then the smallest
    const glm::uvec3 axis_order = centroid_bounds.axis_ordering();
    int split_axis = axis_order.x;
    float split_pos = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        split_axis = axis_order[i];

        std::sort(points + lo, points + hi, [&](const LBuildPoint &a, const LBuildPoint &b) {
            return a.pos[split_axis] < b.pos[split_axis];
        });
        split_pos = points[lo + num_prims / 2].pos[split_axis];

        if (split_pos != centroid_bounds.lower[split_axis] &&
            split_pos != centroid_bounds.upper[split_axis]) {
            break;
        }
    }

    auto right_start = std::upper_bound(
        points + lo, points + hi, split_pos, [&](const float &split, const LBuildPoint &p) {
            return split < p.pos[split_axis];
        });

    // Ranges within points array for the left/right child
    const size_t left_lo = lo;
    const size_t left_hi = std::distance(points, right_start);

    const size_t right_lo = left_hi;
    const size_t right_hi = hi;

    const uint32_t inner_idx = nodes.size();
    nodes.push_back(KdNode(split_pos, split_axis));

    // Build the left child following the inner node, and the right node after the left subtree
    build(left_lo, left_hi, depth + 1);
    const uint32_t right_child = build(right_lo, right_hi, depth + 1);
    nodes[inner_idx].set_right_child(right_child);

    return inner_idx;
}

uint32_t LBATreelet::build_leaf(const size_t lo, const size_t hi)
{
    const size_t num_prims = hi - lo;
    const uint32_t index = nodes.size();
    nodes.push_back(KdNode(num_prims, lo));
    // TODO: Here we can also re-order the attributes and compute our min/max range info
    // Then the parent would compute its own min/max and update the child ranges to replace
    // it with the attribute split position and direction.
    return index;
}

void LBATreelet::compact(std::shared_ptr<OwnedArray<KdNode>> &out_nodes,
                         std::shared_ptr<OwnedArray<glm::vec3>> &out_points) const
{
    // We already have the tree in the correct flattened layout, we just need to apply a global
    // offset to the indices used by the nodes to put them into the global array.
    const uint32_t index_offset = out_nodes->size();
    const uint32_t prims_offset = out_points->size();
    std::copy(nodes.begin(), nodes.end(), std::back_inserter(out_nodes->array));
    // Now that all treelets share the build points view, we could just copy it once and then track
    // the primitive offsets instead of having each treelet copy its points independently
    std::transform(
        points, points + n_prims, std::back_inserter(out_points->array), [](const LBuildPoint &p) {
            return p.pos;
        });

    for (size_t i = 0; i < nodes.size(); ++i) {
        KdNode &n = out_nodes->at(i + index_offset);
        if (!n.is_leaf()) {
            const uint32_t right_child_offset = n.right_child_offset();
            n.set_right_child(right_child_offset + index_offset);
        } else {
            n.prim_indices_offset += prims_offset;
        }
    }
}

LBATreeBuilder::LBATreeBuilder(std::vector<glm::vec3> points, std::vector<Attribute> in_attributes)
    : attribs(in_attributes)
{
    using namespace std::chrono;
    std::cout << "Starting KD tree build\n";
    auto start = high_resolution_clock::now();

    tree_bounds = compute_bounds(points.begin(), points.end());
    std::cout << "bounds: " << tree_bounds << "\n";

    // Compute Morton codes for each point
    const uint32_t morton_grid_size = (1 << morton_bits) - 1;
    build_points.resize(points.size(), LBuildPoint());
    tbb::parallel_for(size_t(0), points.size(), [&](const size_t i) {
        const glm::vec3 p =
            morton_grid_size * ((points[i] - tree_bounds.lower) / tree_bounds.diagonal());
        const uint32_t morton_code = encode_morton32(p.x, p.y, p.z);
        build_points[i] = LBuildPoint(points[i], i, morton_code, glm::uvec3(p.x, p.y, p.z));
    });

    tbb::parallel_sort(
        build_points.begin(), build_points.end(), [&](const LBuildPoint &a, const LBuildPoint &b) {
            return a.morton_code < b.morton_code;
        });

    // We build the tree over fewer morton bits to build a coarse tree bottom-up, then build
    // better kd-treelets within the leaves of the tree in parallel
    auto start_unique_find = high_resolution_clock::now();
    // TODO: This could be a parallel compaction
    kd_morton_mask = 0xffffffff << (morton_bits - kd_morton_bits) * 3;
    morton_codes.push_back(build_points[0].morton_code & kd_morton_mask);
    for (size_t i = 1; i < build_points.size(); ++i) {
        const uint32_t key = build_points[i].morton_code & kd_morton_mask;
        if (morton_codes.back() != key) {
            morton_codes.push_back(key);
        }
    }
    auto end_unique_find = high_resolution_clock::now();
    std::cout << "unique find step took: "
              << duration_cast<milliseconds>(end_unique_find - start_unique_find).count() << "ms\n";

    const size_t num_unique_keys = morton_codes.size();
    const size_t num_inner_nodes = num_unique_keys - 1;

    std::cout << "num unique keys: " << num_unique_keys << ", num inner nodes: " << num_inner_nodes
              << "\n";

    // better name/use
    const auto delta = [&](const int i, const int j) {
        if (j < 0 || j > num_inner_nodes) {
            return -1;
        }
        return int(longest_prefix(morton_codes[i], morton_codes[j]));
    };

    // Build the treelets for each leaf node we'll produce of the coarse k-d tree, in
    // parallel with the coarse k-d tree construction
    auto treelets_task = std::async(std::launch::async, [&]() { build_treelets(); });

    inner_nodes.resize(num_inner_nodes, RadixTreeNode{});
    std::vector<uint32_t> parent_pointers(num_inner_nodes, std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> leaf_parent_pointers(num_unique_keys,
                                               std::numeric_limits<uint32_t>::max());
    // Build the inner nodes of our coarse k-d tree
    tbb::parallel_for(size_t(0), num_inner_nodes, [&](const size_t i) {
        const int dir = (delta(i, i + 1) - delta(i, i - 1)) >= 0 ? 1 : -1;
        const int delta_min = delta(i, i - dir);

        // Find upper bound for the length of the range
        int l_max = 2;
        while (delta(i, i + l_max * dir) > delta_min) {
            l_max *= 2;
        }

        // Find the other end of the range using binary search
        int l = 0;
        for (int t = l_max / 2; t > 0; t /= 2) {
            if (delta(i, i + (l + t) * dir) > delta_min) {
                l += t;
            }
        }
        const size_t j = i + l * dir;

        // Find the split position of the node in the array
        const int delta_node = delta(i, j);

        int s = 0;
        for (int t = std::ceil(l / 2.f); t > 0; t = std::ceil(t / 2.f)) {
            if (delta(i, i + (s + t) * dir) > delta_node) {
                s += t;
            }
            if (t == 1) {
                break;
            }
        }
        const size_t gamma = i + s * dir + std::min(dir, 0);

        // Write the child pointers for this inner node
        inner_nodes[i].left_child = gamma;
        inner_nodes[i].right_child = gamma + 1;
        if (std::min(i, j) == gamma) {
            inner_nodes[i].left_leaf = true;
            leaf_parent_pointers[gamma] = i;
        } else {
            inner_nodes[i].left_leaf = false;
            parent_pointers[gamma] = i;
        }

        if (std::max(i, j) == gamma + 1) {
            inner_nodes[i].right_leaf = true;
            leaf_parent_pointers[gamma + 1] = i;
        } else {
            inner_nodes[i].right_leaf = false;
            parent_pointers[gamma + 1] = i;
        }

        // This split plane computation is definitely wrong
        const uint32_t node_prefix = morton_codes[i] & (0xffffffff << (32 - delta_node));
        const uint32_t partition_prefix = node_prefix | (0x1 << (32 - delta_node - 1));

        // Compute split plane position based on the prefix
        uint32_t sx, sy, sz;
        decode_morton32(partition_prefix, sx, sy, sz);
        uint32_t split_pos = 0;
        if (delta_node % 3 == 0) {
            split_pos = sy;
            inner_nodes[i].split_axis = Y;
        } else if (delta_node % 3 == 1) {
            split_pos = sx;
            inner_nodes[i].split_axis = X;
        } else if (delta_node % 3 == 2) {
            split_pos = sz;
            inner_nodes[i].split_axis = Z;
        }

        // TODO: Seems like the split planes on nanosphere aren't quite right? or the layout
        // after compaction isn't or has the wrong split info? We should migrate to just use the
        // LBVH layout though instead of compacting to the other layout
        inner_nodes[i].split_pos = tree_bounds.diagonal()[inner_nodes[i].split_axis] *
                                       (split_pos / static_cast<float>(morton_grid_size)) +
                                   tree_bounds.lower[inner_nodes[i].split_axis];
    });

    // Sync for the treelets to finish
    treelets_task.wait();

    // TODO: do the bottom-up coarse tree traversal in parallel to propagate
    // the min/max ranges up the tree.

#if 0
    std::vector<std::atomic<uint32_t>> node_touched(num_inner_nodes);
    // Can't do a copy ctor for the atomics, we've got to go init them all
    for (auto &v : node_touched) {
        v = 0;
    }

    std::vector<Box> inner_node_bounds(num_inner_nodes, Box{});

    // Future thing to do: once the treelets are built and have all their info we can traverse
    // up from the coarse leaves and propage the treelet min/max info etc. up to the root
    for (size_t i = 0; i < leaf_parent_pointers.size(); ++i) {
        uint32_t cur_node = i;
        uint32_t parent = leaf_parent_pointers[cur_node];
        // std::cout << "leaf " << cur_node << " parent is: " << parent
        //          << ", particle: " << glm::to_string(build_points[cur_node].pos) << "\n";
        do {
            // std::cout << "node " << cur_node << " parent is: " << parent << "\n";
            uint32_t count = node_touched[parent].fetch_add(1);
            // If we're the second to reach this node, compute its bounds from the children
            if (count == 1) {
                const RadixTreeNode &node = inner_nodes[parent];
                Box left_bounds, right_bounds;
                if (node.left_leaf) {
                    left_bounds.extend(build_points[node.left_child].pos);
                } else {
                    left_bounds.box_union(inner_node_bounds[node.left_child]);
                }
                if (node.right_leaf) {
                    right_bounds.extend(build_points[node.right_child].pos);
                } else {
                    right_bounds.box_union(inner_node_bounds[node.right_child]);
                }

                inner_node_bounds[parent] = box_union(left_bounds, right_bounds);
            } else {
                // If we're the first to reach the node the work below this one is incomplete
                // and we terminate
                break;
            }
            cur_node = parent;
            parent = parent_pointers[cur_node];
            // The root will have no parent, so terminate once we've processed it
        } while (parent != std::numeric_limits<uint32_t>::max());
        // std::cout << "=======\n";
    }
#endif

    std::cout << "Avg. # prims/treelet: "
              << static_cast<float>(build_points.size()) / num_unique_keys << "\n";

    auto end = high_resolution_clock::now();
    auto dur = duration_cast<milliseconds>(end - start);
    std::cout << "Tree build on " << build_points.size() << " points took " << dur.count() << "ms ("
              << build_points.size() / (dur.count() * 1e-3f) << "prim/s)\n";
}

BATree LBATreeBuilder::compact()
{
    using namespace std::chrono;
    std::cout << "Starting tree compaction\n";
    auto start = high_resolution_clock::now();

    // For testing/validation: compact the tree to match the BATree layout
    auto nodes = std::make_shared<OwnedArray<KdNode>>();
    nodes->reserve(inner_nodes.size() + morton_codes.size());

    auto points = std::make_shared<OwnedArray<glm::vec3>>();
    points->reserve(build_points.size());

    compact_tree(0, nodes, points);

    // TODO: dumping for now for testing
    attribs.clear();
    // TODO: This would no longer be computed just during compaction
    auto attrib_ranges = std::make_shared<OwnedArray<glm::vec2>>();

    auto end = high_resolution_clock::now();
    auto dur = duration_cast<milliseconds>(end - start);
    std::cout << "Tree compaction took " << dur.count() << "ms\n";

    return BATree(tree_bounds,
                  std::dynamic_pointer_cast<AbstractArray<glm::vec3>>(points),
                  std::dynamic_pointer_cast<AbstractArray<KdNode>>(nodes),
                  std::dynamic_pointer_cast<AbstractArray<glm::vec2>>(attrib_ranges),
                  attribs);
}

uint32_t LBATreeBuilder::compact_tree(uint32_t n,
                                      std::shared_ptr<OwnedArray<KdNode>> &nodes,
                                      std::shared_ptr<OwnedArray<glm::vec3>> &points)
{
    const RadixTreeNode &rnode = inner_nodes[n];
    const uint32_t index = nodes->size();
    nodes->push_back(KdNode(rnode.split_pos, rnode.split_axis));
    if (!rnode.left_leaf) {
        compact_tree(rnode.left_child, nodes, points);
    } else {
        compact_leaf(rnode.left_child, nodes, points);
    }

    uint32_t right_child = 0;
    if (!rnode.right_leaf) {
        right_child = compact_tree(rnode.right_child, nodes, points);
    } else {
        right_child = compact_leaf(rnode.right_child, nodes, points);
    }
    nodes->at(index).set_right_child(right_child);

    return index;
}

uint32_t LBATreeBuilder::compact_leaf(uint32_t n,
                                      std::shared_ptr<OwnedArray<KdNode>> &nodes,
                                      std::shared_ptr<OwnedArray<glm::vec3>> &points)
{
    const uint32_t index = nodes->size();
    treelets[n].compact(nodes, points);
    return index;
}

void LBATreeBuilder::build_treelets()
{
    treelets.resize(morton_codes.size());
    tbb::parallel_for(size_t(0), morton_codes.size(), [&](const size_t i) {
        // Find the points falling into the leaf's bin
        const uint32_t leaf_bin = morton_codes[i] & kd_morton_mask;
        auto prims = std::equal_range(build_points.begin(),
                                      build_points.end(),
                                      leaf_bin,
                                      [&](const uint32_t &a, const uint32_t &b) {
                                          return (a & kd_morton_mask) < (b & kd_morton_mask);
                                      });

        treelets[i] =
            LBATreelet(&(*prims.first), std::distance(prims.first, prims.second), &attribs);
    });
}

