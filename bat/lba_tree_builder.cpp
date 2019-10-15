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

LBATreelet::LBATreelet(LBuildPoint *points,
                       uint32_t treelet_prim_offset,
                       uint32_t treelet_prims,
                       const std::vector<Attribute> *in_attributes)
    : points(points),
      in_attribs(in_attributes),
      treelet_prim_offset(treelet_prim_offset),
      treelet_prims(treelet_prims),
      max_depth(8 + 1.3 * std::log2(treelet_prims))
{
    if (in_attribs) {
        for (const auto &a : *in_attribs) {
            auto arr = std::make_shared<OwnedArray<uint8_t>>(a.stride() * treelet_prims);
            attributes.push_back(Attribute(a.name, arr, a.data_type));
        }
    }
    build(treelet_prim_offset, treelet_prim_offset + treelet_prims, 0);
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
    nodes.push_back(KdNode::inner(split_pos, static_cast<AXIS>(split_axis)));

    // Reserve the attribute range indices in the array for the inner node's ranges
    if (in_attribs) {
        for (size_t i = 0; i < in_attribs->size(); ++i) {
            attribute_ranges.push_back(glm::vec2(std::numeric_limits<float>::infinity(),
                                                 -std::numeric_limits<float>::infinity()));
        }
    }

    // Build the left child following the inner node, and the right node after the left subtree
    build(left_lo, left_hi, depth + 1);
    const uint32_t right_child = build(right_lo, right_hi, depth + 1);
    nodes[inner_idx].set_right_child(right_child);

    // Merge the ranges of the child nodes to the parent
    if (in_attribs) {
        const size_t left_start = (inner_idx + 1) * in_attribs->size();
        const size_t right_start = right_child * in_attribs->size();
        for (size_t i = 0; i < in_attribs->size(); ++i) {
            glm::vec2 &r = attribute_ranges[inner_idx * in_attribs->size() + i];
            const glm::vec2 &left = attribute_ranges[left_start + i];
            const glm::vec2 &right = attribute_ranges[right_start + i];

            r.x = std::min(left.x, right.x);
            r.y = std::max(left.y, right.y);
        }
    }
    return inner_idx;
}

uint32_t LBATreelet::build_leaf(const size_t lo, const size_t hi)
{
    const uint32_t num_prims = hi - lo;
    const uint32_t index = nodes.size();
    nodes.push_back(KdNode::leaf(num_prims, lo));

    // Now re-order the attributes and compute our min/max range info for the leaf, which
    // will be propagated up the treelet to the parent.
    if (in_attribs) {
        const auto &inattr = *in_attribs;
        for (size_t i = 0; i < inattr.size(); ++i) {
            glm::vec2 range(std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity());
            const size_t stride = inattr[i].stride();
            for (uint32_t j = lo; j < hi; ++j) {
                std::memcpy(
                    attributes[i].at(j - treelet_prim_offset), inattr[i].at(points[j].id), stride);
                update_range(attributes[i], j - treelet_prim_offset, range);
            }
            attribute_ranges.push_back(range);
        }
    }
    return index;
}

void LBATreelet::compact(std::shared_ptr<OwnedArray<KdNode>> &out_nodes,
                         std::shared_ptr<OwnedArray<glm::vec3>> &out_points,
                         std::shared_ptr<OwnedArray<glm::vec2>> &out_ranges,
                         std::vector<Attribute> &compact_attribs) const
{
    // We already have the tree in the correct flattened layout, we just need to apply a global
    // offset to the indices used by the nodes to put them into the global array.
    const uint32_t index_offset = out_nodes->size();
    const uint32_t prims_offset = out_points->size();
    std::copy(nodes.begin(), nodes.end(), std::back_inserter(out_nodes->array));
    std::copy(
        attribute_ranges.begin(), attribute_ranges.end(), std::back_inserter(out_ranges->array));

    // Copy our points into the compacted points list
    std::transform(points + treelet_prim_offset,
                   points + treelet_prim_offset + treelet_prims,
                   std::back_inserter(out_points->array),
                   [](const LBuildPoint &p) { return p.pos; });

    // Copy the attributes into the corresponding compacted buffers, each child keeps its
    // own copy of the re-ordered attributes so we can just directly overwrite the original input
    for (size_t i = 0; i < attributes.size(); ++i) {
        std::memcpy(compact_attribs[i].at(prims_offset),
                    attributes[i].at(0),
                    attributes[i].data->size_bytes());
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        KdNode &n = out_nodes->at(i + index_offset);
        if (!n.is_leaf()) {
            const uint32_t right_child_offset = n.right_child_offset();
            n.set_right_child(right_child_offset + index_offset);
        } else {
            n.prim_indices_offset = n.prim_indices_offset - treelet_prim_offset + prims_offset;
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
        // TODO: seems like sometimes we get the child info wrong here? or is that b/c the attrib
        // out of bounds mem write bug?
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

    // Now propagate the attribute min/max values up the tree to the root
    std::vector<std::atomic<uint32_t>> node_touched(num_inner_nodes);
    // Can't do a copy ctor for the atomics, we've got to go init them all
    for (auto &v : node_touched) {
        v = 0;
    }

    const uint32_t num_attribs = attribs.size();
    attribute_ranges.resize(num_attribs * inner_nodes.size());
    tbb::parallel_for(size_t(0), leaf_parent_pointers.size(), [&](const size_t i) {
        uint32_t cur_node = i;
        uint32_t parent = leaf_parent_pointers[cur_node];
        do {
            uint32_t count = node_touched[parent].fetch_add(1);
            // If we're the second to reach this node, compute its attribute ranges by
            // combining the children
            if (count == 1) {
                const RadixTreeNode &node = inner_nodes[parent];
                for (uint32_t j = 0; j < num_attribs; ++j) {
                    glm::vec2 left_range, right_range;
                    if (node.left_leaf) {
                        // Range of the root node of the treelet
                        left_range = treelets[node.left_child].attribute_ranges[j];
                    } else {
                        left_range = attribute_ranges[node.left_child * num_attribs + j];
                    }
                    if (node.right_leaf) {
                        right_range = treelets[node.right_child].attribute_ranges[j];
                    } else {
                        right_range = attribute_ranges[node.right_child * num_attribs + j];
                    }

                    attribute_ranges[parent * num_attribs + j].x =
                        std::min(left_range.x, right_range.x);
                    attribute_ranges[parent * num_attribs + j].y =
                        std::max(left_range.y, right_range.y);
                }
                cur_node = parent;
                parent = parent_pointers[cur_node];
            } else {
                // If we're the first to reach the node the work below this one is incomplete
                // and we terminate
                break;
            }
            // The root will have no parent, so terminate once we've processed it
        } while (parent != std::numeric_limits<uint32_t>::max());
    });

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

    auto attrib_ranges = std::make_shared<OwnedArray<glm::vec2>>();
    compact_tree(0, nodes, points, attrib_ranges, 0);

    for (size_t i = 0; i < attribs.size(); ++i) {
        attribs[i].range = attrib_ranges->at(i);
    }

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
                                      std::shared_ptr<OwnedArray<glm::vec3>> &points,
                                      std::shared_ptr<OwnedArray<glm::vec2>> &ranges,
                                      uint32_t depth)
{
    const RadixTreeNode &rnode = inner_nodes[n];
    const uint32_t index = nodes->size();
    nodes->push_back(KdNode::inner(rnode.split_pos, rnode.split_axis));
    for (uint32_t i = 0; i < attribs.size(); ++i) {
        ranges->push_back(attribute_ranges[n * attribs.size() + i]);
    }
    if (!rnode.left_leaf) {
        compact_tree(rnode.left_child, nodes, points, ranges, depth + 1);
    } else {
        compact_leaf(rnode.left_child, nodes, points, ranges);
    }

    uint32_t right_child = 0;
    if (!rnode.right_leaf) {
        right_child = compact_tree(rnode.right_child, nodes, points, ranges, depth + 1);
    } else {
        right_child = compact_leaf(rnode.right_child, nodes, points, ranges);
    }
    nodes->at(index).set_right_child(right_child);
    return index;
}

uint32_t LBATreeBuilder::compact_leaf(uint32_t n,
                                      std::shared_ptr<OwnedArray<KdNode>> &nodes,
                                      std::shared_ptr<OwnedArray<glm::vec3>> &points,
                                      std::shared_ptr<OwnedArray<glm::vec2>> &ranges)
{
    // TODO: We also need the attributes and ranges here to compact the treelet's info in
    const uint32_t index = nodes->size();
    treelets[n].compact(nodes, points, ranges, attribs);
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

        const size_t treelet_prims = std::distance(prims.first, prims.second);
        const size_t lo = std::distance(build_points.begin(), prims.first);

        treelets[i] = LBATreelet(build_points.data(), lo, treelet_prims, &attribs);
    });
}

