#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <immintrin.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tbb.h>
#include "box.h"
#include "tinyxml2.h"
#include <glm/glm.hpp>

// Format the bytes as XKB, XMB, etc. depending on the size
std::string format_bytes(size_t nbytes);

// Format the count as #G, #M, #K, depending on its magnitude
std::string pretty_print_count(const double count);

uint64_t align_to(uint64_t val, uint64_t align);

std::string canonicalize_path(const std::string &path);

std::string get_cpu_brand();

// Read the contents of a file into the string
std::string get_file_content(const std::string &fname);

bool compute_divisor(uint32_t x, uint32_t &divisor);

// Compute a 3D grid which has num grid cells
glm::uvec3 compute_grid3d(uint32_t num);

std::string get_file_extension(const std::string &fname);

std::string get_file_basename(const std::string &path);

std::string get_file_basepath(const std::string &path);

bool starts_with(const std::string &str, const std::string &prefix);

std::string tinyxml_error_string(const tinyxml2::XMLError e);

template <typename RandomIt>
Box compute_bounds(RandomIt begin, RandomIt end)
{
    using range_type = tbb::blocked_range<RandomIt>;
    return tbb::parallel_reduce(range_type(begin, end),
                                Box{},
                                [](const range_type &r, const Box &b) {
                                    Box res = b;
                                    for (auto it = r.begin(); it != r.end(); ++it) {
                                        res.extend(*it);
                                    }
                                    return res;
                                },
                                [](const Box &a, const Box &b) { return box_union(a, b); });
}

template <typename RandomIt, typename GetPos>
Box compute_bounds(RandomIt begin, RandomIt end, GetPos get_pos)
{
    using range_type = tbb::blocked_range<RandomIt>;
    return tbb::parallel_reduce(range_type(begin, end),
                                Box{},
                                [&](const range_type &r, const Box &b) {
                                    Box res = b;
                                    for (auto it = r.begin(); it != r.end(); ++it) {
                                        res.extend(get_pos(*it));
                                    }
                                    return res;
                                },
                                [](const Box &a, const Box &b) { return box_union(a, b); });
}

template <typename T>
inline T clamp(const T &x, const T &lo, const T &hi)
{
    if (x < lo) {
        return lo;
    } else if (x > hi) {
        return hi;
    }
    return x;
}

inline float srgb_to_linear(const float x)
{
    if (x <= 0.04045f) {
        return x / 12.92f;
    } else {
        return std::pow((x + 0.055f) / 1.055f, 2.4f);
    }
}

template <typename T>
glm::vec2 compute_range(const T *begin, const T *end)
{
    auto min_max = std::minmax_element(begin, end);
    return glm::vec2(*min_max.first, *min_max.second);
}

inline uint32_t encode_morton32(uint32_t x, uint32_t y, uint32_t z)
{
    // 32   28   24   20   16   12   8    4
    // 00zy xzyx zyxz yxzy xzyx zyxz yxzy xzyx
    // x  0    9    2    4    9    2    4    9
    // y  1    2    4    9    2    4    9    2
    // z  2    4    9    2    4    9    2    4
#define X_MASK 0x09249249
#define Z_MASK 0x24924924
#define Y_MASK 0x12492492
    return _pdep_u32(x, X_MASK) | _pdep_u32(y, Y_MASK) | _pdep_u32(z, Z_MASK);
}

inline void decode_morton32(uint32_t code, uint32_t &x, uint32_t &y, uint32_t &z)
{
#define X_MASK 0x09249249
#define Y_MASK 0x12492492
#define Z_MASK 0x24924924
    x = _pext_u32(code, X_MASK);
    y = _pext_u32(code, Y_MASK);
    z = _pext_u32(code, Z_MASK);
}

inline uint32_t leading_zeros(uint32_t x)
{
    return _lzcnt_u32(x);
}

inline uint32_t longest_prefix(uint32_t a, uint32_t b)
{
    return leading_zeros(a ^ b);
}

