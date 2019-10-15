#pragma once

#include <memory>
#include <string>
#include "abstract_array.h"
#include "data_type.h"
#include "owned_array.h"
#include <glm/glm.hpp>

struct Attribute {
    std::string name;
    std::shared_ptr<AbstractArray<uint8_t>> data = nullptr;
    DTYPE data_type = UNKNOWN;
    glm::vec2 range =
        glm::vec2(-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());

    Attribute(const std::string &name,
              const std::shared_ptr<AbstractArray<uint8_t>> &data,
              DTYPE data_type);

    Attribute() = default;

    size_t stride() const;

    size_t size() const;

    // Get a raw pointer to the attribute element at index i, applying the data
    // type stride
    uint8_t *at(size_t i);
    const uint8_t *at(size_t i) const;
};

struct AttributeQuery {
    std::string name;
    std::shared_ptr<OwnedArray<uint8_t>> data = nullptr;
    DTYPE data_type = UNKNOWN;
    glm::vec2 range =
        glm::vec2(-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());

    AttributeQuery(const std::string &name);

    AttributeQuery(const std::string &name, const glm::vec2 &range);

    AttributeQuery() = default;

    size_t stride() const;

    size_t size() const;

    void push_back(const uint8_t *t);

    void clear();

    const uint8_t *at(size_t i) const;

    glm::vec2 compute_range() const;
};

inline void update_range(const Attribute &attr, uint32_t id, glm::vec2 &range)
{
    switch (attr.data_type) {
    case INT_8: {
        const int8_t x = *reinterpret_cast<const int8_t *>(attr.at(id));
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
        break;
    }
    case UINT_8: {
        const uint8_t x = *reinterpret_cast<const uint8_t *>(attr.at(id));
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
        break;
    }
    case INT_16: {
        const int16_t x = *reinterpret_cast<const int16_t *>(attr.at(id));
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
        break;
    }
    case UINT_16: {
        const uint16_t x = *reinterpret_cast<const uint16_t *>(attr.at(id));
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
        break;
    }
    case INT_32: {
        const int32_t x = *reinterpret_cast<const int32_t *>(attr.at(id));
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
        break;
    }
    case UINT_32: {
        const uint32_t x = *reinterpret_cast<const uint32_t *>(attr.at(id));
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
        break;
    }
    case FLOAT_32: {
        const float x = *reinterpret_cast<const float *>(attr.at(id));
        range.x = std::min(x, range.x);
        range.y = std::max(x, range.y);
        break;
    }
    case FLOAT_64: {
        const double x = *reinterpret_cast<const double *>(attr.at(id));
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
        break;
    }
    default:
        throw std::runtime_error("range computation on unsupported DTYPE!");
    }
}

