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
