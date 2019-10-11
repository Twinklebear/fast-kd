#include "attribute.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include "abstract_array.h"
#include "data_type.h"
#include "util.h"

Attribute::Attribute(const std::string &name,
                     const std::shared_ptr<AbstractArray<uint8_t>> &data,
                     DTYPE data_type)
    : name(name), data(data), data_type(data_type)
{
}

size_t Attribute::stride() const
{
    return dtype_stride(data_type);
}

size_t Attribute::size() const
{
    return data->size() / stride();
}

const uint8_t *Attribute::at(size_t i) const
{
    return data->data() + i * stride();
}

AttributeQuery::AttributeQuery(const std::string &name)
    : name(name), data(std::make_shared<OwnedArray<uint8_t>>())
{
}

AttributeQuery::AttributeQuery(const std::string &name, const glm::vec2 &range)
    : name(name), data(std::make_shared<OwnedArray<uint8_t>>()), range(range)
{
}

size_t AttributeQuery::stride() const
{
    if (data_type != UNKNOWN) {
        return dtype_stride(data_type);
    }
    return 0;
}

size_t AttributeQuery::size() const
{
    if (data) {
        return data->size() / stride();
    }
    return 0;
}

void AttributeQuery::push_back(const uint8_t *t)
{
    if (data->array.capacity() == data->array.size()) {
        data->array.reserve(data->array.size() * 1.5);
    }
    const size_t offs = data->array.size();
    data->array.resize(data->array.size() + stride(), 0);
    std::memcpy(data->array.data() + offs, t, stride());
}

void AttributeQuery::clear()
{
    data->array.clear();
}

const uint8_t *AttributeQuery::at(size_t i) const
{
    return data->data() + i * stride();
}

glm::vec2 AttributeQuery::compute_range() const
{
    switch (data_type) {
    case INT_8:
        return ::compute_range(reinterpret_cast<const int8_t *>(at(0)),
                               reinterpret_cast<const int8_t *>(at(size() - 1)));
    case UINT_8:
        return ::compute_range(reinterpret_cast<const uint8_t *>(at(0)),
                               reinterpret_cast<const uint8_t *>(at(size() - 1)));
    case INT_16:
        return ::compute_range(reinterpret_cast<const int16_t *>(at(0)),
                               reinterpret_cast<const int16_t *>(at(size() - 1)));
    case UINT_16:
        return ::compute_range(reinterpret_cast<const uint16_t *>(at(0)),
                               reinterpret_cast<const uint16_t *>(at(size() - 1)));
    case INT_32:
        return ::compute_range(reinterpret_cast<const int32_t *>(at(0)),
                               reinterpret_cast<const int32_t *>(at(size() - 1)));
    case UINT_32:
        return ::compute_range(reinterpret_cast<const uint32_t *>(at(0)),
                               reinterpret_cast<const uint32_t *>(at(size() - 1)));
    case FLOAT_32:
        return ::compute_range(reinterpret_cast<const float *>(at(0)),
                               reinterpret_cast<const float *>(at(size() - 1)));
    case FLOAT_64:
        return ::compute_range(reinterpret_cast<const double *>(at(0)),
                               reinterpret_cast<const double *>(at(size() - 1)));
    default:
        break;
    }
    throw std::runtime_error("Unsupported datatype in compute range");
}

