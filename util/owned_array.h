#pragma once

#include <array>
#include <vector>
#include "abstract_array.h"

template <typename T>
class OwnedArray : public AbstractArray<T> {
public:
    std::vector<T> array;

    OwnedArray() = default;

    OwnedArray(const std::vector<T> &array) : array(array) {}

    OwnedArray(std::vector<T> &&array) : array(array) {}

    explicit OwnedArray(size_t size)
    {
        resize(size);
    }

    virtual ~OwnedArray() = default;

    T &at(const size_t i)
    {
        return array[i];
    }

    const T &operator[](const size_t i) const override
    {
        return array[i];
    }

    T &operator[](const size_t i)
    {
        return array[i];
    }

    const T *data() const override
    {
        return array.data();
    }

    T *data() override
    {
        return array.data();
    }

    size_t size() const override
    {
        return array.size();
    }

    size_t size_bytes() const override
    {
        return array.size() * sizeof(T);
    }

    const T *cbegin() const override
    {
        return array.data();
    }

    void reserve(const size_t n)
    {
        array.reserve(n);
    }

    void resize(const size_t n, const T &t = T{})
    {
        array.resize(n, t);
    }

    void push_back(const T &t)
    {
        array.push_back(t);
    }
};

