#pragma once

#include "abstract_array.h"

template <typename T>
class BorrowedArray : public AbstractArray<T> {
    T *array = nullptr;
    size_t sz = 0;

public:
    BorrowedArray() = default;

    BorrowedArray(std::vector<T> &v) : array(v.data()), sz(v.size()) {}

    BorrowedArray(T *arr, size_t sz) : array(arr), sz(sz) {}

    virtual ~BorrowedArray() = default;

    const T &at(const size_t i)
    {
        return array[i];
    }

    const T &operator[](const size_t i) const override
    {
        return array[i];
    }

    const T *data() const override
    {
        return array;
    }

    T *data() override
    {
        return array;
    }

    size_t size() const override
    {
        return sz;
    }

    size_t size_bytes() const override
    {
        return sz * sizeof(T);
    }

    const T *cbegin() const override
    {
        return array;
    }
};


