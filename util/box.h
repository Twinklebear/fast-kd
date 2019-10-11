#pragma once
#include <ostream>
#include <glm/glm.hpp>

enum AXIS { X = 0, Y = 1, Z = 2 };

#pragma pack(push, 1)
struct Box {
    glm::vec3 lower, upper;

    Box();

    Box(const glm::vec3 &lower, const glm::vec3 &upper);

    void extend(const glm::vec3 &p);

    void box_union(const Box &b);

    bool overlaps(const Box &b);

    bool contains_point(const glm::vec3 &p) const;

    AXIS longest_axis() const;

    // Return the axis indices ordered by largest to smallest
    glm::uvec3 axis_ordering() const;

    glm::vec3 center() const;

    glm::vec3 diagonal() const;
};
#pragma pack(pop)

std::ostream &operator<<(std::ostream &os, const Box &b);

inline Box box_union(const Box &a, const Box &b)
{
    Box r = a;
    r.box_union(b);
    return r;
}

inline Box box_extend(const Box &b, const glm::vec3 &v)
{
    Box r = b;
    r.extend(v);
    return r;
}
