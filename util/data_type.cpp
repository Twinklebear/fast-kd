#include "data_type.h"
#include <stdexcept>
#include <string>

std::string print_data_type(DTYPE type)
{
    switch (type) {
    case INT_8:
        return "INT_8";
    case UINT_8:
        return "UINT_8";
    case INT_16:
        return "INT_16";
    case UINT_16:
        return "UINT_16";
    case INT_32:
        return "INT_32";
    case UINT_32:
        return "UINT_32";
    case INT_64:
        return "INT_64";
    case UINT_64:
        return "UINT_64";
    case FLOAT_32:
        return "FLOAT_32";
    case FLOAT_64:
        return "FLOAT_64";
    case VEC2_I8:
        return "VEC2_I8";
    case VEC2_U8:
        return "VEC2_U8";
    case VEC2_I16:
        return "VEC2_I16";
    case VEC2_U16:
        return "VEC2_U16";
    case VEC2_I32:
        return "VEC2_I32";
    case VEC2_U32:
        return "VEC2_U32";
    case VEC2_FLOAT:
        return "VEC2_FLOAT";
    case VEC2_DOUBLE:
        return "VEC2_DOUBLE";
    case VEC3_I8:
        return "VEC3_I8";
    case VEC3_U8:
        return "VEC3_U8";
    case VEC3_I16:
        return "VEC3_I16";
    case VEC3_U16:
        return "VEC3_U16";
    case VEC3_I32:
        return "VEC3_I32";
    case VEC3_U32:
        return "VEC3_U32";
    case VEC3_FLOAT:
        return "VEC3_FLOAT";
    case VEC3_DOUBLE:
        return "VEC3_DOUBLE";
    case VEC4_I8:
        return "VEC4_I8";
    case VEC4_U8:
        return "VEC4_U8";
    case VEC4_I16:
        return "VEC4_I16";
    case VEC4_U16:
        return "VEC4_U16";
    case VEC4_I32:
        return "VEC4_I32";
    case VEC4_U32:
        return "VEC4_U32";
    case VEC4_FLOAT:
        return "VEC4_FLOAT";
    case VEC4_DOUBLE:
        return "VEC4_DOUBLE";
    case MAT2_I8:
        return "MAT2_I8";
    case MAT2_U8:
        return "MAT2_U8";
    case MAT2_I16:
        return "MAT2_I16";
    case MAT2_U16:
        return "MAT2_U16";
    case MAT2_I32:
        return "MAT2_I32";
    case MAT2_U32:
        return "MAT2_U32";
    case MAT2_FLOAT:
        return "MAT2_FLOAT";
    case MAT2_DOUBLE:
        return "MAT2_DOUBLE";
    case MAT3_I8:
        return "MAT3_I8";
    case MAT3_U8:
        return "MAT3_U8";
    case MAT3_I16:
        return "MAT3_I16";
    case MAT3_U16:
        return "MAT3_U16";
    case MAT3_I32:
        return "MAT3_I32";
    case MAT3_U32:
        return "MAT3_U32";
    case MAT3_FLOAT:
        return "MAT3_FLOAT";
    case MAT3_DOUBLE:
        return "MAT3_DOUBLE";
    case MAT4_I8:
        return "MAT4_I8";
    case MAT4_U8:
        return "MAT4_U8";
    case MAT4_I16:
        return "MAT4_I16";
    case MAT4_U16:
        return "MAT4_U16";
    case MAT4_I32:
        return "MAT4_I32";
    case MAT4_U32:
        return "MAT4_U32";
    case MAT4_FLOAT:
        return "MAT4_FLOAT";
    case MAT4_DOUBLE:
        return "MAT4_DOUBLE";
    default:
        return "UNKNOWN";
    }
}

size_t dtype_stride(DTYPE type)
{
    switch (type) {
    case INT_8:
    case UINT_8:
    case VEC2_I8:
    case VEC2_U8:
    case VEC3_I8:
    case VEC3_U8:
    case VEC4_I8:
    case VEC4_U8:
    case MAT2_I8:
    case MAT2_U8:
    case MAT3_I8:
    case MAT3_U8:
    case MAT4_I8:
    case MAT4_U8:
        return dtype_components(type);
    case INT_16:
    case UINT_16:
    case VEC2_I16:
    case VEC2_U16:
    case VEC3_I16:
    case VEC3_U16:
    case VEC4_I16:
    case VEC4_U16:
    case MAT2_I16:
    case MAT2_U16:
    case MAT3_I16:
    case MAT3_U16:
    case MAT4_I16:
    case MAT4_U16:
        return dtype_components(type) * 2;
    case INT_32:
    case UINT_32:
    case VEC2_I32:
    case VEC2_U32:
    case VEC3_I32:
    case VEC3_U32:
    case VEC4_I32:
    case VEC4_U32:
    case MAT2_I32:
    case MAT2_U32:
    case MAT3_I32:
    case MAT3_U32:
    case MAT4_I32:
    case MAT4_U32:
    case FLOAT_32:
    case VEC2_FLOAT:
    case VEC3_FLOAT:
    case VEC4_FLOAT:
    case MAT2_FLOAT:
    case MAT3_FLOAT:
    case MAT4_FLOAT:
        return dtype_components(type) * 4;
    case FLOAT_64:
    case UINT_64:
    case INT_64:
    case VEC2_DOUBLE:
    case VEC3_DOUBLE:
    case VEC4_DOUBLE:
    case MAT2_DOUBLE:
    case MAT3_DOUBLE:
    case MAT4_DOUBLE:
        return dtype_components(type) * 8;
    default:
        throw std::runtime_error("No stride for unknown DTYPE");
    }
}

size_t dtype_components(DTYPE type)
{
    switch (type) {
    case INT_8:
    case UINT_8:
    case INT_16:
    case UINT_16:
    case INT_32:
    case UINT_32:
    case INT_64:
    case UINT_64:
    case FLOAT_32:
    case FLOAT_64:
        return 1;
    case VEC2_I8:
    case VEC2_U8:
    case VEC2_I16:
    case VEC2_U16:
    case VEC2_I32:
    case VEC2_U32:
    case VEC2_FLOAT:
    case VEC2_DOUBLE:
        return 2;
    case VEC3_I8:
    case VEC3_U8:
    case VEC3_I16:
    case VEC3_U16:
    case VEC3_I32:
    case VEC3_U32:
    case VEC3_FLOAT:
    case VEC3_DOUBLE:
        return 3;
    case VEC4_I8:
    case VEC4_U8:
    case VEC4_I16:
    case VEC4_U16:
    case VEC4_I32:
    case VEC4_U32:
    case VEC4_FLOAT:
    case VEC4_DOUBLE:
    case MAT2_I8:
    case MAT2_U8:
    case MAT2_I16:
    case MAT2_U16:
    case MAT2_I32:
    case MAT2_U32:
    case MAT2_FLOAT:
    case MAT2_DOUBLE:
        return 4;
    case MAT3_I8:
    case MAT3_U8:
    case MAT3_I16:
    case MAT3_U16:
    case MAT3_I32:
    case MAT3_U32:
    case MAT3_FLOAT:
    case MAT3_DOUBLE:
        return 9;
    case MAT4_I8:
    case MAT4_U8:
    case MAT4_I16:
    case MAT4_U16:
    case MAT4_I32:
    case MAT4_U32:
    case MAT4_FLOAT:
    case MAT4_DOUBLE:
        return 16;
    default:
        throw std::runtime_error("No component count for unknown DTYPE");
    }
}

