/***  File Header  ************************************************************/
/**
* @file tensor_spec.h
*
* Tensor spec object holds dtype and shape.
* @author	Shozo Fukuda
* @date	    create Mon Sep 12 21:53:27 2022
* System	Windows,P10RC <br>
*
**/
/**************************************************************************{{{*/
#ifndef _TENSOR_SPEC_H
#define _TENSOR_SPEC_H

#include <string>
#include <vector>
#include <ostream>

/***  Class Header  *******************************************************}}}*/
/**
* Tensor spec
* @par DESCRIPTION
*   holder: dtype and shape of the tensor
**/
/**************************************************************************{{{*/
struct TensorSpec {
//TYPE:
    enum DType {
      DTYPE_NONE = 0,
      DTYPE_F32,
      DTYPE_U8,
      DTYPE_I8,
      DTYPE_U16,
      DTYPE_I16,
      DTYPE_I32,
    };

//LIFECYCLE:
    TensorSpec(const std::string& spec, bool alloc_blob);
    ~TensorSpec();

//INQUIRY:
    size_t count() {
        size_t sum = 0;
        for (const auto& item : mShape) {
            sum += item;
        }
        return sum;
    }

    size_t byte_size() {
        size_t size;

        switch (mDType) {
        case DTYPE_F32: case DTYPE_I32:
            size = 4;
            break;
        case DTYPE_U16: case DTYPE_I16:
            size = 2;
            break;
        case DTYPE_U8:  case DTYPE_I8:
            size = 1;
            break;
        default:
            size = 0;
            break;
        }

        for (const auto& dim : mShape) {
            size *= dim;
        }

        return size;
    }

//ATTRIBUTE:
    DType                mDType;
    std::string          mName;
    std::vector<int64_t> mShape;
    uint8_t*             mBlob;
};

std::vector<TensorSpec*> parse_tensor_spec(std::string specs, bool alloc_blob=false);

std::ostream& operator<<(std::ostream& s, TensorSpec t);

#endif /* _TENSOR_SPEC_H */
/*** tensor_spec.h ********************************************************}}}*/