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

#include "tensor_spec.h"

/***  Module Header  ******************************************************}}}*/
/**
* constructor
* @par DESCRIPTION
*   instansiate object.
**/
/*************************************************************************{{{*/
TensorSpec::TensorSpec(const std::string& spec, bool alloc_blob)
    :mDType(DType::DTYPE_NONE), mBlob(nullptr)
{
    if (spec.empty()) {
        return;
    }

    size_t element_size = 0;
    auto offset = std::string::size_type(0);
    auto pos    = spec.find(',', offset);
    mName = spec.substr(offset, pos-offset);
    offset = pos + 1;

    pos = spec.find(',', offset);
    std::string chunk = spec.substr(offset, pos-offset);
    if (chunk == "u8") {
        mDType = DType::DTYPE_U8;
        element_size = 1;
    }
    else if (chunk == "i32") {
        mDType = DType::DTYPE_I32;
        element_size = 4;
    }
    else if (chunk == "f32") {
        mDType = DType::DTYPE_F32;
        element_size = 4;
    }
    else {
        return;
    }
    offset = pos + 1;

    size_t count = 1;
    for (;;) {
        int size;
        pos = spec.find(',', offset);
        if (pos == std::string::npos) {
            size = std::stoi(spec.substr(offset));
            mShape.push_back(size);
            count *= size;
            break;
        }

        size = std::stoi(spec.substr(offset, pos-offset));
        mShape.push_back(size);
        count *= size;
        offset = pos + 1;
    }
    
    if (alloc_blob) {
        mBlob = new uint8_t[count * element_size];
    }
}

/***  Module Header  ******************************************************}}}*/
/**
* destructor
* @par DESCRIPTION
*   destruct object.
**/
/**************************************************************************{{{*/
TensorSpec::~TensorSpec()
{
    if (mBlob != nullptr) {
        delete [] mBlob;
    }
}

/***  Module Header  ******************************************************}}}*/
/**
* parse tensor spec string
* @par DESCRIPTION
*   convert tensor spec string to TensorSpec object.
**/
/**************************************************************************{{{*/
std::vector<TensorSpec*>
parse_tensor_spec(std::string specs, bool alloc_blob)
{
    std::vector<TensorSpec*> tensor_specs;

    if (!specs.empty()) {
        auto offset = std::string::size_type(0);
        for (;;) {
            auto pos = specs.find(':', offset);
            if (pos == std::string::npos) {
                tensor_specs.push_back(new TensorSpec(specs.substr(offset), alloc_blob));
                break;
            }

            tensor_specs.push_back(new TensorSpec(specs.substr(offset, pos - offset), alloc_blob));
            offset = pos + 1;
        }
    }

    return tensor_specs;
}

/***  Module Header  ******************************************************}}}*/
/**
* print tensor spec
* @par DESCRIPTION
*   put tensor spec to std::ostream.
**/
/**************************************************************************{{{*/
std::ostream&
operator<<(std::ostream& s, TensorSpec t)
{
    s << "DType: " << t.mDType << ", Shape: {";
    for (const auto& i : t.mShape) {
        s << i << ",";
    }
    s << "}";
    
    if (!t.mName.empty()) {
        s << ", Name: " << t.mName;
    }

    return s;
}

/*** tensor_spec.h ********************************************************}}}*/
