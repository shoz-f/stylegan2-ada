/***  File Header  ************************************************************/
/**
* tf2_interp.cpp
*
* Tiny ML interpreter on Libtensorflow
* @author   Shozo Fukuda
* @date     create Fri Apr 15 12:55:02 JST 2022
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
* Undocumented Tensorflow C API:
* https://medium.com/@vladislavsd/undocumented-tensorflow-c-api-b527c0b4ef6

* Tensorflow 1 vs Tensorflow 2 C-API:
* https://danishshres.medium.com/tensorflow-1-vs-tensorflow-2-c-api-d94982a5eb16
**/
/**************************************************************************{{{*/

#include <stdio.h>
#include <memory.h>
#include "tensor_spec.h"
#include "tf2_interp.h"


/***  Module Header  ******************************************************}}}*/
/**
* initialize interpreter
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
#if 0
void init_interp(SysInfo& sys, std::string& model, std::string& inputs, std::string& outputs)
{
    sys.mInterp = new Tf2Interp(model, inputs, outputs);
}
#endif

/***  Method Header  ******************************************************}}}*/
/**
* constructor
* @par DESCRIPTION
*   construct an instance.
**/
/**************************************************************************{{{*/
Tf2Interp::Tf2Interp(std::string tf2_model, std::string inputs, std::string outputs)
{
    mStatus  = TF_NewStatus();
    mGraph   = TF_NewGraph();
    mSession = nullptr;

    // load saved model
	const char* tags[] = { "serve" };
	TF_SessionOptions* session_opts = TF_NewSessionOptions();
    mSession = TF_LoadSessionFromSavedModel(session_opts, nullptr, tf2_model.c_str(), tags, 1, mGraph, nullptr, mStatus);
	TF_DeleteSessionOptions(session_opts);
    TF_Code res = TF_GetCode(mStatus);
    if (res != TF_OK) {
	    throw res;
	}

    // conversion table from TensorSpec::DTytpe to TF_DataType.
    const TF_DataType _dtype[] = {
        TF_VARIANT, // DTYPE_NONE
        TF_FLOAT,   // DTYPE_F32
        TF_UINT8,   // DTYPE_U8
        TF_INT8,    // DTYPE_I8
        TF_UINT16,  // DTYPE_U16
        TF_INT16,   // DTYPE_I16
        TF_INT32    // DTYPE_I32
    };

	// prepare input tensors
    std::vector<TensorSpec*> input_spec = parse_tensor_spec(inputs);
    mInputCount = input_spec.size();
    mInputs.resize(mInputCount);
    mInputTensors.resize(mInputCount);
    for (int i = 0; i < mInputCount; i++) {
        TensorSpec* spec = input_spec[i];
        mInputs[i].oper  = TF_GraphOperationByName(mGraph, spec->mName.c_str());
        mInputs[i].index = i;
        mInputTensors[i] = TF_AllocateTensor(_dtype[spec->mDType], spec->mShape.data(), spec->mShape.size(), spec->byte_size());

        delete spec;
    }
    input_spec.clear();

	// prepare output tensors
    std::vector<TensorSpec*> output_spec = parse_tensor_spec(outputs);
    mOutputCount = output_spec.size();
    mOutputs.resize(mOutputCount);
    mOutputTensors.resize(mOutputCount);
    for (int i = 0; i < mOutputCount; i++) {
        TensorSpec* spec = output_spec[i];
        mOutputs[i].oper = TF_GraphOperationByName(mGraph, spec->mName.c_str());
        mOutputs[i].index = i;
        mOutputTensors[i] = TF_AllocateTensor(_dtype[spec->mDType], spec->mShape.data(), spec->mShape.size(), spec->byte_size());

        delete spec;
    }
    output_spec.clear();
}

/***  Method Header  ******************************************************}}}*/
/**
* destructor
* @par DESCRIPTION
*   delate an instance.
**/
/**************************************************************************{{{*/
Tf2Interp::~Tf2Interp()
{
    if (mSession) {
        for (int i = 0; i < mInputCount; i++) {
		    TF_DeleteTensor(mInputTensors[i]);
	    }
	    for (int i = 0; i < mOutputCount; i++) {
		    TF_DeleteTensor(mOutputTensors[i]);
	    }
        TF_DeleteSession(mSession, mStatus);
    }
	TF_DeleteGraph(mGraph);
	TF_DeleteStatus(mStatus);
}

/***  Module Header  ******************************************************}}}*/
/**
* query dimension of input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
void
Tf2Interp::info(json& res)
{
    const std::string _dtype[] = {
        "UNDEFINED",
        "FLOAT",        // maps to c type float
        "DOUBLE",       // maps to c type double
        "INT32",        // maps to c type int32_t
        "UINT8",        // maps to c type uint8_t
        "INT16",        // maps to c type int16_t
        "INT8",         // maps to c type int8_t
        "STRING",       // maps to c++ type std::string
        "COMPLEX64",    // complex with float32 real and imaginary components
        "COMPLEX",
        "INT64",        // maps to c type int64_t
        "BOOL",
        "QINT8",
        "QUINT8",
        "QINT32",
        "BFLOAT16",     // Non-IEEE floating-point format based on IEEE754 single-precision
        "QINT16",
        "QUINT16",
        "UINT16",       // maps to c type uint16_t
        "COMPLEX128",   // complex with float64 real and imaginary components
        "HALF",
        "RESOURCE",
        "VARIANT",
        "UINT32",       // maps to c type uint32_t
        "UINT64"        // maps to c type uint64_t
    };

    int     num_dims;
    int64_t shape[10];

    for (int index = 0; index < mInputCount; index++) {
        json tf2_tensor;
        TF_Output& op = mInputs[index];
        TF_Tensor* t = mInputTensors[index];

        tf2_tensor["index"] = index;
        tf2_tensor["name"] = TF_OperationName(op.oper);

        tf2_tensor["type"] = _dtype[TF_TensorType(t)];
 
        num_dims = TF_GraphGetTensorNumDims(mGraph, op, mStatus);
        if (num_dims > 10) { num_dims = 10; }

        TF_GraphGetTensorShape(mGraph, op, shape, num_dims, mStatus);
        for (int k = 0; k < num_dims; k++) {
            if (shape[k] != -1) {
                tf2_tensor["dims"].push_back(shape[k]);
            }
            else {
                tf2_tensor["dims"].push_back("none");
            }
        }

        res["inputs"].push_back(tf2_tensor);
    }

    for (int index = 0; index < mOutputCount; index++) {
        json tf2_tensor;
        TF_Output& op = mOutputs[index];
        TF_Tensor* t = mOutputTensors[index];

        tf2_tensor["index"] = index;
        tf2_tensor["name"] = TF_OperationName(op.oper);

        tf2_tensor["type"] = _dtype[TF_TensorType(t)];

        num_dims = TF_GraphGetTensorNumDims(mGraph, op, mStatus);
        if (num_dims > 10) { num_dims = 10; }

        TF_GraphGetTensorShape(mGraph, op, shape, num_dims, mStatus);
        for (int k = 0; k < num_dims; k++) {
            if (shape[k] != -1) {
                tf2_tensor["dims"].push_back(shape[k]);
            }
            else {
                tf2_tensor["dims"].push_back("none");
            }
        }

        res["outputs"].push_back(tf2_tensor);
    }
}

/***  Module Header  ******************************************************}}}*/
/**
* set input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
int
Tf2Interp::set_input_tensor(unsigned int index, const uint8_t* data, int size)
{
    if (size == TF_TensorByteSize(mInputTensors[index])) {
        memcpy(TF_TensorData(mInputTensors[index]), data, size);
        return size;
    }
    else {
        return -2;
    }
}

/***  Module Header  ******************************************************}}}*/
/**
* set input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
int
Tf2Interp::set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv)
{
    float* dst = reinterpret_cast<float*>(TF_TensorData(mInputTensors[index]));
    const uint8_t* src = data;
    for (int i = 0; i < size; i++) {
        *dst++ = conv(*src++);
    }

    return size;
}

/***  Module Header  ******************************************************}}}*/
/**
* execute inference
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
bool
Tf2Interp::invoke()
{
    TF_SessionRun(mSession, nullptr, mInputs.data(), mInputTensors.data(), mInputCount, mOutputs.data(), mOutputTensors.data(), mOutputCount, nullptr, 0, nullptr, mStatus);
    return true;
}

/***  Module Header  ******************************************************}}}*/
/**
* get result tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
std::string
Tf2Interp::get_output_tensor(unsigned int index)
{
    return std::string(reinterpret_cast<char*>(TF_TensorData(mOutputTensors[index])), TF_TensorByteSize(mOutputTensors[index]));
}

/*** tf2_interp.cpp ******************************************************}}}*/
