/***  File Header  ************************************************************/
/**
* @file tf2_interp.h
*
* Tiny ML interpreter on Libtensorflow
* @author   Shozo Fukuda
* @date     create Fri Apr 15 12:55:02 JST 2022
* @date     update $Date:$
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
*******************************************************************************/
#ifndef _TF2_INTERP_H
#define _TF2_INTERP_H

/*--- INCLUDE ---*/
#include <string>
#include <vector>
#include <functional>

#include "tensorflow/c/c_api.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

/*--- CONSTANT ---*/

/*--- TYPE ---*/

/***  Class Header  *******************************************************}}}*/
/**
* Tensorflow2 Interpreter
* @par DESCRIPTION
*   Tiny ML Interpreter on Libtensorflow
*
**/
/**************************************************************************{{{*/
class Tf2Interp {
friend class Tf2InterpTest;

//CONSTANT:
public:

//LIFECYCLE:
public:
  Tf2Interp(std::string tf2_model, std::string inputs, std::string outputs);
  virtual ~Tf2Interp();

//ACTION:
public:
    void info(json& res);
    int set_input_tensor(unsigned int index, const uint8_t* data, int size);
    int set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv);
    bool invoke();
    std::string get_output_tensor(unsigned int index);

//ACCESSOR:
public:

//INQUIRY:
public:

//ATTRIBUTE:
private:
    TF_Status*   mStatus;
    TF_Graph*    mGraph;
    TF_Session*  mSession;

    size_t mInputCount;
    std::vector<TF_Output>  mInputs;
    std::vector<TF_Tensor*> mInputTensors;

    size_t mOutputCount;
    std::vector<TF_Output>  mOutputs;
    std::vector<TF_Tensor*> mOutputTensors;
};

/*INLINE METHOD:
--$-----------------------------------*/

/*--- MACRO ---*/

/*--- EXTERNAL MODULE ---*/

/*--- EXTERNAL VARIABLE ---*/

#endif /* _TF2_INTERP_H */
