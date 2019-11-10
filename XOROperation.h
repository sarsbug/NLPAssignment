//
// Created by zxz on 2019/11/10.
//

#ifndef NIUTENSOR_MASTER_XOROPERATION_H
#define NIUTENSOR_MASTER_XOROPERATION_H

#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"

using namespace nts;

namespace xorop{
    struct XORModel
    {
        XTensor weight1;
        XTensor weight2;
        XTensor b;
        int h_size;
        int devID;
    };

    struct XORNet
    {
        XTensor hidden_state1;
        XTensor hidden_state2;
        XTensor hidden_state3;
        XTensor output;
    };

    int XORMain(int argc, const char ** argv);
};

#endif //NIUTENSOR_MASTER_XOROPERATION_H
