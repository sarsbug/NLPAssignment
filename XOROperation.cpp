//
// Created by zxz on 2019/11/10.
//

#include "XOROperation.h"
#include "../../tensor/function/FHeader.h"
#include <iostream>

using namespace std;

namespace xorop{

    float learningRate = 0.1F;           // learning rate
    int nEpoch = 100;                      // max training epochs
    float minmax = 0.01F;                 // range [-p,p] for parameter initialization

    int N = 7;  //产生的测试数据为0~7两两异或
    int M = 4;  //每个输入表示成二进制，用int[4]存储
    void decToBin(int num,int* decNum);
    void generate_train_data(int** trainX,int** trainY);
    void show_num(int* num,int length);
    void show_result(int* num1,int* num2,int* num,int* result);
    void convertIntToFloat(int** X,int** Y,float** trainX,float** trainY);

    void Init(XORModel &model);
    void Train(int** trainX, int**trainY, int dataSize, XORModel &model);
    void InitGrad(XORModel &model, XORModel &grad);
    void Forword(XTensor &input, XORModel &model, XORNet &net);
    void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
    void Backward(XTensor &input, XTensor &gold, XORModel &model, XORModel &grad, XORNet &net);
    void Update(XORModel &model, XORModel &grad, float learningRate);
    void CleanGrad(XORModel &grad);
    void Test(int **testData, int testDataSize, XORModel &model);

    int XORMain(int argc, const char ** argv)
    {
        int** trainX = new int*[N*N];
        int** trainY = new int*[N*N];
        generate_train_data(trainX,trainY);
        //float** trainX = new float*[N*N];
        //float** trainY = new float*[N];
        //convertIntToFloat(X,Y,trainX,trainY);

        /*
        for(int i=0;i<N*N;i++){
            cout<<"trainX["<<i<<"]";
            show_num(trainX[i],2*M);
            cout<<"trainX["<<i<<"]";
            show_num(trainY[i],M);
        }*/

        XORModel model;
        model.h_size = 5;
        const int dataSize = N*N;
        //const int testDataSize = 3;
        model.devID = 0;
        Init(model);


        Train(trainX, trainY, dataSize, model);

        int testX[M] = {0,1,0,1,0,0,1,1};
        Test(trainX, 1, model);

        delete[] trainX;
        delete[] trainY;
        return 0;
    }

    void convertIntToFloat(int** X,int** Y,float** trainX,float** trainY){
        for(int i=0;i<N*N;i++){
            for(int j=0;j<2*M;j++){
                trainX[i][j] = float(X[i][j]);
            }
            for(int j=0;j<M;j++){
                trainY[i][j] = float(Y[i][j]);
            }
        }
    }

    void decToBin(int num,int* decNum){
        int i,j=0;
        i=num;
        while(i){
            decNum[j]=i%2;
            i/=2;
            j++;
        }
    }

    void generate_train_data(int** trainX,int** trainY){
        int index = 0;
        for(int i=0;i<N;i++){
            for(int k=0;k<N;k++){
                int r = i^k;

                int* num1 = new int[M];
                decToBin(i,num1);
                int* num2 = new int[M];
                decToBin(k,num2);
                int* result = new int[M];
                decToBin(r,result);

                int* num = new int[2*M];
                int size = M*sizeof(int);
                memcpy(num,num1,size);
                memcpy(num+M,num2,size);

                trainX[index] = new int[2*M];
                trainY[index] = new int[M];
                memcpy(trainX[index],num,2*M*sizeof(int));
                memcpy(trainY[index],result,M*sizeof(int));
                /*
                if(index==48){
                    show_result(num1,num2,num,result);
                    cout<<"trainX[index]";
                    show_num(trainX[index],2*M);
                }*/
                index ++;
            }
        }
        //cout<<index<<endl;
    }

    void show_num(int* num,int length){
        for(int j=0;j<length;j++){
            cout<<num[length-j-1];
        }
        cout<<endl;
    }

    void show_result(int* num1,int* num2,int* num,int* result){
        cout<<"num1:";
        show_num(num1,M);
        cout<<"num2:";
        show_num(num2,M);
        cout<<"cat num1 and num2:";
        show_num(num,2*M);
        cout<<"xor result:";
        show_num(result,M);
    }

    void Init(XORModel &model)
    {
        InitTensor2D(&model.weight1, 2*M, model.h_size, X_FLOAT, model.devID);
        InitTensor2D(&model.weight2, model.h_size, M, X_FLOAT, model.devID);
        InitTensor2D(&model.b, 1,model.h_size, X_FLOAT, model.devID);
        model.weight1.SetDataRand(-minmax, minmax);
        model.weight2.SetDataRand(-minmax, minmax);
        model.b.SetZeroAll();
        printf("Init model finish!\n");
    }

    void Train(int** trainX, int** trainY, int dataSize, XORModel &model)
    {
        printf("prepare data for train\n");
        /*prepare for train*/
        TensorList inputList;
        TensorList goldList;
        for (int i = 0; i < dataSize; ++i)
        {
            XTensor*  inputData = NewTensor2D(1, 2*M, X_FLOAT, model.devID);
            for(int j=0;j<2*M;j++){
                inputData->Set2D(float(trainX[i][j]), 0, j);
            }
            inputList.Add(inputData);
            XTensor*  goldData = NewTensor2D(1, M, X_FLOAT, model.devID);
            for(int j=0;j<M;j++){
                goldData->Set2D(float(trainY[i][j]), 0, j);
            }
            goldList.Add(goldData);
        }

        printf("start train\n");
        XORNet net;
        XORModel grad;
        InitGrad(model, grad);
        for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
        {
            cout<<"epoch:"<<epochIndex;
            float totalLoss = 0;
            if ((epochIndex + 1) % 50 == 0)
                learningRate /= 3;
            for (int i = 0; i < inputList.count; ++i)
            {
                XTensor *input = inputList.GetItem(i);
                XTensor *gold = goldList.GetItem(i);

                Forword(*input, model, net);

                //output.Dump(stderr);
                XTensor loss;
                MSELoss(net.output, *gold, loss);

                //loss.Dump(stderr);
                totalLoss += loss.Get1D(0);

                Backward(*input, *gold, model, grad, net);

                Update(model, grad, learningRate);

                CleanGrad(grad);

            }
            cout<<" loss:"<<totalLoss / inputList.count<<endl;
        }
    }

    void InitGrad(XORModel &model, XORModel &grad)
    {
        InitTensor(&grad.weight1, &model.weight1);
        InitTensor(&grad.weight2, &model.weight2);
        InitTensor(&grad.b, &model.b);

        grad.h_size = model.h_size;
        grad.devID = model.devID;
    }

    void Forword(XTensor &input, XORModel &model, XORNet &net)
    {

        net.hidden_state1 = MatrixMul(input, model.weight1);

        net.hidden_state2 = net.hidden_state1 + model.b;
        net.hidden_state3 = HardTanH(net.hidden_state2);

        net.output = MatrixMul(net.hidden_state3, model.weight2);
    }

    void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)
    {
        XTensor tmp = output - gold;
        loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
    }

    void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad)
    {
        XTensor tmp = output - gold;
        grad = tmp * 2;
    }

    void Backward(XTensor &input, XTensor &gold, XORModel &model, XORModel &grad, XORNet &net)
    {
        XTensor lossGrad;
        XTensor &dedw2 = grad.weight2;
        XTensor &dedb = grad.b;
        XTensor &dedw1 = grad.weight1;

        MSELossBackword(net.output, gold, lossGrad);

        MatrixMul(net.hidden_state3, X_TRANS, lossGrad, X_NOTRANS, dedw2);

        XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.weight2, X_TRANS);

        _HardTanHBackward(&net.hidden_state3, &net.hidden_state2, &dedy, &dedb);

        dedw1 = MatrixMul(input, X_TRANS, dedb, X_NOTRANS);
    }

    void Update(XORModel &model, XORModel &grad, float learningRate)
    {
        model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
        model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
        model.b = Sum(model.b, grad.b, -learningRate);
    }

    void CleanGrad(XORModel &grad)
    {
        grad.b.SetZeroAll();
        grad.weight1.SetZeroAll();
        grad.weight2.SetZeroAll();
    }

    void Test(int **testData, int testDataSize, XORModel &model)
    {
        XORNet net;
        XTensor*  inputData = NewTensor2D(1, 2*M, X_FLOAT, model.devID);
        for (int i = 0; i < testDataSize; ++i)
        {
            for(int j=0;j<2*M;j++){
                cout<<"-"<<endl;
                inputData->Set2D(testData[i][j], 0, j);
                cout<<"+"<<endl;
            }
            cout<<"1"<<endl;
            Forword(*inputData, model, net);
            cout<<"2"<<endl;
            float ans = net.output.Get2D(0,M) ;

            cout<<"3"<<endl;
            //for(int k=0;k<M;k++){
            //    cout<<ans[k];
            //}
            cout<<ans<<endl;
        }

    }
};