# NLPAssignment

项目介绍：用Niutensor，训练一个前馈神经网络，实现异或的功能

实现细节：

1.生成训练数据，我生成了0～N内数字两两组合，然后做异或。两两组和的数字和异或的结果都转换成了二进制，并用M个int来存储，其中trainX为N*Nx2*M维，trainY为N*NxM维。

2.构建神经网络 输入层:2*M,隐藏层：H,输出层:M

3.训练网络

4.测试结果


完成程度:

//测试结果，目前只测试了训练集，正确率只有35%,还在查找问题

隐藏层调为10后，训练集正确率达到了100%

测试集正确率100%

训练后的损失

![训练后的损失](https://github.com/sarsbug/NLPAssignment/blob/master/results/1.png)

训练集前5个结果及全部数据的正确率

![训练集前5个结果及全部数据的正确率](https://github.com/sarsbug/NLPAssignment/blob/master/results/2.png)

测试集前5个结果及全部数据的正确率

![测试集前5个结果及全部数据的正确率](https://github.com/sarsbug/NLPAssignment/blob/master/results/3.png)



意见建议:

1.Linux下，写完代码，make编译过程比较慢

2.如果make编译完执行./Niutensor.GPU -xxx如果程序内部再报错的话，不会有具体错误显示，只显示为'段错误 (核心已转储)'

3.可以多加一些常用的方法，方便调用，比如多维数组的一些操作，如打印数组
