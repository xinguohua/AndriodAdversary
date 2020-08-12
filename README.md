## Research on Android malware detection based on ML models & the weakness of DNNs in adversarial examples.


# 一 Analyze
实验分析文件夹
## CalculateFeature.py
计算全部软件和恶意软件的特征数量，1分位数，均值，3分位数，中位数


# data
数据文件夹

训练数据 测试数据 一行测试数据


# 二 malwareclassification

## nn_grid_search.py

使用训练数据寻找不同架构DNN的最佳参数，使用Scikit-learn封装了Keras方法的GridSearchCV模块进行自动网格搜索，使用3倍交叉验证
返回最佳参数

create_model----200 200 （可以观察所有model 改变createmodel）

选定优化器adam

结果形式
 
    0.9577262851349082 0.002337225030502123 with {'batch_size': 50, 'bias_initializer': 'Zeros', 'dropout_rate': 0.2, 'epochs': 1, 'kernel_initializer': 'uniform', 'learn_rate': 0.001}

参考

[https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/](https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/)

[https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/)

[https://www.davex.pw/2017/09/17/Cross-Validation/](https://www.davex.pw/2017/09/17/Cross-Validation/)
## models_grid_search.py
决策树，随机森林，k最近邻，逻辑回归和支持向量机的最佳参数
DT、RF、KNN、LR、SVM

## train_models.py
我们要训练不同架构的DNN和机器学习模型

DNN模型训练了13种架构

机器学习模型训练了9种

## evaluate_models.py
评价所有机器学习和DNN架构

选出三种架构的DNN架构，准确率高的，FNR低的，介于两者之间的。


# 三 oneFeature文件夹
实施oneFeature攻击 
## tutorials文件夹
过程

* 在oneFeature攻击时
* 先进行deepfoolattack线性攻击（没用）
* 模拟退火oneFeature攻击

### tutorial_oneFeature_k.py
* 进行oneFeature测试 测试一个恶意软件生成一个对抗样本
* 返回扰动特征数量
* 得到对抗样本
### attackall.py
* 进行oneFeature测试 测试整个测试集的恶意软件去生成对抗样本
* 返回平均扰动数量=扰动总数/恶意软件数量
* 针对某一架构DNN的对抗样本存到onefeature_xxx_xxx.csv中



功能
（分别针对不同架构）
* 计算平均扰动
* 得到对抗样本
* 不同攻击fit值变化

1 对于一个攻击
* 计算平均扰动
* 得到对抗样本
* 不同攻击fit值变化（画图）
  * 画两个图  
  * 每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（红线）******遇到问题 量级相差特别大
  * bestvalue的曲线（每次修改特征的bestvalue）(可能要多种攻击进行比较)


2 实现其他攻击的集成，deepfool攻击，fgsm攻击，jsmf攻击,
3 开发一个deepfool+fgsm 集成
4 不同架构DNN
5 开发第二种攻击模式
6  文字部分