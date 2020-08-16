## Research on Android malware detection based on ML models & the weakness of DNNs in adversarial examples.


# 一 Analyze
实验分析文件夹
## CalculateFeature.py
计算全部软件和恶意软件的特征数量，1分位数，均值，3分位数，中位数

## evaluate_adv.py
选定一种分类器（加载一个特定的模型） 看不同攻击的效果（不同数据集）

正常恶意样本+良性软件的准确率，FNR，FPR

评价对抗样本+良性软件的准确率，FNR，FPR

正常和对抗计算误分类率(两个FNR相减)

## characteristicsAnalyse.py
分析不同攻击（样本）下的特征（kd,bu）区分效果
* 读取特征npy文件
* 比较npy文件
>1一个数 <1一个数
# 二 data
数据文件夹

训练数据 x_train01.csv  y_train01.csv

测试数据  x_test01.csv  y_test01.csv 

一行测试数据 one_row.csv

对xxx_xxx架构下攻击得到对抗样本
## jsmf
原始恶意软件及其标签 
JSMF_xxx_xxx_X_normal.csv JSMF_xxx_xxx_Y_normal.csv
jsmf对抗恶意软件及其标签  
JSMF_200_200_X_adv.csv JSMF_200_200_Y_adv.csv

得到原始良性软件及其标签
JSMF_200_200__X_begin.csv JSMF_200_200__Y_begin.csv
## deepfool
原始恶意软件及其标签 
deepfool_xxx_xxx_X_normal.csv deepfool_xxx_xxx_Y_normal.csv
deepfool对抗恶意软件及其标签  
deepfool_200_200_X_adv.csv deepfool_200_200_Y_adv.csv

得到原始良性软件及其标签
deepfool_200_200__X_begin.csv deepfool_200_200__Y_begin.csv
## fgsm
原始恶意软件及其标签 
fgsm_xxx_xxx_X_normal.csv fgsm_xxx_xxx_Y_normal.csv
fgsm对抗恶意软件及其标签  
fgsm_200_200_X_adv.csv fgsm_200_200_Y_adv.csv

得到原始良性软件及其标签
fgsm_200_200__X_begin.csv fgsm_200_200__Y_begin.csv
## fgsm
原始恶意软件及其标签 
onefeature_xxx_xxx_X_normal.csv onefeature_xxx_xxx_Y_normal.csv
onefeature对抗恶意软件及其标签  
onefeature_200_200_X_adv.csv onefeature_200_200_Y_adv.csv

得到原始良性软件及其标签
onefeature_200_200__X_begin.csv onefeature_200_200__Y_begin.csv
## features
feature_names.csv

# 三 malwareclassification

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


# 四 oneFeature文件夹(针对200_200以后换架构)
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
* 画图
	* 过程（单个样本）每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（黑色虚线）      	量级相差特别大，为避免量级相差特别大的原因采用归一化
    * 结果（单个样本）bestvalue的曲线（每次修改特征的bestvalue）（目前只画出一个样本）(可能要多个样本进行比较) 
### oneFeatureattackall.py
* 进行oneFeature测试 测试整个测试集的恶意软件去生成对抗样本
* 返回平均扰动数量=扰动总数/恶意软件数量
* 针对某一架构DNN(到时候csv修改名字)
* 
	得到原始良性软件及其标签onefeature_200_200__X_begin.csv 
	onefeature_200_200__Y_begin.csv
	
	得到原始恶意软件及其标签 onefeature_xxx_xxx_X_normal.csv onefeature_xxx_xxx_Y_normal.csv
	
	得到对抗恶意软件及其标签  onefeature_200_200_X_adv.csv onefeature_200_200_Y_adv.csv

* 画图
	* 加时间戳---过程（单个样本）每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（黑色虚线）      	量级相差特别大，为避免量级相差特别大的原因采用归一化
    * 汇总起来---结果（多个样本）bestvalue的曲线（每次修改特征的bestvalue）
    	* 归一化 
    	* 多个样本进行比较

	* 不同攻击重复画即可（以后）
	

# 五 deepfool文件夹(针对200_200以后换架构)
实施deepfool+oneFeature攻击 
## tutorials文件夹
过程

* 在deepfool+oneFeature攻击时
* 先进行deepfoolattack线性攻击（有用，筛选索引）
* 模拟退火oneFeature攻击

### tutorial_deepfool_k.py
* 进行deepfool测试 测试一个恶意软件生成一个对抗样本
* 返回扰动特征数量
* 得到对抗样本
* 画图
	* 过程（单个样本）每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（黑色虚线）      	量级相差特别大，为避免量级相差特别大的原因采用归一化
    * 结果（单个样本）bestvalue的曲线（每次修改特征的bestvalue）（目前只画出一个样本）(可能要多个样本进行比较) 
### deepfooattackall.py
* 进行deepfool测试 测试整个测试集的恶意软件去生成对抗样本
* 返回平均扰动数量=扰动总数/恶意软件数量
* 针对某一架构DNN(到时候csv修改名字)
* 
	得到原始良性软件及其标签deepfool_200_200__X_begin.csv 
	deepfool_200_200__Y_begin.csv
	
	得到原始恶意软件及其标签 deepfool_xxx_xxx_X_normal.csv deepfool_xxx_xxx_Y_normal.csv
	
	得到对抗恶意软件及其标签  deepfool_200_200_X_adv.csv deepfool_200_200_Y_adv.csv

* 画图
	* 加时间戳---过程（单个样本）每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（黑色虚线）      	量级相差特别大，为避免量级相差特别大的原因采用归一化
    * 汇总起来---结果（多个样本）bestvalue的曲线（每次修改特征的bestvalue）
    	* 归一化 
    	* 多个样本进行比较

	* 不同攻击重复画即可（以后）

# 六 fgsmattack文件夹(针对200_200以后换架构)
实施fgsm+oneFeature攻击 
## tutorials文件夹
过程

* 在fgsm+oneFeature攻击时
* 先进行deepfoolattack线性攻击（无用）
* 模拟退火oneFeature攻击（FGSM每次迭代筛选索引）

### tutorial_fgsm_k.py
* 进行fgsm测试 测试一个恶意软件生成一个对抗样本
* 返回扰动特征数量
* 得到对抗样本
* 画图
	* 过程（单个样本）每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（黑色虚线）      	量级相差特别大，为避免量级相差特别大的原因采用归一化
    * 结果（单个样本）bestvalue的曲线（每次修改特征的bestvalue）（目前只画出一个样本）(可能要多个样本进行比较) 


### fgsmttackall.py
* 进行deepfool测试 测试整个测试集的恶意软件去生成对抗样本
* 返回平均扰动数量=扰动总数/恶意软件数量
* 针对某一架构DNN(到时候csv修改名字)
* 
	得到原始良性软件及其标签fgsm_200_200__X_begin.csv 
	fgsm_200_200__Y_begin.csv
	
	得到原始恶意软件及其标签 fgsm_xxx_xxx_X_normal.csv fgsm_xxx_xxx_Y_normal.csv
	
	得到对抗恶意软件及其标签  fgsm_200_200_X_adv.csv fgsm_200_200_Y_adv.csv

* 画图
	* 加时间戳---过程（单个样本）每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（黑色虚线）      	量级相差特别大，为避免量级相差特别大的原因采用归一化
    * 汇总起来---结果（多个样本）bestvalue的曲线（每次修改特征的bestvalue）
    	* 归一化 
    	* 多个样本进行比较

	* 不同攻击重复画即可（以后）



# 七 JSMF文件夹(针对200_200以后换架构)
## tutorial_jsmf_k.py
对几个恶意软件进行jsmf攻击

得到原始恶意软件及其标签

得到原始良性软件及其标签

得到对抗恶意软件及其标签

得到平均干扰

## attackall_jsmf.py
对测试集恶意软件进行jsmf攻击 

得到原始良性软件及其标签JSMF_200_200__X_begin.csv 
JSMF_200_200__Y_begin.csv

得到原始恶意软件及其标签 JSMF_xxx_xxx_X_normal.csv JSMF_xxx_xxx_Y_normal.csv

得到对抗恶意软件及其标签  JSMF_200_200_X_adv.csv JSMF_200_200_Y_adv.csv

保存到/data/jsmf  
得到平均干扰


# 八 第二种攻击模式 secondattack**** （难待突破）
* featurenames.csv筛选哪些特征(code）不能攻击
	
![](imgs/features.png) 
* 改攻击算法的输入还原输出
1 记录不能更改的特征对应的值A feature_names.csv
先找索引，记录值
	
2 将这些不能更改的值设置成1进入

3 攻击完在将不能更改的值由1还原成A


# 九 defence文件夹(只针对200_200一个架构)
## detector
### data
测试数据

X_adv.csv,Y_adv.csv对抗恶意样本

X_nomal.csv，Y_normal 正常恶意样本

X_train,Y_train 训练样本（其实是对应的test测试数据）
### extract_characteristics.py
对不同攻击（样本）得到两种特征lid,kd

运行的参数 -a deepfool -r lid -k 20 -b 100

-a 'jsmf', 'deepfool', 'onefeature', 'fgsm'

-r 'kd','lid'

# 重新设置bandwith*** （难待突破）
# 统计adv和normal对应值的信息（缺少Uncertites部分，待完全）
	kd
	Uncertities(另一个项目)
### Uncertainty(文件夹--提取不确定性）
#### train_pytorh_model.py
训练一个pytorch版本的模型 模型存放在Uncertainty/model下 保存为dnn.pkl
#### datajsma
小样本对抗数据 
#### datafinal 
大样本对抗数据 
### featuresjsma 
小样本结果
### featuresfinal
大样本结果
### extract_smalljsma_beforemethod.py
小样本提取 按之前的方法
两次提取 
normal提取一遍
adv提取一遍
### extract_testconbineFuture.py
大样本提取datajsma 

目前的方法

两次提取 

normal提取一遍

adv提取一遍
### extract_conbineFuture.py
对应的攻击样本提取 

目前的方法
两次提取 
normal提取一遍
adv提取一遍
### featuretoNpy.py
提取的文件到npy形式
### detect_adv_examples.py
1 加上不同的攻击参数 
2 两种情况 
训练集攻击和测试集攻击相同
训练集攻击和测试集攻击不同
Uncertities(另一个项目)pytorch
3 分类器的效率
### detect_and_detect.py
经过检测器在经过分类器
## 不同防御方法的比较

# 十 服务器上的操作
## 1 平均扰动针对三个攻击模型(服务器上待做)--已完成
直接运行所有攻击文件夹下的attackall
![](imgs/平均扰动.png) 

不同架构DNN重复上述操作 换加载模型，换生成样本的名字fgsm_xxx_xxx_X_begin.csv
## 2 选定一DNN模型不同攻击样本的准确率，FNR---以及误分类率
![](imgs/攻击模型准确率.png) 

运行Analyze/evaluate_adv.py
* 对于同一模型加载不同对抗样本 得到一个表的数据
* 不同架构DNN重复上述操作 换加载模型 得到所有表的数据


## 3 第二种攻击模式重复上述操作（带突破）

## 4 提取不同攻击对应的对抗样本的特征
extract_characteristics.py
不断更换参数
-a 'jsmf', 'deepfool', 'onefeature', 'fgsm'

-r 'kd','lid'

得到data/characteristic下八个文件

## 5 分析kd,bu在不同攻击下的区分效果
运行 characteristicsAnalyse.py

-a fgsm -r kd，bu
对应参数攻击参数
'jsmf', 'deepfool', 'onefeature', 'fgsm'

一次填

'kd', 'bu'

不停换攻击参数得到图表
![](imgs/区分特征.png) 






第二种攻击模式 secondattack**** （难待突破）
重新设置bandwith*** （难待突破）
Uncertainty(文件夹--提取不确定性）偶然不确定性就是小
* extract_combineFuture.py 已改完adv的改normal的路径+攻击+指标 能否参数化
* featuretoNpy.npy  对不同攻击 三种不确定性--->npy
统计adv和normal对应值的信息（缺少Uncertites部分，待完全）Analyze里
detect_adv_examples.py
1 加上不同的攻击参数 
2 两种情况 
训练集攻击和测试集攻击相同
训练集攻击和测试集攻击不同





