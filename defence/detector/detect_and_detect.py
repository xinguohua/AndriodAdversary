# 经过检测器 对抗样本--->恶性
# 			 非对抗样本-->分类器---->良性
# 								 ---->恶性

# 在x_train上训练得到detecor
# 在x_test上检测
# 如果x_test是阳性 --> 1--------------------------------------------------------->
# 如果x_test是阴性 -->index X_test[index]--->原始分类器 预测predict 1 或者 0----> 两者于Y_test比较算准确率


from joblib import load

lr=load('logistic_allfeatures.model')
