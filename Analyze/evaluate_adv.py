import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix


#加载测试模型
#测试一种的分类模型
trained_model = tf.keras.models.load_model('..//malwareclassification//models//best_model_200_200.h5')


#0，1型数据
#正常恶意样本预测,加载正常的恶意样本数据
#加载不同的攻击方法产生不同的对抗样本******
#加载JSMF的恶意软件和良性软件
# normal_data = np.loadtxt(open("..//data//jsmf//JSMF_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# normal_labels = np.loadtxt(open("..//data//jsmf//JSMF_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# begin_data = np.loadtxt(open("..//data//jsmf//JSMF_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# begin_labels = np.loadtxt(open("..//data//jsmf//JSMF_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
#加载onefeature的恶意软件和良性软件
# normal_data = np.loadtxt(open("..//data//onefeature//onefeature_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# normal_labels = np.loadtxt(open("..//data//onefeature//onefeature_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# begin_data = np.loadtxt(open("..//data//onefeature//onefeature_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# begin_labels = np.loadtxt(open("..//data//onefeature//onefeature_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
#加载deepfool的恶意软件和良性软件
# normal_data = np.loadtxt(open("..//data//deepfool//deepfool_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# normal_labels = np.loadtxt(open("..//data//deepfool//deepfool_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# begin_data = np.loadtxt(open("..//data//deepfool//deepfool_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# begin_labels = np.loadtxt(open("..//data//deepfool//deepfool_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
#加载fgsm的恶意软件和良性软件
normal_data = np.loadtxt(open("..//data//fgsm//fgsm_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
normal_labels = np.loadtxt(open("..//data//fgsm//fgsm_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
begin_data = np.loadtxt(open("..//data//fgsm//fgsm_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
begin_labels = np.loadtxt(open("..//data//fgsm//fgsm_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
#拼接矩阵
#将良性软件与正常恶意软件拼接
val_data=np.concatenate((normal_data, begin_data), axis=0)
val_labels=np.concatenate((normal_labels, begin_labels), axis=0)

print("原始预测==============================")
#==============================================
#原始恶意样本预测
# print(val_labels)
predict_original = trained_model.predict(val_data)
confusion = confusion_matrix(val_labels, np.argmax(predict_original, axis=1))
# print(confusion)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
FNR_original = FN / float(FN + TP) * 100
FPR = FP / float(FP + TN) * 100
accuracy = ((TP + TN) / float(TP + TN + FP + FN)) * 100
print(confusion)
print("Original FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
print("Original Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR_original)
del predict_original


print("对抗预测=================")
#================对抗样本+原始良性软件预测
#加载jsmf对抗样本
# adv_data = np.loadtxt(open("..//data//jsmf//JSMF_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# adv_labels = np.loadtxt(open("..//data//jsmf//JSMF_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
#加载onefeature对抗样本
# adv_data = np.loadtxt(open("..//data//onefeature//onefeature_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# adv_labels = np.loadtxt(open("..//data//onefeature//onefeature_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
#加载deepfool对抗样本
# adv_data = np.loadtxt(open("..//data//deepfool//deepfool_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
# adv_labels = np.loadtxt(open("..//data//deepfool//deepfool_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)

#加载deepfool对抗样本
adv_data = np.loadtxt(open("..//data//fgsm//fgsm_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
adv_labels = np.loadtxt(open("..//data//fgsm//fgsm_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
#拼接矩阵
#将良性软件与正常恶意软件拼接
val_data1=np.concatenate((adv_data, begin_data), axis=0)
val_labels1=np.concatenate((adv_labels, begin_labels), axis=0)
# evaluate the model on adversarial examples
predictions = trained_model.predict(val_data1)
confusion = confusion_matrix(val_labels1, np.argmax(predictions, axis=1))
print(confusion)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
FNR = FN / float(FN + TP) * 100
FPR = FP / float(FP + TN) * 100
accuracy = ((TP + TN) / float(TP + TN + FP + FN)) * 100
print("Adversarial  FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
print("Adversarial Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)
print("Misclassification Rate:", FNR - FNR_original)