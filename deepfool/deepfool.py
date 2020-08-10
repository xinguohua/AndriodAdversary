import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
       ：param图片：大小为HxWx3的图片
       ：param net：网络（输入：图像，输出：激活值** BEFORE ** softmax）。
       ：param num_classes：num_classes（限制要测试的类的数量，默认为10）
       ：param overshoot：用作终止条件以防止更新消失（默认= 0.02）。
        ：param max_iter：深度傻瓜的最大迭代次数（默认= 50）
        ：return：最小化分类器的扰动，所需的迭代次数，新的estimate_label和扰动的图像
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes] #默认十类
    label = I[0]
    # 获取原始样本的维度，返回（第一维长度，第二维长度，...）
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)  # 深度复制原始样本，复制出来后就独立了
    w = np.zeros(input_shape)  # 返回来一个给定形状和类型的用0填充的数组
    r_tot = np.zeros(input_shape)

    loop_i = 0 # 循环

    x = Variable(pert_image[None, :], requires_grad=True) # 因为神经网络只能输入Variable
    fs = net.forward(x)   # 调用forward函数
    fs_list = [fs[0,I[k]] for k in range(num_classes)]    # 每个类别的取值情况，及其对应的梯度值
    k_i = label

    while k_i == label and loop_i < max_iter:  # 分类标签变化时结束循环

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes): # 获得x到各分类边界的距离
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy() # 现在梯度

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert: # 获得最小的分类边界距离向量
                pert = pert_k  # 更新perk，pert为最小距离
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)  # 累积扰动

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda() # 添加扰动
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot  # 最终累积的扰动

    return r_tot, loop_i, label, k_i, pert_image
