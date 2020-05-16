## 数学基础
### 基本概念
>张量：多维数组
>阶、轴：张量的维度
>标量：0D张量 向量：1D张量 矩阵：2D张量

    x = np.array([12, 3, 6, 14, 7])    1个中括号
    x = np.array([[5, 78, 2, 34, 0],[7, 80, 4, 36, 2]])    2个中括号
    x = np.array([[[5, 78, 2, 34, 0],[7, 80, 4, 36, 2]],
                [[5, 78, 2, 34, 0],[7, 80, 4, 36, 2]],
                [[5, 78, 2, 34, 0],[7, 80, 4, 36, 2]]])   3个中括号

  > 深度学习处理的一般是 0D 到 4D 的张量，但处理视频数据时可能会遇到 5D 张量
   
   关键属性
  > 张量的阶(ndim)  形状(shape)  数据类型(dtype)
	
张量切片（tensor slicing）
  > 可以沿着每个张量轴在任意两个索引之间进行选择
  > 与 Python 列表中的负数索引类似，它表示与当前轴终点的相对位置，可以在图像中心裁剪出     > 14 像素×14 像素的区域：
  > my_slice = train_images[:, 7:-7, 7:-7]
    
   数据批量（batch)
  >通常来说，深度学习中所有数据张量的第一个轴（0 轴，因为索引从 0 开始）都是样本轴   >（samples axis，有时也叫样本维度）批量轴（batch axis) 批量维度
    >图像：4D张量，形状为 (samples, height, width, channels) 或 (samples, channels,height, >width)
    
   > 图像通常具有三个维度：高度、宽度和颜色深度
    如果图像大小为 256×256，那么 128 张灰度图像组成的批量可以保存在一个形状为 (128, 256, 256, 1) 的张量中，
    而 128 张彩色图像组成的批量则可以保存在一个形状为 (128, 256, 256, 3) 的张量中
    
  ######  神经网络的“齿轮”：张量运算
  逐元素运算
   >relu 运算和加法都是逐元素（element-wise）的运算,在 Numpy 中可以直接进行下列逐元素运算
   >import numpy as np
   >z = x + y
   >z = np.maximum(z, 0.)
    
   广播
   >将两个形状不同的张量相加,较小的张量会被广播（broadcast），以匹配较大张量的形状
    
   张量点积
   >高等代数里面矩阵乘法
    
   张量变形
   >张量变形是指改变张量的行和列，以得到想要的形状。变形后的张量的元素总个数与初始张量相同
    
   张量转置
   >经常遇到的一种特殊的张量变形是转置（transposition）。对矩阵做转置是指将行和列互换
    
   >仿射变换、旋转、缩放等基本的几何操作都可以表示为张量运算
    
   神经网络的“引擎”：基于梯度的优化
   >output = relu(dot(W, input) + b)
   >W 和 b 都是张量，均为该层的属性。它们被称为该层的权重（weight）或可训练参数（trainable parameter），分别对应 kernel 和 bias 属性
   >一种更好的方法是利用网络中所有运算都是可微（differentiable）的这一事实，计算损失相对于网络系数的梯度（gradient），然后向梯度的反方向改变系数，从而使损失降低。
    
   梯度（gradient）是张量运算的导数
   >使用动量方法可以避免局部极小
    
   链式求导：反向传播算法
   >将链式法则应用于神经网络梯度值的计算，得到的算法叫作反向传播（backpropagation，有时也叫反式微分，reverse-mode differentiation）。反
   >向传播从最终损失值开始，从最顶层反向作用至最底层，利用链式法则计算每个参数对损失值的贡献大小。
    
  损失
   >损失是在训练过程中需要最小化的量，因此，它应该能够衡量当前任务是否已成功解决。
   
  优化器
   >是使用损失梯度更新参数的具体方式，比如 RMSProp 优化器、带动量的随机梯度下降（SGD）等