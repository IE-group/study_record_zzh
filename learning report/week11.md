---
title: U-net基本原理
tags: 图像分割,全卷积神经网络FCN,U-net基本原理
renderNumberedHeading: true
grammar_cjkRuby: true
---
1.图像分割与图像检测的区别
图像检测是用框出框出物体，而图像分割要分出一个物体的准确轮廓。

2.CNN 与 FCN
通常CNN在卷积层之后会接上若干个全连接层, 将卷积层产生的特征图映射成一个固定长度的特征向量。经典CNN结构适合于图像级的分类和回归任务，因为它们最后都期望得到整个输入图像的一个数值描述(概率)。

FCN对图像进行像素级的分类，从而解决了语义级别的图像分割问题。与经典的CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类（全连接层＋softmax输出）不同，FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的特征图进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。

最后逐个像素计算softmax分类的损失, 相当于每一个像素对应一个训练样本。

简单的说，FCN与CNN的区别在于FCN把CNN最后的全连接层换成卷积层，输出的是一张已经label好的图片。

3.FCN的整体结构
![avatar](\FCN.png)
全卷积部分和反卷积部分。
全卷积部分：借用了一些经典的CNN网络并把最后的全连接层换成卷积，用于提取特征，形成热点图；
反卷积部分：将小尺寸的热点图上采样得到原尺寸的语义分割图像。

4.FCN的主要技术
`卷积化（Convolutional）`
全连接层都变成卷积层，适应任意尺寸输入，输出低分辨率的分割图片。为什么要把全连接层变为卷积层？【当我们输入的图片大小和卷积核大小一致时，其实等价于建立全连接，但是还是有区别。全连接的结构是固定的，当我们训练完时每个连接都是有权重的。而卷积过程我们其实为训练连接结构，学习了目标和那些像素之间有关系，权重较弱的像素我们可以忽略。连接不会学习过滤，会给每个连接分权重并不会修改连接关系。卷积则是会学习有用的关系，没用得到关系它会弱化或者直接 dropout。这样卷积块可以共用一套权重，减少重复计算，还可以降低模型复杂度。】

`上采样（Upsample)`
上采样指的是任何可以让图像变成更高分辨率的技术。最简单的方式是重采样和插值：将输入图片进行rescale到一个想要的尺寸，而且计算每个点的像素点，使用如双线性插值等插值方法对其余点进行插值来完成上采样过程。

`跳跃结构（Skip Layer）`
如果利用之前提到的上采样技巧对最后一层的特征图进行上采样的到原图大小的分割，由于最后一层的特征图太小，我们会损失很多细节。因而作者提出增加Skips结构将最后一层的预测（有更富的全局信息）和更浅层（有更多的局部细节）的预测结合起来，这样可以在遵守全局预测的同时进行局部预测。

5.FCN的优点与缺点
优点：实现端到端的分割
缺点：分割结果细节不够好，对各个像素进行分类，没有充分考虑像素与像素之间的关系。

6.图像的上采样（upsampling）与下采样（subsampled）

下采样原理：对于一幅图像 I 尺寸为 MxN，对其进行 s 倍下采样，即得到(M/s)x(N/s) 尺寸的分辨率图像，当然 s 应该是 M 和 N 的公约数才行，如果考虑的是矩阵形式的图像，就是把原始图像 sxs 窗口内的图像变成一个像素，这个像素点的值就是窗口内所有像素的均值。

上采样原理：上采样几乎都是采用内插值方法，即在原有图像像素的基础上在像素点之间采用合适的插值算法插入新的元素。常用的插值算法有最近邻插值，双线性插值，均值插值，中值插值。

7.FCN的上采样
上采样对应于最后生成heatmap的过程。
采用的网络经过5次卷积+池化后，图像尺寸依次缩小了 2、4、8、16、32倍，对最后一层做32倍上采样，就可以得到与原图一样的大小，现在我们需要将卷积层输出的图片大小还原成原始图片大小，在FCN中就设计了一种方式，叫做上采样，具体实现就是反卷积。

8.U-net网络结构
![avatar](\U-net.png)
`详解`
(1)输入是572x572的，但是输出变成了388x388，说明经过网络以后，输出的结果和原图不是完全对应的
(2)蓝色箭头代表3x3的卷积操作，并且stride是1，padding策略是vaild，因此，每个该操作以后，feature map的大小会减2。
(3)红色箭头代表2x2的max pooling操作，需要注意的是，此时的padding策略也是vaild（same 策略会在边缘填充0，保证feature map的每个值都会被取到，vaild会忽略掉不能进行下去的pooling操作，而不是进行填充），这就会导致如果pooling之前feature map的大小是奇数，那么就会损失一些信息 。
(4)绿色箭头代表2x2的反卷积操作，操作会将feature map的大小乘2。
(5)灰色箭头表示复制和剪切操作，可以发现，在同一层左边的最后一层要比右边的第一层要大一些，这就导致了，想要利用浅层的feature，就要进行一些剪切，也导致了最终的输出是输入的中心某个区域。
(6)输出的最后一层，使用了1x1的卷积层做了分类。

`特性`
(1)很多分割网络都是基于FCN做改进，包括Unet。Unet包括第一部分，特征提取，与VGG类似。第二部分上采样部分。由于网络结构像U型，所以叫Unet网络。
在特征提取部分，每经过一个池化层就更改一次尺度，包括原图尺度一共有5个尺度。在上采样部分，每上采样一次，就和特征提取部分对应的通道数相同的尺度融合，但是融合之前要将其crop。这里的融合也是拼接。 
(2)Ｕ-net输入输出 
医学图像是一般很大，进行分割的时候不能将原图太小输入网络，必须切成一张一张小的patch，在切成小patch的时候，U-net由于网络结构原因适合有overlap（即在切图时要包含周围区域）的切图，并且周围overlap部分可以为分割区域边缘部分提供文理等信息。
(3)U-net的反向传播
U-net反向传播过程，众所周知卷积层和池化层都能反向传播，而U-net上采样部分可以用上采样或反卷积，事实上反卷积和上采样也都可以进行反向传播。
反卷积就是转置卷积，也是一种卷积，是由小尺寸到大尺寸的过程。也就是说反卷积也可以表示为两个矩阵乘积，很显然转置卷积的反向传播就是也是可进行的。

9.U-net的keras简单实现
```python
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model
```



