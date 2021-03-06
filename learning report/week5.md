# 第四章 机器学习基础
 
------
 
**目录** 如下：
 
> * 机器学习的四个分支
> * 评估机器学习模型
> * 数据预处理、特征工程和特征学习
> * 过拟合与欠拟合
> * 机器学习的通用工作流程
 
> 我们将把所有这些概念——`模型评估、数据预处理、特征工程、解决过拟合`——整合为详细的**七步**工作流程，用来解决任何 *机器学习* 任务。
 
------

### 1. 机器学习的四个分支
> * 监督学习
>给定一组样本（通常由人工标注），它可以学会将输入数据映射到已知目标［也叫**标注**]
**二分类问题、多分类问题和标量回归问题**都是**监督学习**（supervised learning）的例子，其目标是学习训练输入与训练目标之间的关系。
> * 无监督学习
> * 自监督学习
> * 强化学习
---
### 2.评估机器学习模型 
 > 评估模型的重点是将数据划分为三个集合：**训练集、验证集和测试集**
 > * 简单的留出验证
 > 留出一定比例的数据作为测试集。在剩余的数据上训练模型，然后在测试集上评估模型。为了防止信息泄露，不能基于测试集来调节模型，所以还应该保留一个验证集。
 ```python
 np.random.shuffle(data)# 打乱顺序
 validation_data = data[:10000]# 定义验证集
 data = data[10000:]
 training_data = data[:]# 定义训练集
 model = get_model()
 model.train(training_data)# 在训练集上训练模型
 validation_score = model.evaluate(validation_data)# 在验证集上评估模型
 
# 一旦调节好超参数，通常在所有非测试数据上从头开始训练最终模型
# 超参数：原因在于开发模型时总是需要调节模型配置，比如选择层数或每层大小［这叫作模型的超参数，以便与模型参数（即权重）区分开］

model = get_model()
model.train(np.concatenate([training_data,validation_data]))
test_score = model.evaluate(test_data)
 ```
 > * K折交叉验证
 ```python
 k = 4
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for i in range(k):
    validation_data = data[num_validation_samples * i:
    num_validation_samples * (i + 1)]
    training_data = data[:num_validation_samples * i] +
    data[num_validation_samples * (i + 1):]
	
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)
	
validation_score = np.average(validation_scores)

# 在所有非测试数据上训练最终模型
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)
```
 > * 带有打乱顺序的重复K折交叉验证
 > 如果可用的数据相对较少，而你又需要尽可能精确地评估模型，那么可以选择带有打乱数据的重复 K 折验证。
 > **具体做法**是多次使用 K 折验证，在每次将数据划分为 K 个分区之前都先将数据打乱。最终分数是每次 K 折验证分数的平均值。注意，这种方法一共要训练和评估 P×K 个模型（P是重复次数），计算代价很大。

>**评估模型的注意事项**
> * `数据代表性`例如，你想要对数字图像进行分类，而图像样本是按类别排序的，如果你将前 80% 作为训练集，剩余 20% 作为测试集，那么会导致训练集中只包含类别 0~7，而测试集中只包含类别 8~9。因此，在将数据划分为训练集和测试集
之前，通常应该**随机打乱数据**。
> * `时间箭头`如果想要根据过去预测未来（比如明天的天气、股票走势等），那么在划分数据前你不应该随机打乱数据，因为这么做会造成时间泄露。在这种情况下，你应该始终**确保测试集中所有数据的时间都晚于训练集数据**。
> * `数据冗余`如果数据中的某些数据点出现了两次，那么打乱数据并划分成训练集和验证集会导致训练集和验证集之间的数据冗余。从效果上来看，你是在部分训练数据上评估模型，这是极其糟糕的！一定要**确保训练集和验证集之间没有交集**
---

### 3. 数据预处理、特征工程和特征学习
 
> * 神经网络的数据预处理 `使原始数据更适于用神经网络处理`

> 1.向量化
神经网络的所有输入和目标都必须是**浮点数张量**（在特定情况下可以是整数张量）。无论处理什么数据（声音、图像还是文本），都必须首先将其转换为张量，这一步叫作数据向量化

> 2.值标准化
输入数据应该具有以下特征。
`取值较小`：大部分值都应该在 0~1 范围内。
`同质性`：所有特征的取值都应该在大致相同的范围内。
```python
# 假设 x 是一个形状为 (samples,features) 的二维矩阵
x -= x.mean(axis = 0)
x /= x.std(axis = 0)
```
> 3.值标准化
你的数据中有时可能会有缺失值。例如在房价的例子中，第一个特征（数据中索引编号为0 的列）是人均犯罪率。如果不是所有样本都具有这个特征的话，怎么办？那样你的训练数据或测试数据将会有缺失值。
一般来说，对于神经网络，将缺失值设置为 0 是安全的，只要 0 不是一个有意义的值。网络能够从数据中学到 0 意味着缺失数据，并且会忽略这个值。
注意，**如果测试数据中可能有缺失值，而网络是在没有缺失值的数据上训练的**，那么网络不可能学会忽略缺失值。在这种情况下，你应该`人为生成一些有缺失项的训练样本：多次复制一些训练样本，然后删除测试数据中可能缺失的某些特征`。
 
 > * 特征工程 
 > 特征工程的**本质**：用更简单的方式表述问题，从而使问题变得更容易。它通常需要深入理解问题。
 >  良好的特征仍然可以让你用更少的资源更优雅地解决问题。
 >  良好的特征可以让你用更少的数据解决问题。
---

### 4. 过拟合与欠拟合
 >机器学习的**根本问题**是优化和泛化之间的对立
 >欠拟合->过拟合
 >如何防止模型从训练数据中学到错误或无关紧要的模式？
 >`最优解决方法`是获取更多的训练数据
`次优解决方法`是调节模型允许存储的信息量，或对模型允许存储的信息加以约束，称为`正则化`
 > `正则化方法`包括：
 > * 减小网络容量
 > * 添加权重正则化
 > * 添加 dropout 正则化

>1.减小网络容量
>这由**层数**和**每层的单元个数**决定
>**工作流程**：开始时选择相对较少的层和参数，然后逐渐增加层的大小或增加新层，直到这种增加对验证损失的影响变得很小。

>2.添加权重正则化
>**奥卡姆剃刀原理**：如果一件事情有两种解释，那么最可能正确的解释就是最简单的那个，即假设更少的那个
>**简单模型**是指参数值分布的熵更小的模型（或参数更少的模型）。一种常见的降低过拟合的方法就是强制让模型权重只能取较小的值，从而限制模型的复杂度，这**使得权重值的分布更加规则**。其**实现方法**是向网络损失函数中添加与较大权重值相关的成本。
> `L1正则化`：添加的成本与权重系数的绝对值成正比。
> `L2 正则化`：添加的成本与权重系数的平方成正比。也叫`权重衰减`。
```python
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# l2(0.001) 的意思是该层权重矩阵的每个系数都会使网络总损失增加 0.001 * weight_coefficient_value 。注意，由于这个惩罚项只在训练时添加，所以这个
网络的训练损失会比测试损失大很多。
```
```python
from keras import regularizers
regularizers.l1(0.001)
regularizers.l1_l2(l1=0.001, l2=0.001)
# 同时做 L1 和 L2 正则化
```
>3.添加 dropout 正则化
>对某一层使用 dropout，就是在训练过程中随机将该层的一些输出特征舍弃（设置为 0）
>**dropout 比率**是被设为 0 的特征所占的比例，通常在 0.2~0.5 范围内
>**注意**：测试时没有单元被舍弃，而该层的输出值需要按 dropout 比率缩小，因为这时比训练时有更多的单元被激活，需要加以平衡。
```python
1.训练时，舍弃50%的输出单元；测试时，*0.5
    layer_output *= np.random.randint(0,high=2,size=layer_output.shape)
    layer_output *= 0.5
2.训练时，舍弃50%的输出单元，并且/0.5
    layer_output *= np.random.randint(0,high=2,size=layer_output.shape)
    layer_output /= 0.5
# 在 Keras 中，通过 Dropout 层向网络中引入 dropout，dropout 将被应用于前面一层的输出。
    model.add(layers.Dropout(0.5))
```
---
### 5. 机器学习的通用工作流程
 > * `定义问题，收集数据集`
 > 1.你的输入数据是什么？你要预测什么？
 > 2.你面对的是什么类型的问题？是二分类问题、多分类问题、标量回归问题、向量回归问题，还是多分类、多标签问题？或者是聚类、生成或强化学习？
 > * `选择衡量成功的指标`
 > 对于**平衡分类问题**（每个类别的可能性相同），精度和接收者操作特征曲线下面积是常用的指标。
 > 对于**类别不平衡的问题**，可以使用准确率和召回率。
 > 对于**排序问题或多标签分类**，你可以使用平均准确率均值
 > * `确定评估方法`
 > * `准备数据`
 > * `开发比基准更好的模型`
 > 这一阶段的目标是获得统计功效，即开发一个小型模型，它能够打败纯随机的基准。在 MNIST 数字分类的例子中，任何精度大于 0.1 的模型都可以说具有统计功效；在 IMDB 的例子中，任何精度大于 0.5 的模型都可以说具有统计功效。如果你尝试了多种合理架构之后仍然无法打败随机基准，那么原因可能是问题的答案并不在输入数据中。
 > **如果一切顺利，还需要选择三个关键参数来构建第一个工作模型**
 >1.最后一层的激活。它对网络输出进行有效的限制。例如，IMDB 分类的例子在最后一层使用了 sigmoid ，回归的例子在最后一层没有使用激活，等等。
>2.损失函数。它应该匹配你要解决的问题的类型。例如，IMDB 的例子使用binary_crossentropy 、回归的例子使用 mse 
>3.优化配置。你要使用哪种优化器？学习率是多少？大多数情况下，使用rmsprop 及其默认的学习率是稳妥的。
> * `扩大模型规模：开发过拟合的模型`
>1.添加更多的层。
>2.让每一层变得更大。
>3.训练更多的轮次。
> * `模型正则化与调节超参数`
> 1.添加 dropout。
2.尝试不同的架构：增加或减少层数。
3.添加 L1 和 / 或 L2 正则化。
4.尝试不同的超参数（比如每层的单元个数或优化器的学习率）以找到最佳配置。
5.反复做特征工程：添加新特征或删除没有信息量的特征。
---
