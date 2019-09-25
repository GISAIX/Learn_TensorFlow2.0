# Learn TensorFlow2.0

<a href="https://github.com/StdCoutZRH/Learn_TensorFlow2.0/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/StdCoutZRH/Learn_TensorFlow2.0"></a>
<a href="https://github.com/StdCoutZRH/Learn_TensorFlow2.0/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/StdCoutZRH/Learn_TensorFlow2.0"></a>
<a href="https://github.com/StdCoutZRH/Learn_TensorFlow2.0/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/StdCoutZRH/Learn_TensorFlow2.0"></a>
<img alt="TF - TensorFlow Version" src="https://img.shields.io/badge/TensorFlow-2.0beta-orange">

<p align="center">
  <img src="TensorFlow2.0.gif" width="250" align="middle">
</p>

**TensorFlow2.0: An end-to-end open source machine learning platform.**

## 安装TensorFlow2.0

CPU版TF2.0安装命令：
```shell
 pip install --upgrade pip	# upgrade pip
 pip install tensorflow==2.0.0-rc1	# Install TensorFlow 2.0 RC
```

GPU版TF2.0安装命令(需要提前配置好CUDA等库)：
```shell
pip install --upgrade pip
pip install tensorflow-gpu==2.0.0-rc1
```

-------------

## 使用tf.keras构建你的model

本部分简单翻译自:[[Standardizing on Keras: Guidance on High-level APIs in TensorFlow 2.0]](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a)

虽然TensorFlow1.x支持Keras，但是Slim,Tensorlayer等功能重复地API常常让新手摸不着头脑。单一的高级API可以减少混淆，使TensorFlow能够专注于为研究人员提供高级功能。于是2.0版本的TensorFlow将Keras更紧密地集成进来，通过将Keras建立为TensorFlow的高级API，TensorFlow2.0使新的机器学习开发人员更容易地上手TensorFlow和进行相关开发。


 
**Keras是一个非常受欢迎的用于构建和训练深度学习模型的高级API**。
它有几个关键优势：
* **对用户友好**： Keras具有针对常见应用案例进行优化的简洁、一致的接口。它对用户的错误提供清晰且可操作的反馈、有用的建议和易于理解的错误信息。
* **模块化和可组合**： Keras的模型是由可配置的构建块连接而成，并且构建几乎没有限制。即使你没有使用Keras平台中的模型，Keras中的部分组件也是可复用的。例如，你可以使用Keras中的layers或Optimizers即使你并没有使用Keras Model进行训练。
* **易于扩展**：你可以编写自定义构建块来表达研究的新想法，包括新的层，损失函数和以开发最先进的想法。
* **对于初学者和专家**：深度学习开发人员来自许多背景和经验水平，无论你是刚刚开始，还是有多年的经验，Keras都提供非常有用的API。从学习ML到研究，应用程序开发到部署，这些工作流程可以在更广泛的用例中实现更轻松，更高效的工作流程。


## 常见问题总结

#### Keras是一个单独的库吗？ 
Keras是一个API规范。Keras的实现是作为[独立的开源项目](www.keras.io) 维护的，该项目独立于TensorFlow，并拥有一个活跃的开发者和用户社区。TensorFlow的tf.keras模块中包含Keras API的完整实现，具有TensorFlow特定的增强功能。

#### Keras只是TensorFlow或其他库的封装吗？
不，这是一个常见，但可以理解的错误观念。Keras是用于定义和训练机器学习模型的API标准。Keras与特定实现无关：Keras API具有TensorFlow，MXNet，TypeScript，JavaScript，CNTK，Theano，PlaidML，Scala，CoreML和其他库的实现。

#### 内置于TensorFlow的Keras版本与在keras.io上可以找到的版本有什么区别？
TensorFlow包含Keras API（在tf.keras模块中）的实现，具有TensorFlow特定的增强功能。包括支持Eager execution立即执行、直观调试和快速迭代等，支持TensorFlow SavedModel模型转换格式，以及对分布式训练的集成支持，包括TPU训练模型。使用tf.keras 模型子类 API 时，Eager execution特别有用。这个API的灵感来自[Chainer](https://chainer.org) ，使你能够强制性地编写模型的正向传递。tf.keras紧密集成到TensorFlow生态系统中，还包括对以下内容的支持：
* tf.data:使您能够构建高性能输入管道。如果您愿意，可以使用NumPy格式的数据训练模型，或使用tf.data进行缩放和性能训练。
* 分发策略：用于在各种计算配置分布式训练，包括分布在许多计算机上的GPU和TPU。
* 导出模型：使用tf.keras API创建的模型可以使用TensorFlow SavedModel格式进行序列化，并使用TensorFlow Serving或其他 语言绑定（Java，Go，Rust，C＃等）提供。导出的模型可以使用TensorFlow Lite部署在移动和嵌入式设备上，也可以与TensorFlow.js一起使用（也可以使用相同的Keras API直接在JavaScript中开发模型）。
* 功能序列：用于有效地表示和分类结构化数据。 

还有更多的工作。。。

#### TensorFlow为初学者和专家提供不同的API样式，这些用起来怎么样？
TensorFlow开发人员拥有许多经验水平（从第一次学习ML的学生到ML专家和研究人员）。同样，TensorFlow的优势之一是它提供了多个API来支持不同的工作流程和目标。同样，这是TensorFlow Keras集成的主要设计目标，用户可以选择Keras的一部分，而不必采用整个框架。

## tf.keras常用模型构建方法

#### Sequential API
如果你是学生学习ML，建议开始使用tf.keras Sequential API。它直观，简洁，适用于实践中95％的ML问题。使用此API，大约10行代码就可以编写你的第一个神经网络。

定义模型的最常用方法是构建层级图，这与我们研究深度学习时通常使用的模型相对应。最简单的模型类型是一堆层，你可以使用Sequential API定义这样的模型，如下所示：
```python
model = tf.keras.Sequential()
model.add(layers.Dense(64,activation ='relu'))
model.add(layers.Dense(64,activation ='relu'))
model.add(layers.Dense)(10,activation ='softmax')
```
然后可以在几行中编译和训练这样的模型：
```python
model.compile(optimizer ='adam',
loss ='sparse_categorical_crossentropy',
metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 5)
model.evaluate(x_test,y_test)
```
你可以在tensorflow.org/tutorials上找到更多使用Sequential API的示例。

#### Functional API

当然，顺序模型是一个简单的层次堆栈，不能用来构建任意输入输出的模型。使用Functional API可以构建更高级的模型，比如你可以定义复杂的拓扑模型，该模型包括多输入和多输出，或者具有共享层的模型以及具有残差连接的模型。

在使用Functional API构建模型时，layers可以回调（在张量上），并返回张量作为输出。然后可以使用这些输入张量和输出张量来定义模型。例如：
```python
inputs = tf.keras.Input(shape =(32,))
#图层实例可在张量上调用，并返回张量。
x = layers.Dense(64,activation ='relu')(input)
x = layers.Dense(64,activation ='relu')(x)
predictions = layers.Dense(10,activation ='softmax')(x)
#实例化给定输入和输出的模型。
model = tf.keras.Model(inputs = inputs,outputs = predictions)
```
可以使用上面相同的简单语句来编译和训练这个模型。

#### Model Subclassing API

使用Model Subclassing API构建完全可自定义的模型。你可以在类方法的主体中以此样式强制定义自己的前向传递。例如：
```python
class MyModel(tf.keras.Model):
    def __init (self):
        super(MyModel,self).__init__()
        #在这里定义你的网络层
        self.dense_1 = layers.Dense(32，activation ='relu')
        self.dense_2 = layers.Dense(num_classes，activation ='sigmoid')
    def call(self,inputs):
        #在这里定义你的前向传递，
        x = self.dense_1(input)
        return self.dense_2(x)
```
这些模型更加灵活，但可能更难调试。可以使用前面显示的简单编译和拟合命令编译和训练所有三种类型的模型，或者你可以编写自己的自定义训练循环以进行完全控制。
例如：
```python
model = MyModel()
with tf.GradientTape()as tape:
    logits = model(images,training = True)
    loss_value = loss(logits,labels)
    grads = tape.gradient(loss_value,model.variables)
    optimizer.apply_gradients(zip(grads,model.variables))
```

#### 如果我的研究不适合这些风格怎么办？
如果你发现tf.keras在你的应用领域有所局限，你有很多选择。
你可以：
将kef.keras.layers与Keras模型定义分开使用，并编写自己的梯度和训练代码。你可以单独地使用tf.keras.optimizers，tf.keras.initializers，tf.keras.losses或tf.keras.metrics。或者完全忽略tf.keras并使用低级TensorFlow，Python和AutoGraph来获得所需的结果。这完全取决于你！请注意，不推荐使用tf.layers中的非面向对象层，并且tf.contrib.*（包括tf.contrib.slim和tf.contrib.learn等高级API）将无法在TF 2.0中使用。

#### Estimators会发生什么样的变化？
估算器广泛用于Google以及更广泛的TensorFlow社区。已经将几种模型打包为Premade Estimators，包括线性分类器，DNN分类器，组合DNN线性分类器（又名宽和深模型）和梯度增强树。这些模型已经投入生产并得到广泛部署，由于所有这些原因，Estimator API（包括Premade Estimators）将包含在TensorFlow 2.0中。
对于Premade Estimators的用户来说，新焦点对Keras和Eager execution的影响将是微乎其微的。我们可能会更改Premade Estimators的实现，同时保持API表面相同。我们还将努力添加作为Premade Estimators实现的模型的Keras版本，我们将扩展Keras以更好地满足大规模生产要求。
也就是说，如果您正在开发自定义架构，我们建议使用tf.keras来构建模型而不是Estimator。如果您正在使用需要Estimators的基础架构，您可以使用model_to_estimator（）来转换模型，同时确保KerasTensorFlow生态系统中工作。

## 向TensorFlow 2.0进发！
我们希望您能像我们一样喜欢使用tf.keras！在接下来的几个月里，TensorFlow团队将专注于完善开发人员体验。我们的文档和教程将反映这一方向。我们期待您的想法和反馈，以及通过GitHub问题和PR的贡献。感谢大家！