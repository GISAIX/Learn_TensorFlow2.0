# 快速入门tensorflow1.x

TensorFlow是谷歌开源的**基于数据流图的科学计算库**，它适用于机器学习等人工智能领域。
相关链接：                             
[[TensorFlow官网]](https://tensorflow.google.cn/)                             
[[TensorFlow Github]](https://tensorflow.google.cn/)                              

---------------------------------------
## 1.TensorFlow的架构
![tf](https://img-blog.csdnimg.cn/2019042911425781.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BlY29IZQ==,size_16,color_FFFFFF,t_70)
* **前段(网络搭建)**：编程模型，构造计算图，Python，C++，Java。
* **后端(网络训练)**：运行计算图，C++。

TensorFlow具有高度的灵活性，真正的可移植性，多语言支持，性能最优化，社区内容丰富等优势特点。                         

## 2.TensorFlow中的基本概念
### 计算图Graph
计算图(Graph)描述了计算的过程，它是在前段完成的，并且可以通过tensorboard可视化出来。
* 声明一个图       
                                
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429120217953.PNG)

* 保存和加载pb文件，pb文件包括网络结构和网络参数

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429122840413.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BlY29IZQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429122900692.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BlY29IZQ==,size_16,color_FFFFFF,t_70)

* TensorBoard可视化计算图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429125302591.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BlY29IZQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429125347625.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BlY29IZQ==,size_16,color_FFFFFF,t_70)


### 会话Session
计算图的实际运算是在会话(Session)中进行的。会话将计算图的运算分发到设备(CPU OR GPU)上执行。
* 创建和关闭会话

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429125946146.PNG)

* 注入机制

```python
	sess.run()
```

* 指定设备

```python
 	with tf.device("/gpu:0"):
 		...
 ```
 
* GPU资源分配(按需分配)

```python
 	config = tf.ConfigProto()
 	config.gpu_options.allow_growth=True
 	sess = tf.Session(config = cnfig,...)
 ```
 
### 张量Tensor
在TensorFlow中所有节点之间传递的数据都是张量对象，对于图像而言，一般是一个**四维张量(batch x height x width x channels)**。

```python
tf.constant()
tf.Variable()
tf.placeholder()
```

### Operation
OP就是计算图中计算节点，输入输出都是张量。
### Feed

通过feed可以在计算时为定义为placeholder类型的变量赋值，参数是一个字典类型的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429131836380.PNG)




## 3.TensorFlow中的CV相关核心API总结
TensorFlow中CV相关核心API分为四类：
* 基本运算
* 搭建CNN网络
* 模型训练优化
* 数据读取相关

```
import tensorflow as tf
```

#### 基本运算

```
tf.expand_dims(input,dim,name=None)
tf.split(split_dim,num_split,value,name='split')
tf.concat(concat_dim,values,name='comcat')
tf.cast()
tf.reshape()
tf.equal()
tf.matmual(a,b)
tf.argmax()
tf.squeeze()
```

#### 搭建CNN网络(tf.nn)

```
tf.nn.conv2d()
tf.nn.max_pool()
tf.nn.avg_pool()
tf,nn.relu()
tf.nn.dropout()
tf.nn.l2_normalize()
tf.nn.batch_normalization()
tf.nn.l2_loss()
tf.nn.softmax_cross_entropy_with_logits
```

#### 模型训练优化(tf.train)
```
tf.train.Saver.save()
tf.train.Saver.restore()
tf.train.GradientDescentOptimize(learning_rate).minimize(loss)
tf.train.expoential_decay()
```

#### 数据读取相关
```
tf.train.string_input_prodicer(filenames,num_epochs,shuffle=True)
tf.train.shuffle_batch([example,label],batch_size,capacity,min_after_dequeue)
tf.train.Coordinator()
tf.train.start_queue_runners(sess,coord)
```


