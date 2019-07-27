---
layout:     post
title:      TensorFlow模型持久化
subtitle:   null
date:       2019-07-25
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - TensorFlow

---

# TensorFlow模型持久化

为了让训练结果可以复用，需要将训练得到的神经网络模型持久化。

## 持久化代码实现

TensorFlow提供了`tf.train.Saver`类来保存和还原一个神经网络模型。

```python
import tensorflow as tf

# 声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存到/path/to/model/model.ckpt文件
    saver.save(sess, "/path/to/model/model.ckpt")
```

以上代码实现了**持久化**一个简单的TensorFlow模型的功能。

以下代码给出了**加载**这个已经保存的TensorFlow模型的方法。

```python
import tensorflow as tf

# 使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess, "/path/to/model/model.ckpt")
    print(sess.run(result))
```

如果不希望重复定义图上的运算，也可以**直接加载**已经持久化的图。

```python
saver = tf.train.import_meta_graph("/path/to/model/model.ckpt/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "/path/to/model/model.ckpt")
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))  # 输出[3.]
```

为了**保存或者加载部分变量**，在声明`tf.train.Saver`类时，可以提供一个列表来指定需要保存或者加载的变量。

如在加载模型的代码中使用`save = tf.train.Saver([v1])`命令来构建`tf.train.Saver`类，那么就只有变量v1会被加载进来。

除了可以选取需要被加载的变量，tf.train.Saver类也支持在保存或者加载时给**变量重命名**。

```python
# 这里声明的变量名称和已经保存的模型中变量的名称不同
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")

# 如直接使用tf.train.Saver()来加载模型，会报变量找不到的错误。

# 使用一个字典（dictionary）来重命名变量就可以加载原来的模型了。
# 这个字典指定了原来名称为v1的变量现在加载到变量v1中（名称为other-v1），名称为v2的变量现在加载到变量v2中（名称为other-v2）。
saver = tf.train.Saver({"v1": v1, "v2": v2})
```

这样做主要目的之一是**方便使用变量的滑动平均值**。在TensorFlow中，每一个变量的滑动平均值是通过影子变量维护的，所以要获取变量的滑动平均值实际上就是获取这个影子变量的取值。如果在加载模型时直接将影子变量映射到变量自身，那么在使用训练好的模型时就不需要再调用函数来获取变量的滑动平均值了。这样大大方便了滑动平均模型的使用。

```python
v = tf.Variable(0, dtype=tf.float32, name="v")

# 在没有申请滑动平均模型时只有一个变量v
for varialbes in tf.global_variables():
    print(varialbes.name)  # v:0

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())

# 在申明滑动平均模型之后，TensorFlow会自动生成一个影子变量v/ExponentialMovingAverage
for varialbes in tf.global_variables():
    print(varialbes.name)  # v:0和v/ExponentialMovingAverage:0

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时，TensorFlow会将v:0v:0和v/ExponentialMovingAverage:0两个变量都存下来
    saver.save(sess, "/path/to/model/model.ckpt")
    print(sess.run([v, ema.average(v)]))  # [10.0, 0.099999905]
```

以下代码给出了如何**通过变量重命名直接读取变量的滑动平均值**。

```python
v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名将原来变量v的滑动平均值直接赋值给v
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "/path/to/model/model.ckpt")
    print(sess.run(v))  # 0.099999905 （原来模型中变量v的滑动平均值）
```

为了方便加载时重命名滑动平均变量，`tf.train.ExponentialMovingAverage`类提供了`variable_to_restore`函数来生成`tf.train.Saver`类所需要的变量重命名字典。

```python
v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用variables_to_restore函数可以直接生成上面代码中提供的字典{"v/ExponentialMovingAverage": v}
print(ema.variables_to_restore())  # 输出：{'v/ExponentialMovingAverage': <tensorflow.Variable 'v:0' shape() dtype=float32_ref>}

saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    saver.restore(sess, "/path/to/model/model.ckpt")
    print(sess.run(v))  # 0.099999905 （原来模型中变量v的滑动平均值）
```

将变量取值和计算图结构分成不同的文件存储有时候也不方便，于是TensorFlow提供了`convert_variables_to_constants`函数，通过这个函数可以**将计算图中的变量及其取值通过常量的方式保**存，这样整个TensorFlow计算图可以统一存放在一个文件中。

```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉
    # ['add']给出了需要保存的节点名称
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    
    # 将导出的模型存入文件
    with tf.gfile.GFile("/path/to/model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
```

通过以下程序可以直接计算定义的加法运算的结果。

```python
with tf.Session() as sess:
    model_filename = "/path/to/model/combined_model.pb"
    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前图中
    # return_elements=["add:0"]给出了返回的张量的名称
    # 在保存的时候给出的是计算节点的名称，所以是“add”。在加载的时候给出的是张量的名称，所以是“add:0”
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
```