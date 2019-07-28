---
layout:     post
title:      TensorBoard可视化
subtitle:   null
date:       2019-07-27
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - TensorFlow

---

# TensorBoard可视化

TensorBoard可以有效地战士TensorFlow在运行过程中的计算图、各种指标随着时间的变化趋势以及训练中使用到的图像等信息。

## TensorBoard简介

TensorBoard和TensorFlow跑在不同的进程中，TensorBoard会自动读取最新的TensorFlow日志文件，并呈现当前TensorFlow程序运行的最新状态。

以下代码展示了一个简单的TensorFlow程序，在这个程序中完成了**TensorBoard日志输出**的功能。
```python
import tensorflow as tf

# 定义一个简单的计算图，实现向量加法的操作 
input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='add')

# 生成一个写的日志writer，并将当前的TensorFlow计算图写入日志
writer = tf.summary.FileWriter("/path/to/log", tf.get_default_graph())
writer.close()
```

运行以下命令便可以**启动TensorBoard**。

```python
# 运行TensorBoard，并将日志的地址指向上面程序日志输出的地址
tensorboard --logdir=/path/to/log
```

运行以上命令会启动一个服务，这个服务的端口默认为6006。通过浏览器打开localhost:6006，可以看到TensorBoard的界面，如下图所示。

![TensorBoard可视化向量相加程序的TensorFlow计算图结果](https://i.imgur.com/31JJusm.png)

## TensorFlow计算图可视化

为了更好地组织可视化效果图中的计算节点，TensorBoard支持通过TensorFlow命名空间来整理可视化效果图上的节点。

通过对命名空间管理，可以改进之前向量相加的样例代码，使得可视化得到的效果图更加清晰。

```python
import tensorflow as tf

# 将输入定义放入各自的命名空间中，从而使得TensorBoard可以根据命名空间来整理可视化效果图上的节点
with tf.name_scope('input1'):
    input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
with tf.name_scope('input2'):
    input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='add')

writer = tf.summary.FileWriter("F:/TensorFlowLog/testLog", tf.get_default_graph())
writer.close()
```

改进后的可视化效果图如下。

![改进后向量加法程序TensorFlow计算图的可视化效果图](https://i.imgur.com/7LZ1GbH.png)

## 节点信息

使用TensorBoard可以非常直观地展现所有TensorFlow计算节点在某一次运行时所消耗的时间和内存。

修改version3的mnist_train.py神经网络训练部分，就可以将不同迭代轮数的每个TensorFlow计算节点的运行时间和消耗的内存写入TensorBoard的日志文件中。

```python
with tf.Session() as sess:
    tf.global_variables_initializer().run()
	for i in range(TRAINING_STEPS):
    	xs, ys = mnist.train.next_batch(BATCH_SIZE)
		
		# 每1000轮保存一次模型
        if i % 1000 == 0:
        	# 配置运行时需要记录的信息
        	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
       		# 运行时记录运行信息的proto
        	run_metadata = tf.RunMetadata()
			# 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys}, options=run_options, run_metadata=run_metadata)
        	# 将节点在运行时的信息写入日志文件
        	train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			print("After %d training step(s), loss on training batch is %g" % (step, loss_value))
		else:
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
```

运行以上程序，并使用这个程序输出的日志启动TensorBoard，这样就可以可视化每个TensorFlow计算节点在某一次运行时所消耗的时间和空间。

## 监控指标可视化

tensorboard除了可以可视化TensorFlow的计算图，还可以可视化TensorFlow程序运行过程中各种有助于了解程序运行状态的监控指标。

*TensorFlow日志生成函数与TensorBoard界面栏对应关系*

| TensorFlow日志生成函数 | TensorBoard界面栏 | 展示内容 |
| --- | --- | --- |
| tf.summary.scalar | EVENTS | TensorFlow中标量（scalar）监控数据随着迭代进行的变化趋势 |
| tf.summary.image | IMAGES | TensorFlow中使用的图片数据，这一栏一般用于可视化当前使用的训练/测试数图片 |
| tf.summary.audio | AUDIO | TensorFlow中使用的音频数据 |
| tf.summary.text | TEXT | TensorFlow中使用的文本数据 |
| tf.summary.histogram | HISTOGRAMS, DISTRIBUTIONS | TensorFlow中张量分布监控数据随着迭代轮数的变化趋势 |

具体监控指标可视化操作代码见[https://github.com/LudoArt/MNIST-Project/tree/master/version4](https://github.com/LudoArt/MNIST-Project/tree/master/version4 "监控指标可视化版本")

## 高维向量可视化

TensorBoard提供了PROJECTOR界面来可视化高维向量之间的关系。

具体可视化操作实现代码见[https://github.com/LudoArt/MNIST-Project/tree/master/vector_visualization](https://github.com/LudoArt/MNIST-Project/tree/master/vector_visualization)

