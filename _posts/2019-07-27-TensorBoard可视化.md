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

```
# 运行TensorBoard，并将日志的地址指向上面程序日志输出的地址
tensorboard --logdir=/path/to/log
```

运行以上命令会启动一个服务，这个服务的端口默认为6006。通过浏览器打开localhost:6006，可以看到TensorBoard的界面，如下图所示。

![TensorBoard可视化向量相加程序的TensorFlow计算图结果](https://i.imgur.com/31JJusm.png)

## TensorFlow计算图可视化