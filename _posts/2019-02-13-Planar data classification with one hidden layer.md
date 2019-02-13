---
layout:     post
title:      Logistic Regression with a Neural Network mindset
subtitle:   null
date:       2019-02-08
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 人工智能
    - 深度学习
---
<head>
	<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
	<script src="https://cdn.rawgit.com/google/code-prettify/master/loader/run_prettify.js">
	</script>
</head>

# 1 导入所需的包

<pre class="prettyprint lang-python">
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# set a seed so that the results are consistent
np.random.seed(1)
</pre>

>#### 包介绍
>- **numpy** 用Python进行科学计算的基本软件包。
>- **sklearn** 为数据挖掘和数据分析提供的简单高效的工具。
>- **matplotlib** 是一个用于在Python中绘制图表的库。
>- **testCase** 提供了一些测试示例来评估函数的正确性，压缩包内提供
>- **planar_utils** 提供了在这个任务中使用的各种有用的功能，压缩包内提供

# 2 数据集

## 2.1 加载数据集

<pre class="prettyprint lang-python">
X, Y = load_planar_dataset() 
</pre>

## 2.2 可视化数据集

<pre class="prettyprint lang-python">
# Visualize the data（绘制散点图）
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
</pre>

**散点图如下：**
![数据集的散点图](https://i.imgur.com/rbfl1HP.png)

## 2.3 数据集的形状

<pre class="prettyprint lang-python">
# Get the shape of the variables X and Y
shape_X = X.shape
shape_Y = Y.shape
# training set size
m = shape_X[1]
</pre>

# 3 简单的线性回归
在实现完整的含有一个隐藏层的神经网络之前，先来看看简单的线性回归会有怎样的效果。

<pre class="prettyprint lang-python">
# Train the logistic regression classifier(use sklearn)
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) + '% ' + "(percentage of correctly labelled datapoints)")
plt.show()
</pre>

**输出结果如下：**
Accuracy of logistic regression: 47 % (percentage of correctly labelled datapoints)

![Simple Logistic Regression](https://i.imgur.com/enhWLoc.png)

> PS：此处的 `plot_decision_boundary()` 函数错误
> 修改bug方法如下：
1. 找到planar_utils.py 文件 
2. 找到plot_decision_boundary() 函数 
3. 修改plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral) 为plt.scatter(X[0, :], X[1, :], c=y.reshape(X[0,:].shape), cmap=plt.cm.Spectral) 

# 4 神经网络模型
含有一层隐藏层的神经网络模型如下图：
![Neural Network model](https://i.imgur.com/PPv6WcB.png)
用数学公式表示如下：

$$ z^{[1](i)}=W^{[1]}x^{(i)}+b^{[1](i)} $$

$$ a^{[1](i)}=tanh(z^{[1](i)}) $$

$$ z^{[2](i)}=W^{[2]}a^{(i)}+b^{[2](i)} $$

$$ \hat{y}^{(i)}=a^{[2](i)}=\sigma(z^{[2](i)}) $$

$$ =a^{[2](i)}=\sigma(z^{[2](i)}) $$

$$ y^{(i)}_prediction=
\begin{cases}
1& \text{if a^{[2](i)}>0.5}\\
0& \text{otherwise}
\end{cases} $$