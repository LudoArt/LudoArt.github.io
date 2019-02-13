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
1. 找到 `planar_utils.py` 文件 
2. 找到 `plot_decision_boundary()` 函数 
3. 修改 `plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)` 为 `plt.scatter(X[0, :], X[1, :], c=y.reshape(X[0,:].shape), cmap=plt.cm.Spectral)`

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

成本 *J* 为：
$$ J=-\frac{1}{m}$\sum_{i=0}^m(y^{(i)}log(a^{[2](i)})+(1-y^{(i)})log(1-a^{[2](i)})) $$

> **建立一个神经网络的通常方法：**
> 1. 定义神经网络的结构（如输入单元的数量，隐藏单元的数量等）
> 2. 初始化模型的参数
> 3. 循环：
>   - 实现正向传播过程
>   - 计算损失
>   - 实现反向传播过程，获取梯度
>   - 梯度下降更新参数
>   
> 通常构建一些函数来分别完成步骤1-3，然后将其整合到一个 `nn_model()` 函数当中。一旦完成了函数 `nn_model()` 并习得了正确的参数，便可以在新的数据上实现预测。

## 4.1 定义神经网络结构

<pre class="prettyprint lang-python">
# GRADED FUNCTION: layer_sizes
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)
</pre>

## 4.2 初始化模型参数

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_parameters
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
</pre>

> **Q1：为什么初始化 *W1* 和 *W2* 的时候要乘 *0.01* ？**
> **A：因为后续要利用梯度更新参数，当 *W1* 和 *W2* 较小的时候，梯度较大，更新参数的速度较快**
> 
> **Q2：为什么要初始化 *W1* 和 *W2* 为随机值？**
> **A：若将 *W1* 和 *W2* 全部初始化为0，则在计算隐藏单元时，每个隐藏单元将会进行相同的计算，那么设置隐藏层则变得没有意义**

## 4.3 循环

**第一步：实现前向传播过程**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: forward_propagation
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache
</pre>

**第二步：计算损失**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: compute_cost
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    # number of example
    m = Y.shape[1]

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = - np.sum(logprobs)/m

    # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17
    cost = np.squeeze(cost)
    assert (isinstance(cost, float))

    return cost
</pre>

**第三步：实现反向传播过程**

![](https://i.imgur.com/dmpn5mZ.png)

<pre class="prettyprint lang-python">
# GRADED FUNCTION: backward_propagation
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2-Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads
</pre>

## 4.4 更新参数

> 使用梯度下降来更新参数
> 一般梯度下降的规则：$$ \theta = \theta - \alpha \frac{∂J}{∂\theta} $$
> 此处的$$ \alpha $$代表学习率，$$ \theta $$代表参数
> 一个好的学习率和一个坏的学习率的表现可以如下图所示：
> ![](https://i.imgur.com/kBB81DT.gif)
> ![](https://i.imgur.com/lzfh0ff.gif)


<pre class="prettyprint lang-python">
# GRADED FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
</pre>

## 4.5 将以上三个步骤整合

<pre class="prettyprint lang-python">
</pre>

## 4.6 预测

<pre class="prettyprint lang-python">
</pre>

## 4.7 调整隐藏层的大小

<pre class="prettyprint lang-python">
</pre>
