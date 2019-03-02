---
layout:     post
title:      Building your Deep Neural Network Step by Step
subtitle:   null
date:       2019-02-17
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
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
</pre>

> **包介绍**
> - **numpy** 用Python进行科学计算的基本软件包。
> - **matplotlib** 是一个用于在Python中绘制图表的库。
> - **dnn_utils** 提供一些必要的函数。
> - **testCases** 提供一些测试用例来评估函数的正确性。

# 2 概要

**主要步骤如下：**
- 为一个2层神经网络和一个L层神经网络初始化参数
- 建立前向传播模型
	- 完成一层中前向传播的线性部分（得到$$ Z^{[l]} $$）
	- 激活函数 `relu/sigmoid` 
	- 将前两步结合到一个新的 `LINEAR->ACTIVATION` 前向函数中
	- 在新建的 `L_model_forward` 函数中，将 `LINEAR->RELU` 前向函数执行L-1 次（1~L-1层），然后再执行一次 `LINEAR->SIGMOID` 前向函数（最后的L层）
- 计算损失
- 建立反向传播模型
	- 完成一层中反向传播的线性部分
	- 激活函数的导数 `relu_backward/sigmoid_backward`
	- 将前两步结合到一个新的 `LINEAR->ACTIVATION` 反向函数中
	- 在新建的 `L_model_backward` 函数中，将 `LINEAR->SIGMOID` 反向函数执行1次，然后再执行L-1次 `LINEAR->RELU` 反向函数
- 更新参数

![final outline](https://i.imgur.com/ckv3bRI.png)

# 3 初始化参数

## 3.1 2层神经网络

> **该模型的结构是： LINEAR -> RELU -> LINEAR -> SIGMOID**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_parameters
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)

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

## 3.2 L层神经网络

> **该模型的结构是： [LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID**
 

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_parameters_deep
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters
</pre>

# 4 前向传播模型

## 4.1 线性部分（Linear）

> **线性部分要计算的公式如下：**
>     
> $$ Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]} $$  
>   
> 此处$$ A^{[0]}=X $$  

<pre class="prettyprint lang-python">
# GRADED FUNCTION: linear_forward
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache
</pre>

## 4.2 线性激活部分（Linear-Activation）

> **为了更方便，将把两个函数（ `LINEAR` 和 `ACTIVATION` ）放入一个函数（ `LINEAR-> ACTIVATION` ）当中。**  
>   
> **因此，该函数将首先执行LINEAR步骤，然后执行ACTIVATION步骤。**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: linear_activation_forward
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
</pre>

## 4.3 L层前向模型

> **该模型需执行使用RELU的 `linear_activation_forward` 函数L-1次，然后执行使用SIGMOID的 `linear_activation_forward` 函数一次**

![model_architecture_kiank](https://i.imgur.com/ikStrez.png)

<pre class="prettyprint lang-python">
# GRADED FUNCTION: L_model_forward
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches
</pre>

# 5 计算成本

> **计算成本的公式如下：**  
>   
> $$ J=-\frac{1}{m}\sum_{i=1}^m(y^{(i)}log(a^{[L](i)})+(1-y^{(i)})log(1-a^{[L](i)})) $$


<pre class="prettyprint lang-python">
# GRADED FUNCTION: compute_cost
def compute_cost(AL, Y):
    """
    Implement the cost function

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / m

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost
</pre>

# 6 反向传播模型

![backprop_kiank](https://i.imgur.com/hCJyP40.png)

## 6.1 线性部分（Linear）

> **假设已经计算了导数$$ dZ^{[l]}= \frac{∂L}{∂Z^{[l]}} $$** 
>  
> **现在需要求得参数$$ dW^{[l]},db^{[l]},dA^{[l-1]} $$**  

![linearback_kiank](https://i.imgur.com/MgfAYDa.png)

> **可使用以下公式：**  
>   
> $$ dW^{[l]}=\frac{∂L}{dW^{[l]}}=\frac{1}{m}dZ^{[l]}A^{[l-1]T} $$  
>   
> $$ db^{[l]}=\frac{∂L}{db^{[l]}}=\frac{1}{m}\sum_{i=1}^mdZ^{[l](i)} $$ 
>    
> $$ dA^{[l-1]}=\frac{∂L}{dA^{[l-1]}}=W^{[l]T}dZ^{[l]} $$  

<pre class="prettyprint lang-python">
# GRADED FUNCTION: linear_backward
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db
</pre>

## 6.2 线性激活部分（Linear-Activation）

> **为了更方便，将把两个函数（`LINEAR`和`ACTIVATION`）放入一个函数（`LINEAR-> ACTIVATION`）当中。**  
>   
> **因此，该函数将首先执行ACTIVATION步骤，然后执行LINEAR步骤。** 
>    
> **其中 $$ dZ^{[l]}=dA^{[l]}∗g′(Z^{[l]}) $$ ， $$ g′() $$ 为激活函数的导数**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: linear_activation_backward
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
</pre>

## 6.3 L层反向模型

![mn_backward](https://i.imgur.com/OCWLoqd.png)

> **首先计算$$ dA^{[L]}=\frac{∂L}{dA^{[L]}} $$**  
>    
> **即 `dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))`**    
>   
> **紧接着，便可以利用这个导数 `dAL` 继续进行反向计算**    
>   
> **首先进行一次 `LINEAR->SIGMOID` 反向函数的计算，然后再循环计算 `LINEAR->RELU` 反向函数L-1次**    
>   
> **应将每个 `dA` ， `dW` 和 `db` 存储在 `grads` 字典中，以便进行后续参数的更新**  

<pre class="prettyprint lang-python">
# GRADED FUNCTION: L_model_backward
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
</pre>

## 6.4 更新参数

> **使用梯度下降来更新参数：**  
>   
> $$ W^{[l]} = W^{[l]} - \alpha dW^{[l]} $$ 
>    
> $$ b^{[l]} = b^{[l]} - \alpha db^{[l]} $$  
>   
> 此处的 $$ \alpha $$ 是学习率

<pre class="prettyprint lang-python">
# GRADED FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate*grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate*grads["db" + str(l + 1)]

    return parameters
</pre>