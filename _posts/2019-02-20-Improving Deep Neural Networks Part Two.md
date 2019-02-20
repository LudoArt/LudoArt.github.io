---
layout:     post
title:      Improving Deep Neural Networks Part Two
subtitle:   null
date:       2019-02-20
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

# 改进深度神经网络之二：正则化

# 1 导入所需的包和数据集

<pre class="prettyprint lang-python">
# import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
</pre>

**数据集为：**

![](https://i.imgur.com/Fa2Tawf.png)

# 2 非正则化模型

> - **L2正则化函数：** `compute_cost_with_regularization()` 和 `backward_propagation_with_regularization()` 
> 
> - **Dropout正则化函数：** `compute_cost_with_dropout()` 和 `backward_propagation_with_dropout()` 
> *（这四个函数将在下文详细介绍）*  
> 
> - **L2正则化模式：** 将变量 `lambd` 设置为一个非零值  
> - **Dropout正则化模式：** 将变量 `keep_prob` 设置为一个小于1的值  

<pre class="prettyprint lang-python">
def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # Backward propagation.
        assert (lambd == 0 or keep_prob == 1)  # it is possible to use both L2 regularization and dropout,
        # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
</pre>

#### 2.1 训练模型

<pre class="prettyprint lang-python">
parameters = model(train_X, train_Y)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
</pre>

**运行结果如下：**   
Cost after iteration 0: 0.6557412523481002  
Cost after iteration 10000: 0.1632998752572419  
Cost after iteration 20000: 0.13851642423239133  

On the training set:  
Accuracy: 0.9478672985781991  
On the test set:  
Accuracy: 0.915  

![](https://i.imgur.com/ZFTjQCl.png)

#### 2.2 决策边界图

<pre class="prettyprint lang-python">
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：** 

![](https://i.imgur.com/fPn9Zbl.png)

#### 2.3 小结
> - 从上图中可以看出，非正则化模型显过度拟合了训练集。（将噪点也拟合了进去）

# 3 L2正则化

#### 3.1 正向传播

**非正则化计算损失：**  
$$ J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} $$  
  
**L2正则化计算损失：**  
$$ J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} $$

<pre class="prettyprint lang-python">
# GRADED FUNCTION: compute_cost_with_regularization
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)  # This gives you the cross-entropy part of the cost

    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / m / 2 * lambd

    cost = cross_entropy_cost + L2_regularization_cost

    return cost
</pre>

#### 3.2 反向传播

**在计算梯度的时候，须加上正则化项的梯度：**  
  
$$ \frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W $$

<pre class="prettyprint lang-python">
# GRADED FUNCTION: backward_propagation_with_regularization
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T) + lambd / m * W3
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) + lambd / m * W2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T) + lambd / m * W1
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
</pre>

#### 3.3 训练模型

<pre class="prettyprint lang-python">
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
</pre>

**运行结果如下：**  
Cost after iteration 0: 0.6974484493131264  
Cost after iteration 10000: 0.2684918873282239  
Cost after iteration 20000: 0.2680916337127301  

On the train set:  
Accuracy: 0.9383886255924171  
On the test set:  
Accuracy: 0.93  

![](https://i.imgur.com/TD5KpZ7.png)

#### 3.4 决策边界图

<pre class="prettyprint lang-python">
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**  

![](https://i.imgur.com/cRaFbDP.png)

#### 3.5 小结

> -  $$ \lambda $$ 是一个超参数，可以使用开发集（dev set）来调整它。
> -  L2正则化使决策边界更加光滑，如果 $$ \lambda $$ 过大，有可能产生“过渡光滑”，结果就是模型的偏差变高。
> - L2正则化依赖于这样的假设：具有小权重的模型比具有大权重的模型更简单。因此，通过在成本函数中加上一个L2正则化的成本（即权重的平方值），可以使所有权重变为为更小的值。
> - **L2正则化的影响有：**
>   - **成本函数的计算：**成本函数中要加上一个正则化项
>   - **反向传播函数：**在求权重的梯度时，要有一个额外的项
>   - **权重在结束的时候变得更小了（“权重衰减”）**

# 4 随机失活（Dropout）

> - Dropout是一个在深度学习中广泛使用的正则化技术，它会在每次迭代的时候随机关闭一些神经元。
> - 在每次迭代中，以 `1-keep_prob` 概率关闭一层中的每个神经元。关闭的神经元在迭代的前向传播和后向传播中的训练没有作用。
> - 当关闭一些神经元时，实际上是修改了模型。 *Dropout* 的思想是，在每次迭代中，都会训练一个仅使用神经元子集的不同的模型。通过 *Dropout* ，神经元对某一个特定神经元的激活变得不那么敏感，因为其他神经元可能随时被关闭。 

#### 4.1 正向传播

**若想要在第一层和第二层随机关闭一些神经元，可按以下步骤：**
- 首先，创建一个和变量 $$ a^{[1]} $$ 一样维度的变量 $$ d^{[1]} $$ ， $$ d^{[1]} $$ 中的元素赋予0到1之间的随机值。对其向量化，即创建一个矩阵 $$ D^{[1]}=[d^{[1](1)}d^{[1](2)}...d^{[1](m)}] $$ ， $$ D^{[1]} $$ 与 $$ A^{[1]} $$ 维度相同；
- 其次，对 $$ D^{[1]} $$ 中的每个元素进行阈值处理，使其以 `1-keep_prob` 的概率为0，以 `keep_prob` 的概率为1；
- 再次，使 $$ A^{[1]}=A^{[1]}*D^{[1]} $$ 。可以将 $$ D^{[1]} $$ 当做一个遮罩，当它乘以别的矩阵的时候，它会关闭一些值。
- 最后， $$ A^{[1]}=A^{[1]} $$/keep_prob 。通过这样做，您可以确保成本的结果仍然具有与不使用Dropout时相同的预期值。（这种技术也称为反向丢失。）

<pre class="prettyprint lang-python">
# GRADED FUNCTION: forward_propagation_with_dropout
def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    D1 = np.random.rand(A1.shape[0], A1.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = (D1 < keep_prob)  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = np.multiply(A1, D1)  # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    D2 = np.random.rand(A2.shape[0], A2.shape[1])  # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = (D2 < keep_prob)  # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = np.multiply(A2, D2)  # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache
</pre>

#### 4.2 反向传播

**反向传播过程中，实现Dropout需要有以下两个步骤：**
- 在前向传播过程中，通过对 $$ A^{[1]} $$ 使用遮罩 $$ D^{[1]} $$ 来关闭一些神经元。在反向传播中，再次对 $$ A^{[1]} $$ 使用遮罩 $$ D^{[1]} $$ 来关闭神经元；
- 在前向传播过程中，将 `A1` 除以 `keep_prob` 。在反向传播中，你需要对 `dA1` 除以 `keep_prob` 。

<pre class="prettyprint lang-python">
# GRADED FUNCTION: backward_propagation_with_dropout
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = np.multiply(dA2, D2)  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1, D1)  # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
</pre>

#### 4.3 训练模型

<pre class="prettyprint lang-python">
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
</pre>

**运行结果如下：**  
Cost after iteration 0: 0.6543912405149825  
Cost after iteration 10000: 0.061016986574905605  
Cost after iteration 20000: 0.060582435798513114  

On the train set:  
Accuracy: 0.9289099526066351  
On the test set:  
Accuracy: 0.95  

![](https://i.imgur.com/uzIlF9i.png)

#### 4.4 决策边界图

<pre class="prettyprint lang-python">
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**  

![](https://i.imgur.com/i64RKMb.png)

#### 4.5 小结

> - Dropout是一种正则化技术。
> - 使用Dropout时常见的错误是在训练和测试中都使用它。应仅在训练中使用dropout（随机消除节点），在测试中不要使用。
> - 在前向传播和反向传播过程中都应该使用Dropout。
> - 在训练期间，每个应用了Dropout技术的层都应该除以keep_prob以保持激活的相同预期值。例如，如果keep_prob为0.5，那么我们平均会关闭一半节点，此时只剩下一半对解决该问题有所贡献，因此输出将缩小0.5呗。除以0.5相当于乘以2。因此，输出将具有相同的期望值。

# 5 总结

**三种不同模型的结果如下：**

| 模型 | 训练准确率 | 测试准确率 |
| :------------: | :-----------: | :-----------: |
| 无正则化的三层神经网络 | 95.0 % | 91.5 % |
| 使用L2正则化的三层神经网络 | 94.0 % | 93.0 % |
| 使用Dropout正则化的三层神经网络 | 93.0 % | 95.0 % |

> **注：**
> - 正则化会损害在训练集上的表现。因为它限制了神经网络过度拟合训练集的能力。但由于它最终会使得测试的准确性更高。
> - 正则化可以帮助减轻过度拟合。
> - 正则化会使权重更低。
> - L2正则化和Dropout是两种非常有效的正则化技术。







