---
layout:     post
title:      Optimization Methos
subtitle:   null
date:       2019-02-22
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
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
</pre>

# 2 梯度下降

使用梯度下降来更新参数，就如图在成本函数J上 “下坡” 。可以把它想象成如下图这样的行为：

![](https://i.imgur.com/KtN5KI6.jpg)

在训练的每个步骤中，梯度下降都会按照特定方向更新参数，以尝试达到最低点。  

**梯度下降的方法：**  

对于 $ l = 1, ..., L $ :  

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} $$  
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} $$  

**梯度下降的代码如下：**  

<pre class="prettyprint lang-python">
# GRADED FUNCTION: update_parameters_with_gd
def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters
</pre>

梯度下降的一个变体是随机梯度下降（SGD），相当于Mini-Batch梯度下降，其中每个Mini-Batch只有一个数据。对于随机梯度下降，其更新参数的方法不会更改，有所变化的是，一次只计算一个训练数据而不是整个训练集上计算梯度。下面的代码示例说明了随机梯度下降和梯度下降之间的差异。

- **梯度下降（Batch Gradient Descent）：**
<pre class="prettyprint lang-python">
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost.
    cost = compute_cost(a, Y)
    # Backward propagation.
    grads = backward_propagation(a, caches, parameters)
    # Update parameters.
    parameters = update_parameters(parameters, grads)
</pre>

- **随机梯度下降（Stochastic Gradient Descent）：**
<pre class="prettyprint lang-python">
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost = compute_cost(a, Y[:,j])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)
</pre>

在随机梯度下降（SGD）中，在更新梯度之前，只使用一个训练数据。当训练集很大时，SGD可以更快。但是参数将 “振荡” 到最小值而不是平滑地收敛。如下图所示：

![](https://i.imgur.com/y5WjkmP.png)

> **注：实现随机梯度下降（SGD）总共需要三个循环：**
> - 迭代次数的循环
> - m个训练数据的循环
> - 神经网络层的循环（从 $ (W^{[1]},b^{[1]}) $ 到 $ (W^{[L]},b^{[L]}) $ ，更新所有的参数）

实际上，如果既不使用整个训练集，也不使用一个训练数据来执行每次的参数更新，则通常会更快获得结果。Mini-Batch梯度下降在每个更新参数的步骤使用的训练数据个数在一个和全部之间，即每步使用训练数据的个数既不是1也不是m，而是在1-m之间的某个数。

![](https://i.imgur.com/FaStw1D.png)

> **注：**
> - 梯度下降（Batch Gradient Descent），Mini-Batch梯度下降（Mini-Batch Gradient Descent）和随机梯度下降（Stochastic Gradient Descent）之间的差异是在执行一个更新步骤中所使用的训练数据的个数。
> - 必须调整超参数——学习率 $ \alpha $ 。
> - 一个调整良好的Mini-Batch大小，通常计算速度优于梯度下降和随机梯度下降（特别是当训练集很大时）。

# 3 Mini-Batch梯度下降

**在一个训练集(X, Y)上建立一个Mini-Batch的主要步骤如下：**
- **打乱：**如下图所示，将训练集(X, Y)随机打乱，X和Y的每一列代表一个训练数据。需注意，X和Y的打乱是同步的，即X的第i列是与Y中的第i个标签对应的样本。
![](https://i.imgur.com/aJD6ErE.png)

- **切分：**将打乱后的训练集进行切分，每部分的大小为 `mini_batch_size` （此处为64）,注意到最后一个切片的大小会比 `mini_batch_size` 小一些，如下图所示：
![](https://i.imgur.com/CgEdJPM.png)

**以下是随机建立Mini-Batch的代码：**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: random_mini_batches
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
</pre>

# 4 Momentum

因为Mini-Batch梯度下降在计算了训练集子集的成本之后就进行参数更新，所以更新的方向具有一些变化，因此Mini-Batch梯度下降所采用的路径将以 “振荡” 的形式收敛。使用Momentum可以减少这些振荡。  

Momentum将之前的梯度考虑了进去，使更新参数更加平滑。将先前梯度的 “方向” 存储在变量 `v` 中。形式上，这将是先前步骤的梯度的指数加权平均值。可以把 `V` 看作是球滚下山坡的速度，根据山坡的坡度（梯度gradient）建立新的速度（动量momentum）。如下图所示：  

![](https://i.imgur.com/9RlqCS5.png)

**首先，初始化速度 `v` ：**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_velocity
def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v
</pre>

**Momentum更新参数的方法如下：**

$$ \begin{cases}
v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\
W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}
\end{cases} $$  

$$ \begin{cases}
v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\
b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}} 
\end{cases} $$  

<pre class="prettyprint lang-python">
# GRADED FUNCTION: update_parameters_with_momentum
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v
</pre>

> **注：**
> - 速度 `v` 是用0来初始化的，因此，该算法需要经过几次迭代才能把速度 “提升” 上来。
> - 如果 $ \beta = 0 $ ，那么这相当于标准的、不使用Momentum的梯度下降。
> - 如何选择 $ \beta $ 的值？
>   -   $ \beta $ 的值越大，更新就越平滑，因为我们把之前更多的梯度都考虑了进去。但是如果 $ \beta $ 值过大，它也会使得更新过于平滑。
>   -   $ \beta $ 的值通常为0.8到0.999之间， $ \beta = 0.9 $ 通常是一个合理的默认值。
>   - 调整模型的最 $ \beta $ 可能需要尝试几个值，以便看看在降低成本函数J的时候，哪个最有效。
> - Momentum考虑之前的梯度，以平滑梯度下降。它可以应用于梯度下降，Mini-Batch梯度下降或随机梯度下降中。
> - 必须调整Momentum超参数 $ \beta $ 和学习率 $ \alpha $ 

# 5 Adam

Adam是用于训练神经网络的最有效的优化算法之一。它结合了RMSProp和Momentum的想法。  

**Adam的思路如下：**  
1. 计算先前梯度的指数加权平均值，并将其存储在变量 $ v $ （偏差校正前）和 $ v^{corrected} $ （偏差校正）中。
2. 计算先前梯度平方的指数加权平均值，并将其存储在变量 $ s $ （偏差校正之前）和 $ s^{corrected} $ （偏差校正）中。
3. 基于来自步骤一和步骤二的组合信息更新参数。

**具体表达式如下：**  

for $l = 1, ..., L$:  

$$ \begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\
v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\
s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_1)^t} \\
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}
\end{cases} $$

**具体代码如下：**

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_adam
def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape))
        s["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape))
        s["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape))

    return v, s
</pre>

<pre class="prettyprint lang-python">
# GRADED FUNCTION: update_parameters_with_adam
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - pow(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - pow(beta1, t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * pow(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * pow(grads['db' + str(l + 1)], 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - pow(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - pow(beta2, t))

        # Update parameters.Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon".Output: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected[
            "dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected[
            "db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s
</pre>

# 6 使用不同优化算法的模型

**加载数据集来测试不同的优化算法**

<pre class="prettyprint lang-python">
train_X, train_Y = load_dataset()
</pre>

**数据集如图：**

![](https://i.imgur.com/0qaejeB.png)

**神经网络模型如下：**

<pre class="prettyprint lang-python">
def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
</pre>

## 6.1 Mini-Batch梯度下降

<pre class="prettyprint lang-python">
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**   
Cost after epoch 0: 0.690736  
Cost after epoch 1000: 0.685273  
Cost after epoch 2000: 0.647072  
Cost after epoch 3000: 0.619525  
Cost after epoch 4000: 0.576584  
Cost after epoch 5000: 0.607243  
Cost after epoch 6000: 0.529403  
Cost after epoch 7000: 0.460768  
Cost after epoch 8000: 0.465586  
Cost after epoch 9000: 0.464518   

![](https://i.imgur.com/BEwOW6s.png)  

Accuracy: 0.7966666666666666   

![](https://i.imgur.com/tiHeewF.png)  

## 6.2 使用Momentum的Mini-Batch梯度下降

<pre class="prettyprint lang-python">
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**  
Cost after epoch 0: 0.690741  
Cost after epoch 1000: 0.685341  
Cost after epoch 2000: 0.647145  
Cost after epoch 3000: 0.619594  
Cost after epoch 4000: 0.576665  
Cost after epoch 5000: 0.607324  
Cost after epoch 6000: 0.529476  
Cost after epoch 7000: 0.460936  
Cost after epoch 8000: 0.465780  
Cost after epoch 9000: 0.464740  

![](https://i.imgur.com/fO6eKDW.png)

Accuracy: 0.7966666666666666   

![](https://i.imgur.com/qDnsSL5.png)

## 6.3 使用Adam的Mini-Batch梯度下降

<pre class="prettyprint lang-python">
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**  
Cost after epoch 0: 0.690552  
Cost after epoch 1000: 0.185501  
Cost after epoch 2000: 0.150830  
Cost after epoch 3000: 0.074454  
Cost after epoch 4000: 0.125959  
Cost after epoch 5000: 0.104344  
Cost after epoch 6000: 0.100676  
Cost after epoch 7000: 0.031652  
Cost after epoch 8000: 0.111973  
Cost after epoch 9000: 0.197940  

![](https://i.imgur.com/cj9Lz8n.png)

Accuracy: 0.94  

![](https://i.imgur.com/aFLttOw.png)

## 6.4 总结

| 优化方法 | 准确率 | 成本函数曲线 |
| :------------: | :-----------: | :-----------: |
| Gradient descent | 79.7 % | 振荡 |
| Momentum | 79.7 % | 振荡 |
| Adam | 94.0 % | 较光滑 |

> **注：**
> - Momentum通常有所帮助，但鉴于学习率较低且数据集过于简单，其影响几乎可以忽略不计。
> - Adam明显优于Mini-Batch梯度下降和Momentum。如果在此简单数据集上以更多的迭代次数来运行这个模型，则这三种方法都将产生非常好的结果。但是，Adam的收敛明显更快。
> - Adam的优势包括：
>   - 相对较低的内存要求（虽然高于梯度下降和使用Momentum的梯度下降）。
>   - 即使很少调整超参数（ & \alpha & 除外）通常也能很好地工作。


**参考资料：**
- Adam论文：[https://arxiv.org/pdf/1412.6980.pdf](https://arxiv.org/pdf/1412.6980.pdf)