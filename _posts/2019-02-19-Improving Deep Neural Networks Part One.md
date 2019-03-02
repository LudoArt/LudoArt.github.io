---
layout:     post
title:      Improving Deep Neural Networks Part One
subtitle:   Initialization
date:       2019-02-19
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

# 改进深度神经网络之一：初始化

# 1 导入所需的包和数据集

<pre class="prettyprint lang-python">
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
</pre>

**数据集为：**

![](https://i.imgur.com/1iYMtUL.png)

**现要做一个分类器将红点和蓝点区分开来**

# 2 神经网络模型

<pre class="prettyprint lang-python">
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
</pre>

# 3 全为零的初始化

#### 初始化方法：

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_parameters_zeros 
def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters
</pre>

#### 训练模型：

<pre class="prettyprint lang-python">
parameters = model(train_X, train_Y, initialization = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
</pre>

**运行结果如下：**  
Cost after iteration 0: 0.6931471805599453  
Cost after iteration 1000: 0.6931471805599453  
Cost after iteration 2000: 0.6931471805599453  
Cost after iteration 3000: 0.6931471805599453  
Cost after iteration 4000: 0.6931471805599453  
Cost after iteration 5000: 0.6931471805599453  
Cost after iteration 6000: 0.6931471805599453  
Cost after iteration 7000: 0.6931471805599453  
Cost after iteration 8000: 0.6931471805599453  
Cost after iteration 9000: 0.6931471805599453  
Cost after iteration 10000: 0.6931471805599455  
Cost after iteration 11000: 0.6931471805599453  
Cost after iteration 12000: 0.6931471805599453  
Cost after iteration 13000: 0.6931471805599453  
Cost after iteration 14000: 0.6931471805599453  

On the train set:  
Accuracy: 0.5  
On the test set:  
Accuracy: 0.5  

![](https://i.imgur.com/UpUdi7z.png)

#### 查看决策边界图：

<pre class="prettyprint lang-python">
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**  

![](https://i.imgur.com/UCaO7Dx.png)

#### 小结：
> - 通常，将所有权重初始化为零会导致神经网络无法破坏对称性。这意味着每层中的每个神经元都会学到同样的东西，相当于在训练一个每一层只有一个隐藏单元的神经网络，这样的神经网络甚至不如线性分类器，如逻辑回归。
> - 应该随机初始化权重 $$ W^{[l]} $$ 以破坏对称性。
> - 可以将偏差 $$ b^{[l]} $$ 初始化为零。只要 $$ W^{[l]} $$ 随机初始化，对称性仍然会被破坏。


# 4 随机数的初始化

#### 初始化方法：

将权重 $$ W^{[l]} $$ 随机初始化，但均初始化为较大的值，观察会有什么样的结果
<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_parameters_random
def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters
</pre>

#### 训练模型：

<pre class="prettyprint lang-python">
parameters = model(train_X, train_Y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
</pre>

**运行结果如下：**
Cost after iteration 0: inf  
Cost after iteration 1000: 0.6250982793959966  
Cost after iteration 2000: 0.5981216596703697  
Cost after iteration 3000: 0.5638417572298645  
Cost after iteration 4000: 0.5501703049199763  
Cost after iteration 5000: 0.5444632909664456  
Cost after iteration 6000: 0.5374513807000807  
Cost after iteration 7000: 0.4764042074074983  
Cost after iteration 8000: 0.39781492295092263  
Cost after iteration 9000: 0.3934764028765484  
Cost after iteration 10000: 0.3920295461882659  
Cost after iteration 11000: 0.38924598135108  
Cost after iteration 12000: 0.3861547485712325  
Cost after iteration 13000: 0.384984728909703  
Cost after iteration 14000: 0.3827828308349524  

On the train set:  
Accuracy: 0.83  
On the test set:  
Accuracy: 0.86  

![](https://i.imgur.com/5oV3a2L.png)

#### 查看决策边界图：

<pre class="prettyprint lang-python">
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**

![](https://i.imgur.com/QHmG4P2.png)

#### 小结：
> - 成本开始很高。这是因为对于大的随机值权重，最后一次激活（sigmoid）输出的结果非常接近0或1，如：当 $$ log(a^{[3]})=log(0) $$ 时，损失将变为无穷大。
> - 不良的初始化可能导致梯度消失/爆炸，这也会降低优化算法的速度。
> - 如训练此网络的时间更长，将会看到更好的结果，但使用过大的随机数进行初始化会降低优化速度。
> - 故使用小的随机值进行初始化效果更佳。

# 5 He方法的初始化

Xavier初始化, 是将权重 $$ W^{[l]} $$ 乘以一个比例因子 `sqrt(1./layers_dims[l-1])`  
He初始化, 是将权重 $$ W^{[l]} $$ 乘以一个比例因子 `sqrt(2./layers_dims[l-1])`  

#### 初始化方法：
 
<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_parameters_he
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2.0/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters
</pre>

#### 训练模型：

<pre class="prettyprint lang-python">
parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
</pre>

**运行结果如下：**  
Cost after iteration 0: 0.8830537463419761  
Cost after iteration 1000: 0.6879825919728063  
Cost after iteration 2000: 0.6751286264523371  
Cost after iteration 3000: 0.6526117768893807  
Cost after iteration 4000: 0.6082958970572937  
Cost after iteration 5000: 0.5304944491717495  
Cost after iteration 6000: 0.4138645817071793  
Cost after iteration 7000: 0.3117803464844441  
Cost after iteration 8000: 0.23696215330322556  
Cost after iteration 9000: 0.18597287209206828  
Cost after iteration 10000: 0.15015556280371808  
Cost after iteration 11000: 0.12325079292273548  
Cost after iteration 12000: 0.09917746546525937  
Cost after iteration 13000: 0.08457055954024274  
Cost after iteration 14000: 0.07357895962677366  

On the train set:  
Accuracy: 0.9933333333333333  
On the test set:  
Accuracy: 0.96  

![](https://i.imgur.com/CRoKyvb.png)

#### 查看决策边界图：

<pre class="prettyprint lang-python">
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
</pre>

**运行结果如下：**

![](https://i.imgur.com/7AxYUUA.png)

#### 小结：
> - 具有He初始化的模型在少量迭代中便可以非常好地将蓝点和红点分离。

# 6 总结

对于相同数量的迭代次数和相同的超参数，三种不同的初始化方法比较如下：

| 模型初始化方法 | 训练准确率 | 测试准确率 | 备注 |
| :------------: | :-----------: | :-----------: | :-----------: |
| 全为零 | 50.0 % | 50.0 % | 未能破坏对称性 |
| 随机值，但值较大 | 83.0 % | 86.0 % | 权重过大 |
| He方法 | 99.3 % | 96.0 % | 推荐的方法 |

> **注：**
> - 不同的初始化会导致不同的结果。
> - 随机初始化用于打破对称性并确保不同的隐藏单元可以学习不同的东西。
> - 不要初始化为太大的值。
> - He初始化适用于具有ReLU激活的网络。
