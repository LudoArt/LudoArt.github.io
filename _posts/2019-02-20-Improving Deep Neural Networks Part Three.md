---
layout:     post
title:      Improving Deep Neural Networks Part Three
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

# 改进深度神经网络之三：梯度校验

# 1 导入所需的包

<pre class="prettyprint lang-python">
# Packages
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
</pre>

# 2 梯度校验是如何实现的

反向传播中计算梯度 $$ \frac{\partial J}{\partial \theta} $$ ,此处的 $$ \theta $$ 代表模型的参数， $$ J $$ 是成本函数。    

导数（或者说梯度）的定义如下：  

$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} $$  

通过上述的公式，使用较小的 $$ \varepsilon $$ 值来确保你的 $$ \frac{\partial J}{\partial \theta} $$ 计算正确。  

# 3 一维的梯度校验

假设一维线性函数为： $$ J(\theta)=\theta x $$ 。一维线性模型如下图所示。

![](https://i.imgur.com/lzKdBF0.png)

#### 3.1 前向传播与反向传播过程

<pre class="prettyprint lang-python">
# GRADED FUNCTION: forward_propagation
def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    J = x * theta 
    
    return J
</pre>

<pre class="prettyprint lang-python">
# GRADED FUNCTION: backward_propagation
def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """
    
    dtheta = x
    
    return dtheta
</pre>

#### 3.2 梯度校验

> 步骤如下：
> - 计算 "gradapprox" ：
>   - $ \theta^{+} = \theta + \varepsilon $
>   - $ \theta^{-} = \theta - \varepsilon $
>   - $ J^{+} = J(\theta^{+}) $
>   - $ J^{-} = J(\theta^{-}) $
>   - $ gradapprox = \frac{J^{+} - J^{-}}{2  \varepsilon} $
> - 使用反向传播计算梯度，将计算的梯度存在变量 "grad" 中。
> - 使用以下公式计算 "gradapprox" 和 "grad" 之间的相对差异：  
>   
> $$ difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} $$  
>   
> - 如果这个差异很小（比如小于 $$ 10^{-7} $$ ），即代表梯度计算无误。否则，梯度计算中可能存在错误。

<pre class="prettyprint lang-python">
# GRADED FUNCTION: gradient_check
def gradient_check(x, theta, epsilon = 1e-7):
    """
    Implement the backward propagation presented in Figure 1.
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    thetaplus = theta + epsilon                          # Step 1
    thetaminus = theta - epsilon                         # Step 2
    J_plus = forward_propagation(x, thetaplus)           # Step 3
    J_minus = forward_propagation(x, thetaminus)         # Step 4
    gradapprox = (J_plus - J_minus)/(2 * epsilon)        # Step 5
    
    # Check if gradapprox is close enough to the output of backward_propagation()
    grad = backward_propagation(x, theta)
    
    numerator = np.linalg.norm(grad - gradapprox)                      # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)    # Step 2'
    difference = numerator / denominator                               # Step 3'
    
    if difference < 1e-7:
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")
    
    return difference
</pre>

**使用以下代码进行测试：**

<pre class="prettyprint lang-python">
x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))
</pre>

**运行结果如下：**  
The gradient is correct!  
difference = 2.919335883291695e-10  

# 4 N维的梯度校验

N维线性模型如下图所示（ *LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID* ）：

![](https://i.imgur.com/6aNpUiF.png)


#### 4.1 前向传播与反向传播过程

<pre class="prettyprint lang-python">
def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples 
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    
    Returns:
    cost -- the cost function (logistic cost for one example)
    """
    
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache
</pre>

<pre class="prettyprint lang-python">
def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()
    
    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) * 2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 4./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
</pre>

#### 4.2 梯度校验

同样的，还是需要比较 "gradapprox" 和 "grad" 之间的相对差异。计算 "gradapprox" 的公式如下：  

$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} $$  

不同的是，此时的 $ \theta $ 不再是一个标量，而是一个 `dictionary` 类型的变量 `parameters` 。使用函数 `dictionary_to_vector()` 将变量 `parameters` 转换为一个向量 `values` ，反函数 `vector_to_dictionary` ，可以将 `values` 转换回 `parameters` 。如下图所示。

![](https://i.imgur.com/7acbKS3.png)

梯度计算的步骤如下：

For each i in num_parameters:
- 计算 `J_plus[i]`:
    1. 使用 `np.copy(parameters_values)` 初始化 $ \theta^{+} $ 
    2. 令 $ \theta^{+}_i $ 为 $ \theta^{+}_i + \varepsilon $
    3. 使用 `forward_propagation_n(x, y, vector_to_dictionary(`$ \theta^{+} $ `))` 计算 $ J^{+}_i $     
- 计算 `J_minus[i]`: 使用 $\theta^{-}$ 做与上面步骤相似的操作
- 计算 $ gradapprox[i] = \frac{J^{+}_i - J^{-}_i}{2 \varepsilon} $

接下来便可以使用以下公式来计算 "gradapprox" 和 "grad" 之间的相对差异：  
$$ difference = \frac {\| grad - gradapprox \|_2}{\| grad \|_2 + \| gradapprox \|_2 } $$

<pre class="prettyprint lang-python">
# GRADED FUNCTION: gradient_check_n
def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        thetaplus = np.copy(parameters_values)                                          # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                     # Step 2
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))     # Step 3
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                         # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                   # Step 2        
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))   # Step 3
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)                                       # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                     # Step 2'
    difference = numerator / denominator                                                # Step 3'

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
</pre>

**使用以下代码进行测试：**

<pre class="prettyprint lang-python">
X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)
</pre>

**运行结果如下：**  
There is a mistake in the backward propagation! difference = 0.2850931566540251

#### 4.3 反向传播中的bug修复

通过上面的梯度校验，发现反向传播过程中有计算错误，修改其错误代码如下：

<pre class="prettyprint lang-python">
# dW2 = 1./m * np.dot(dZ2, A1.T) * 2 有误的
dW2 = 1./m * np.dot(dZ2, A1.T) # 修改的

# db1 = 4./m * np.sum(dZ1, axis=1, keepdims = True) 有误的
db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True) # 修改的
</pre>

# 5 总结

> - 梯度校验是验证 `grads` *（由反向传播计算得出）* 与 `gradapprox` *（由前向传播计算得出）* 之间的接近程度。
> - 梯度校验很慢，因此不可每次训练迭代中都运行它。通常只使用它以确保代码是的正确性，然后将其关闭。
> - 梯度校验不适用于Dropout。通常在没有Dropout的情况下运行梯度校验算法，以确保反向传播过程的计算无误，然后再添加Dropout技术。
