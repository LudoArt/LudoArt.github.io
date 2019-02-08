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
</head>

# 1 导入所需的包

<pre class="prettyprint lang-python">
import numpy as np  
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
​
%matplotlib inline
</pre>

>#### 包介绍
>**numpy** is the fundamental package for scientific computing with Python.
>**h5py** is a common package to interact with a dataset that is stored on an H5 file.
>**matplotlib** is a famous library to plot graphs in Python.
>**PIL** and **scipy** are used here to test your model with your own picture at the end.
>**lr_utils**是压缩包内给的自定义函数，功能是导入数据集

### PS：遇到的几个坑
#### 坑1：PIL包不支持python3以上
解决方案：安装PIL的分支Pillow

安装完成后，使用from PIL import Image就引用使用库了。比如：

<pre class="prettyprint lang-python">
from PIL import Image
im = Image.open("bride.jpg")
im.rotate(45).show()
</pre>

参考链接：[https://blog.csdn.net/dcz1994/article/details/71642979](https://blog.csdn.net/dcz1994/article/details/71642979)

#### 坑2：`%matplotlib inline`在PyCharm中报错
解决方案：使用`from matplotlib import pyplot as plt`

故导入包的代码可改为以下代码

<pre class="prettyprint lang-python">
import numpy as np
from matplotlib import pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
</pre>

参考链接：[https://blog.csdn.net/xinluqishi123/article/details/63523531](https://blog.csdn.net/xinluqishi123/article/details/63523531)

# 2 数据预处理
>#### 数据预处理的一般步骤
>- 弄清楚要处理的数据集的大小和形状
>- 重塑一些数据集的形状
>- “标准化”数据

## 具体步骤如下

<pre class="prettyprint lang-python">
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
</pre>

#### 2.2 识别数据集的大小和形状

<pre class="prettyprint lang-python">
# Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
</pre>

#### 2.3 重塑一些数据的形状

<pre class="prettyprint lang-python">
# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
</pre>

#### 2.4 “标准化”数据

<pre class="prettyprint lang-python">
# standardize our dataset.
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
</pre>

# 3 学习算法的一般体系结构
**Logistic Regression算法示意图如下：**
![Logistic Regression算法示意图](https://i.imgur.com/G08WJGS.png)
**Logistic Regression算法的数学表达式**
![Logistic Regression算法的数学表达式](https://i.imgur.com/Oydi1Eu.png)
**Logistic Regression算法的关键步骤**
- 初始化模型的参数
- 通过最小化损失来学习模型的参数
- 使用学习的参数在测试集上进行预测
- 分析结果并得出结论

# 4 构建Logistic Regression算法的各个部分
>构建一个神经网络的主要步骤如下：
>1、定义模型的结构
>2、初始化模型的参数
>3、循环
>- 计算当前的损失（正向传播过程）
>- 计算当前的梯度（反向传播过程）
>- 更新参数（梯度下降）
>
>通常单独构建1-3并将它们集成到一个我们称之为`model()`的函数中。

## 4.1 sigmoid函数
$$ sigmoid(w^{T}x+b)=\frac{1}{1+e^{-(w^{T}x+b)}} $$

<pre class="prettyprint lang-python">
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z
​
    Arguments:
    z -- A scalar or numpy array of any size.
​
    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))
    return s
</pre>

## 4.2 初始化参数

将参数w和b都初始化为0

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b
</pre>

## 4.3 前向传播和反向传播过程
正向传播过程：
- 获取数据集X
- 计算$$ A=σ(w^{T} + b) = (a^{(0)},a^{(1)},...,a^{(m-1)},a^{(m)}) $$
- 计算成本函数：$$ J=-\frac{1}{m}\sum_{i=1}^m y^{(i)}log(a^{(i)})+(1-y^{(i)})log(1-a^{(i)}) $$

反向传播过程：
- $$ \frac{∂J}{∂w} = \frac{1}{m}X(A - Y)^{T} $$
- $$ \frac{∂J}{∂b} = \frac{1}{m}\sum_{i=1}^m (a^{(i)} - y^{(i)}) $$

<pre class="prettyprint lang-python">
# GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
​
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
​
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X)+b)  # compute activation
    cost = -np.sum((np.dot(Y, np.log(A).T)+np.dot(1-Y, np.log(1-A).T)))/m  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (np.dot(X, (A-Y).T))/m
    db = (np.sum(A-Y))/m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost
</pre>

## 4.4 优化
- 已对参数做了初始化操作
- 已可进行成本函数和梯度的计算
- 现在，使用梯度下降来更新参数

<pre class="prettyprint lang-python">
# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w-learning_rate*dw
        b = b-learning_rate*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs
</pre>


## 4.5 预测
用已经训练好的参数w和b来预测数据集X，为其打上标签0或1
预测计算的主要步骤如下：
- 计算$$ \hat{Y}=A=\sigma(w^{T}X+b)  $$
- 计算结果若 <= 0.5，则打上标签0，计算结果若 > 0.5，则打上标签1；将预测结果存入向量*Y_prediction*中

<pre class="prettyprint lang-python">
# GRADED FUNCTION: predict
def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X)+b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert (Y_prediction.shape == (1, m))
    return Y_prediction
</pre>

# 5 将所有的函数整合到同一个模型中

<pre class="prettyprint lang-python">
# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros
    w = np.zeros((X_train.shape[0], 1))
    b = 0
    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d
</pre>

**使用以下代码来测试模型：**

<pre class="prettyprint lang-python">
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
</pre>

**预期输出：**

| Train Accuracy | Test Accuracy |
| :------------: | :-----------: |
| 99.04306220095694 % | 70.0 % |

# 6 进一步分析
## 选择学习率
学习率α决定了更新参数的速度。如果学习率太大，我们可能会“超调”最佳值。同样，如果它太小，我们将需要经过较多次的迭代来收敛到最佳值。使用良好调整的学习率至关重要。

**运行以下代码，查看不同的学习率会有怎样的影响：**

<pre class="prettyprint lang-python">
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')
​
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
​
plt.ylabel('cost')
plt.xlabel('iterations')
​
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
</pre>

**运行结果如下：**

| Learning Rate | Train Accuracy | Test Accuracy |
|:------------: | :------------: | :-----------: |
| 0.01 | 99.52153110047847 % | 68.0 % |
| 0.001 | 88.99521531100478 % | 64.0 % |
| 0.0001 | 68.42105263157895 % | 36.0 % |

![不同学习率下的成本函数曲线](https://i.imgur.com/ZtKvJwV.png)

**注：**
- 不同的学习率产生不同的成本，因此有不同的预测结果
- 较低的成本并不意味着更好的模型。必须检查是否有可能过度拟合。当训练精度远高于测试精度时，就会发生这种情况。
- 在深度学习中，通常：
 - 选择更好地降低成本函数的学习率
 - 如果模型过度拟合，使用其他技术来减少过度拟合。


>参考材料
>- http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
>- https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c