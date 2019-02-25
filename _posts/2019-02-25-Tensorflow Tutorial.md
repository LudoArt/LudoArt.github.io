---
layout:     post
title:      Tensorflow Turtorial
subtitle:   Building a neural network in tensorflow
date:       2019-02-25
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 人工智能
    - 深度学习
    - Tensorflow
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

# 1 探索Tensorflow库

首先，导入所需的库，如下：

<pre class="prettyprint lang-python">
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)
</pre>

尝试计算如下公式：  

$$ loss = \mathcal{L}(\hat{y}, y) = (\hat y^{(i)} - y^{(i)})^2 $$

<pre class="prettyprint lang-python">
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss
</pre>

> **在TensorFlow中编写和运行程序具有以下步骤：**
> - 创建Tensors（变量，此时还未计算变量的值）。
> - 写下这些Tensors的运算（等式）。
> - 创建Sesssion。
> - 运行Session。这将会计算前面写下的等式。

再看个小例子：
<pre class="prettyprint lang-python">
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)  # will not see 20!

sess = tf.Session()
print(sess.run(c))  # 20
</pre>

占位符（ *placeholder* ）是一个对象，其值只能在以后指定。要指定占位符的值，可以使用 “*feed dictionary*” （ `feed_dict` 变量）传入值。如下面这个例子所示：

<pre class="prettyprint lang-python">
# Change the value of x in the feed_dict
x = tf.placeholder(tf.int64, name = 'x')
sess = tf.Session()
print(sess.run(2 * x, feed_dict = {x: 3}))  # 6
sess.close()
</pre>

## 1.1 线性函数

计算以下的等式：  

$$ Y = WX + b $$

此处的 $W$ 和 $X$ 是随机数值的矩阵， $b$ 是随机数值的向量。

<pre class="prettyprint lang-python">
# GRADED FUNCTION: linear_function
def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """
    np.random.seed(1)

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)
    Y = tf.add(tf.matmul(W, X), b)

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    sess = tf.Session()
    result = sess.run(Y)

    # close the session
    sess.close()

    return result
</pre>

## 1.2 计算sigmoid函数

Tensorflow提供各种常用的神经网络函数，如 `tf.sigmoid` 和 `tf.softmax` 。

<pre class="prettyprint lang-python">
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """

    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name="x")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above.
    # You should use a feed_dict to pass z's value to x.
    sess = tf.Session()
    # Run session and call the output "result"
    result = sess.run(sigmoid, feed_dict={x: z})
    sess.close()

    return result
</pre>

**注：在tensorflow中创建和使用会话（Session）有两种典型方法：**  

**方法一：**  
<pre class="prettyprint lang-python">
sess = tf.Session()
# Run the variables initialization (if needed), run the operations
result = sess.run(..., feed_dict = {...})
sess.close() # Close the session
</pre>

**方法二：**  
<pre class="prettyprint lang-python">
with tf.Session() as sess: 
    # run the variables initialization (if needed), run the operations
    result = sess.run(..., feed_dict = {...})
    # This takes care of closing the session for you :)
</pre>

## 1.3 计算成本

在Tensorflow中，可以使用其内置的函数 `tf.nn.sigmoid_cross_entropy_with_logits` 来计算以下公式：

$$ J = - \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log \sigma(z^{[2](i)}) + (1-y^{(i)})\log (1-\sigma(z^{[2](i)})\large )\small $$

> **参考链接：**  
> [ `tf.nn.sigmoid_cross_entropy_with_logits` 的官方文档](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)

<pre class="prettyprint lang-python">
# GRADED FUNCTION: cost
def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    # Create the placeholders for "logits" (z) and "labels" (y)
    z = tf.placeholder(tf.float32, name="z")
    y = tf.placeholder(tf.float32, name="y")

    # Use the loss function
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    # Create a session
    sess = tf.Session()

    # Run the session
    cost = sess.run(cost, feed_dict={z: logits, y: labels})

    # Close the session
    sess.close()

    return cost
</pre>

## 1.4 使用One Hot编码

很多时候，在深度学习中，你将得到一个y向量，其数字范围从0到C-1，其中C是分类的数量。如果C是4，那么需要转换y向量，如下所示：

![](https://i.imgur.com/wfP75ic.png)

要在numpy中进行此转换，可能需要编写几行代码。在tensorflow中，可以使用一行代码： `tf.one_hot(labels, depth, axis)` 

> **参考链接：**  
> [ `tf.one_hot(labels, depth, axis)` 的官方文档](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/one_hot)

<pre class="prettyprint lang-python">
# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """
    # Create a tf.constant equal to C (depth), name it 'C'.
    C = tf.constant(C, name="C")

    # Use tf.one_hot, be careful with the axis
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    # Create the session
    sess = tf.Session()

    # Run the session
    one_hot = sess.run(one_hot_matrix)

    # Close the session
    sess.close()

    return one_hot
</pre>

## 1.5 初始化为0和1

要初始化为1，调用的函数是 `tf.ones（）` 。要初始化为0，调用的函数是 `tf.zeros（）` 。

<pre class="prettyprint lang-python">
# GRADED FUNCTION: ones
def ones(shape):
    """
    Creates an array of ones of dimension shape

    Arguments:
    shape -- shape of the array you want to create

    Returns:
    ones -- array containing only ones
    """
    # Create "ones" tensor using tf.ones(...).
    ones = tf.ones(shape)

    # Create the session
    sess = tf.Session()

    # Run the session to compute 'ones'
    ones = sess.run(ones)

    # Close the session
    sess.close()

    return ones
</pre>

# 2 使用Tensorflow搭建一个神经网络

## 2.0 问题描述：SIGNS数据集

- **训练集：**1080个图像（64×64像素），每张图像有一个符号，表示从0到5中的某个数字（每个数字180个图像）。  
- **测试集：**120个图像（64×64像素），每张图像有一个符号，表示从0到5中的某个数字（每个数字20个图像）。  

以下是每个数字的示例，以及解释我们如何表示标签。

![](https://i.imgur.com/VmAnuws.png)

<pre class="prettyprint lang-python">
# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
</pre>

要搭建的模型是一个三层模型： *LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX* 

## 2.1 创建placeholder

<pre class="prettyprint lang-python">
# GRADED FUNCTION: create_placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(dtype=float, shape=[n_x, None], name="X")
    Y = tf.placeholder(dtype=float, shape=[n_y, None], name="Y")

    return X, Y
</pre>

## 2.2 初始化变量

<pre class="prettyprint lang-python">
# GRADED FUNCTION: initialize_parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    tf.set_random_seed(1)  # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    return parameters
</pre>

> **注：此时的参数还未参与计算**

## 2.3 Tensorflow中的前向传播

<pre class="prettyprint lang-python">
# GRADED FUNCTION: forward_propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)   # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                 # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                 # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,A2) + b3

    return Z3
</pre>

**在前向传播中，不会输出任何缓存（cache），具体原因见后面的反向传播过程。**

## 2.4 计算成本

使用函数 `tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))` 即可计算成本。

> **参考链接：**  
> [ `tf.reduce_mean` 官方文档](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean "reduce_mean")
> [ `tf.nn.softmax_cross_entropy_with_logits` 官方文档](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)

<pre class="prettyprint lang-python">
# GRADED FUNCTION: compute_cost
def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost
</pre>

## 2.5 反向传播和更新参数

所有的反向传播和参数更新都在一行代码中处理。计算成本函数后。将创建一个 “优化器（ `optimizer` ）” 对象。运行 `tf.session` 时，必须将此对象与成本一起调用。调用时，它将使用所选方法和学习速率对给定成本执行优化。  

例如，对于梯度下降，优化器将是：  
 `optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)`   

要进行优化，需要：  
 `_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})`   

> **注:** 编码时，我们经常使用 `_` 作为 “一次性” 变量来存储我们以后不需要使用的值。这里， `_` 是优化器 `optimizer` 的评估值，我们无需使用它（ `c` 是成本变量 `cost` 的值）

## 2.6 搭建模型

<pre class="prettyprint lang-python">
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters
</pre>

**使用如下代码来训练和测试模型：**

<pre class="prettyprint lang-python">
parameters = model(X_train, Y_train, X_test, Y_test)
</pre>

**运行结果如下：**  
Cost after epoch 0: 1.855702  
Cost after epoch 100: 1.016458  
Cost after epoch 200: 0.733102  
Cost after epoch 300: 0.572938  
Cost after epoch 400: 0.468799  
Cost after epoch 500: 0.380979  
Cost after epoch 600: 0.313819  
Cost after epoch 700: 0.254258  
Cost after epoch 800: 0.203795  
Cost after epoch 900: 0.166410  
Cost after epoch 1000: 0.141497  
Cost after epoch 1100: 0.107579  
Cost after epoch 1200: 0.086229  
Cost after epoch 1300: 0.059415  
Cost after epoch 1400: 0.052237  
   
Parameters have been trained!   
Train Accuracy: 0.9990741  
Test Accuracy: 0.71666664  

![](https://i.imgur.com/GzI7cEo.png)

> **注：**
> - 模型似乎足够大，足以适应训练集。但是，考虑到训练集和测试集准确率之间的差异，可以尝试添加L2或Dropout正则化以减少过度拟合。
> - 将会话（Session）视为训练模型的代码块。每次在小批量（minibatch）运行会话时，它都会训练参数。

# 3 总结

> - Tensorflow是深度学习中使用的编程框架。
> - Tensorflow中的两个主要对象是张量（Tensors）和运算（Operators）。
> - 在tensorflow中编码时，您必须执行以下步骤：
>   - 创建一个图，包含张量（Variables，Placeholders...）和运算（tf.matmul，tf.add，...）；
>   - 创建一个会话（Session）；
>   - 初始化会话（Session）；
>   - 运行会话（Session）以计算图（graph），即 `session.run()` 。
> - 在 `model()` 中可以看到，可以多次计算图。
> - 在“优化器（optimizer）”对象上运行会话时，将自动完成反向传播和优化过程。
