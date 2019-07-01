# Keras and TensorFlow essentials

TensorFlow is the most popular toolbox for deep learning. TensorFlow was created and is maintained by the Google Brain Team. Actually, TensorFlow is more than a deep learning toolbox: it is also a library for differentiable computing. Training deep neural networks requires computation of gradients of networks, which can be done using differentiable operators.

TensorFlow allows you to write computational graphs of operators. For each of these operators, the derivative is known. Therefore, you can backtrace the graph to find a chain of derivatives. Using the chain rule, you can construct a derivative for the full graph, which is useful in backpropagation of loss functions in neural networks (more on that later).

In addition, TensorFlow allows you to run these graphs on Central Processing Units (CPUs, where almost all code you have written in Python or Matlab runs on) *and* on Graphics Processing Units (GPUs). As the name implies, the latter are processors specialized in graphical computations (such as games). However, because that kind of computation (e.g. convolutions, inner products) significantly overlaps with the computations in neural networks, GPUs can accellerate those as well.

Summarized, TensorFlow excels at two things that are suited for training deep neural networks: differentiable computing, and GPU accelleration.

However, using TensorFlow directly means that you have to write a lot of code *de novo* every time you want to program a neural network. Therefore it is useful to run *front-ends* on top of TensorFlow that already contain the building blocks of neural networks. In these front-ends, popular layers, optimization methods, loss functions, and activation functions are already implemented. The most popular front-end to TensorFlow is Keras, and we will use it together with TensorFlow as back-end to build neural networks in this chapter.

Because Keras is such a popular front-end to TensorFlow, Keras has actually been included with TensorFlow since version 1.13.0.

In this tutorial, we will look at very basic TensorFlow code to explain the graph-based auto-differentiation, and then progress to a very simple network in Keras. We will slowly expand on this simple network until we have one that can perform a real task.

This tutorial is by no means a full overview of (convolutional) neural networks, but rather introduces you to the relevant tools to implement basic algorithms. This chapter will cover

* Symbolic computation in TensorFlow
* Logistic regression in Keras
* Multi-layer preceptrons in Keras
* Convolutional neural networks in Keras
* Accessing trained weights and outputs in neural networks



## Installation

You can install TensorFlow and Keras using `pip` on the Anaconda distribution of Python.

### Installation with CPU only version

Fire up a terminal, and type

```bash
$ pip install --user tensorflow
```


### Installation with GPU support

#### Preliminaries

If your computer has an Nvidia GPU, you can use the GPU version of TensorFlow. 
For that to work however, you will also need the CUDA and CUDNN libraries. As a rule of thumb, the latest versions of CUDA, CUDNN, and TensorFlow should be compatible. If you want to install TensorFlow on a Medical Image Analysis Group server, you can skip the following to [Installing `tensorflow-gpu`].

If you have a PC running Windows or Linux, you can find the CUDA installation instructions on the Nvidia website: [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) | [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

The CUDNN libraries can be downloaded [here](https://developer.nvidia.com/rdp/cudnn-download). Unfortunately, you have to become a member for this. The download should come with installation instructions.

#### Installing `tensorflow`

Fire up a terminal, and type

```bash
$ pip install --user tensorflow-gpu
```


## TensorFlow

The simplest functional TensorFlow program we could think of is adding two numbers. The code to do that in TensorFlow is this:

```python
import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
y = a + b

result = sess.run(y, feed_dict={a: 1, b:2})
print(result)
```

which will print '3' (thankfully...).

That seems like an awful lot of code for something so simple. Let's break it down to understand why we need all this session and placeholder misery.

1. The first line imports TensorFlow and gives it the alias `tf`.

2. Then we make a new TensorFlow session. A 'session' is more or less similar to a compiler. It will translate the code to high-performace `C++` and then execute it on the CPU or GPU, depending on which version of TensorFlow you installed.

3. Next, we make two placeholder variables, `a` and `b`. These are exactly what the sound like: place holders for data (i.e. tensors). TensorFlow is a lot more declarative about variables than Python: it forces us to declare beforehand what `a` and `b` will contain: 32-bit floating point values.

4. Then, we put the *operation of adding `a` and `b`* in `y`. This is an important distinction, we do not calculate anything on the line `y = a + b`, rather we say that if you want to run `y`, you need to add `a` and `b`.

5. Next, we do exactly that: we run `y` and feed `1` and `2` to the placeholders `a` and `b`. The `run()` method of the session object will translate the code to `C++` and run it on the CPU and return the result of the whole operation.

This is a rather contrived example, and of course calling the compile to do something this mundane will not lead to faster code (in fact, translating that code to C++ will be a lot slower than just adding those numbers). However, for large computational graphs that will be executed thousands of times, that translation is worth it.

By the way: if you run the code above you will get a lot of output on your screen, that tells you how the code is being run and on which processor. You can safely ignore that if everything works, but it can be useful to troubleshoot.



### Computation graphs

However small this example is, we can actually use it to show what TensorFlow is doing. If we cut-off the script above, and instead print `y`, we will see what it is doing behind the scene:

```python
import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
y = a + b

print(y)
```

This will print the following

```
Tensor("add:0", dtype=float32)
```

So what is `y`? It is a 32-bit float tensor, in this case with the rather unimaginative name 'add:0', which just is an automatic name. It is also an edge (that is a link between nodes) of the computation graph. The operators are the nodes of this graph. That plus operation between `a` and `b` is therefore a node. The two placeholders are edges as well. Hence, a graphical representation would be this:

```

    a         b
    |         |
    |         |
     --> + <-- 
         |
         |
         v

         y
```

This shows you, that in TensorFlow, the data (i.e the tensors) are the stuff that moves around along the edges of a graph of operators.

Why is that useful? It is the perfect way of describing the forward pass and backward pass of a neural network. For that backward pass we need...



### Derivatives on the graphs

Let's look at another example, where we multiply a number by a constant factor of 2:

```python
import tensorflow as tf

sess = tf.Session()

x = tf.placeholder(tf.float32)
y = 2 * x
```

The graph for this looks like this:

```

    2         x
    |         |
    |         |
     --> * <-- 
         |
         |
         v

         y
```

Nothing too exciting. What *is* exciting (ok, maybe not to everone) is the fact that TensorFlow can traverse the graph backwards to compute the derivative of `y` with respect to `x`. Note that that is only possible because the graphs are defined with placeholder inputs, and because of the graphical organization.

Of course, everyone knows that the derivative is `2` in this case. Let's see if TensorFlow agrees:

```python
import tensorflow as tf

sess = tf.Session()

x = tf.placeholder(tf.float32)
y = 2 * x

dy = tf.gradients(y, x)[0] # The derivative of y with respect to x

with sess.as_default():
    result = sess.run(gy, feed_dict={x: 0})

print(result)
```

Which will print `2.0`.


---

###### Exercises

1. We have now shown that the derivative is 2.0 at the point x = 0. Show that this holds for every point between -10 and 10. You can use the fact that the `run()` method also accepts NumPy arrays as inputs, and that TensorFlow can broadcast arrays just like Numpy.

    <details><summary>Answer</summary><p>

    ```python
    import tensorflow as tf
    import numpy as np

    sess = tf.Session()

    x = tf.placeholder(tf.float32)
    y = 2 * x

    dy = tf.gradients(y, x)[0] # The derivative of y with respect to x

    x_values = np.linspace(-10, 10, 1e6)

    with sess.as_default():
        result = sess.run(dy, feed_dict={x: x_values})

    print(np.all(result == np.float(2.0)))
    ```

    Should print `True`. Since we tested it for 1 million equidistant points between -10 and 10, we can safely assume TensorFlow gets the derivative right here.
    
    </p></details>

2. The derivative of sin(x) is cos(x). Show that TensorFlow agrees by plotting the derivative of sin(x) between $-2\pi$ and $2 \pi$. Note that for operators like sin and cos you need to use TensorFlow's versions `tf.sin()` and `tf.cos()`.

    <details><summary>Answer</summary><p>

    ```python
    import tensorflow as tf
    import numpy as np
    import matplotib.pyplot as plt

    sess = tf.Session()

    x = tf.placeholder(tf.float32)
    y = tf.sin(x)

    dy = tf.gradients(y, x)[0] # The derivative of y with respect to x

    x_values = np.linspace(-2 * np.pi, 2 * np.pi, 1e4)

    with sess.as_default():
        y_values = sess.run(dy, feed_dict={x: x_values})

    plt.plot(x_values, y_values)
    plt.show()
    ```

    This should show the graph of cos(x).
    
    </p></details>

---


## How TensorFlow computes derivatives

So how does TensorFlow know that the derivative of 2 * x is 2 and the derivative of sin(x) is cos(x)? Well, it just knows. It has a large look-up table for every operator in which it can look up what it should do. Does the function contain a constant multiplied by the variable? Then the derivative is equal to the constant. Does the function contain a sine? Then the derivative is a cosine. Is there a product of two functions? Then it will apply the product rule. It just looks these things up in a table.

Whenever a function is composed of multiple operators (i.e. sin(2 * x)), it applies operator *priority* and the chain rule. The priority is implicit in the graph. In the sin(2 * x) example, it understands that the graph looks like this:

```
    x       2
    |       |
    |       |
     -- * --
        |
        |
        v

       sin
        |
        |
        v

        y
```

Hence, to compute dy/dx, TensorFlow traverses the graph upwards, first computing the derivative of sin(g(x)) w.r.t. x. It looks up that for a function sin(g(x)) the derivative is cos(g(x)) * g'(x). Then, it needs to go further up the graph to compute g'(x) = 2 * x. Because it knows that the derivative of a constant times a variable is equal to the constant, it will return 2. Then it assembles the full derivative as being cos(2 * x) * 2. 

In essence, it does nothing more than what you learned in your Calculus courses. 

Now you have an idea of how auto-differentiation is implemented in TensorFlow, we are going to apply this on the simplest neural network possible with Keras. Be aware that using Keras, you will not have to explicitly define the computational graph anymore: this will happen automatically.

<!-- 
# Building a perceptron in TensorFlow

The simplest network that we can build is called a perceptron. A perceptron has an input layer, an output layer, and one so-called hidden layer. This hidden layer will have a matrix of weights $\mathbf{W}$ that is learned during training. In addition, there will be one additional bias weight vector $\mathbf{b}$ that will also be learned. The input layer will receive an input vector $\mathbf{x}$. It will compute the inner product $\mathbf{W} \dot \mathbf{x}$ and add the bias weights. On top of that, a so-called activation function will be applied. In this case, we choose the ReLU function. The output layer will return a vector $\mathbf{y}$ with outputs computed by

$$\mathbf{y} = ReLU(\mathbf{W} \dot \mathbf{x})$$

Let's build this model in TensorFlow.

```python

import tensorflow as tf
import keras

# Placeholders for input and output
x = tf.placeholder(dtype=tf.float32, shape=[None, 29*29])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# Variables for the weights and biases
w = tf.Variable(tf.zeros([29*29, 10]))
b = tf.Variable(tf.zeros([10]))

# Compute the dot product
product = tf.matmul(x, w)
with_bias = tf.add(product, b)
y_hat = tf.nn.softmax(logits)
```

That is the whole network! The `y` placeholder will hold the actual labels of the training set, and the `y_hat` variable will output estimates for both images in the training and validation sets.

 -->







## A practical example in Keras

The MNIST data set is an often-used example data set consisting of images of hand-written numbers between zero and nine. The task is to classify which number the images show.

To (down)load the data set, you can execute the following script:

```python
import keras
import numpy as np
import matplotlib.pyplot as plt

(training_images, training_labels), (val_images, val_labels) = mnist.load_data()
```

This will download the MNIST data set if that has not been done before, and divide it into a training set and a validation set.

Let's inspect what is in these sets. Show the size of `training_images` and `training_labels` using

```python
print(training_images.shape)
print(training_labels.shape)
```

This will show `(60000, 28, 28)` and `(60000,)`. Hence, the set contains 60,000 images of 28 x 28 and 60,000 labels.

Plot the first image by executing

```python
plt.imshow(training_images[0])
plt.show()
```

This should print a 5, and the corresponding label, `training_labels[0]` should reflect that.















# Neural networks in Keras


# Convolutional neural networks



