# Numerical and scientific computing in Python

**Contents**

* [NumPy](#numpy)
    * [N-dimensional arrays](#n-dimensional-arrays)
    * [Indexing arrays](#indexing-arrays)
    * [Assigning at indices](#assigning-at-indices)
    * [Reshaping and transposing](#reshaping-and-transposing)
    * [Array products](#array-products)
        * [Elementwise products](#elementwise-products)
        * [Dot products](#dot-products)
    * [Constructing matrices](#constructing-matrices)
    * [Data types in NumPy](#data-types-in-numpy)
    * [Mathematical functions](#mathematical-functions)
    * [Linear algebra](#linear-algebra)
    * [Loading and saving data](#loading-and-saving-data)
    * [Random numbers](#random-numbers)
* [Matplotlib](#matplotlib)
    * [Basic plotting](#basic-plotting)
    * [Figures and axes](#figures-and-axes)
    * [Titles, legends, labels](#titles-legends-labels)
    * [Other plots and diagrams](#other-plots-and-diagrams)
    * [Showing images](#showing-images)
* [SciPy](#scipy)

So far, we have seen basic functionality of Python required to write your own
scientific programs. What has been lacking is a way to implement numeric code.

In this chapter we will look at four libraries that are often used for working
with Python in numeric and scientific computing: NumPy, SciPy, Matplotlib, and
Skimage. We start with the basics of NumPy, look at some linear algebra functions and importing and exporting data, and then we will move onto visualizing (plotting) that data
with Matplotlib and processing images with SciPy.

## NumPy

NumPy is a library: a collection of classes, constants, and functions that you can import to
use in your own code. NumPy supports multi-dimensional arrays and matrices, linear
algebra operations, and has a large collection of functions for mathematical operations. The full
NumPy documentation can be found [here](https://docs.scipy.org/doc/numpy-1.15.0/reference/index.html).
If you are lost, that website is a good start on the quest for a solution.

To start out, first import the NumPy library.

```python
import numpy as np
```

This statement imports the library. If you use a function, class, or constant from the library, you
need to prefix it with `np.`, for example if you want to use the NumPy's square root function `sqrt`
you type:

```python
print(np.sqrt(9))  # prints 3.0
```

### N-dimensional arrays

The most used class in NumPy is the `array` class, which you can use to define
matrices and N-dimensional arrays. For example, you can define

```python
matrix = np.array(
    [   
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)
```

which assigns the two-dimensional array

<a href="https://www.codecogs.com/eqnedit.php?latex=\left(\begin{matrix}1&space;&&space;2&space;&&space;3\\\\4&space;&&space;5&space;&&space;6\\\\7&space;&&space;8&space;&&space;9\end{matrix}\right)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left(\begin{matrix}1&space;&&space;2&space;&&space;3\\\\4&space;&&space;5&space;&&space;6\\\\7&space;&&space;8&space;&&space;9\end{matrix}\right)" title="\left(\begin{matrix}1 & 2 & 3\\\\4 & 5 & 6\\\\7 & 8 & 9\end{matrix}\right)" /></a>

to the variable `matrix`. As you can see, the matrix is defined as a list of lists. 
For a three-dimensional array, you would need a list of lists of lists. For a simple
vector, you can just use a single list:

```python
vector = np.array([1, 2, 3])
```

To find out how many dimensions an array has, you can use the `ndim` property of
an array:

```python
print(matrix.ndim)  # prints 2
print(vector.ndim)  # prints 1
```

Note that `ndim` is not a method: it is not followed by parentheses, and does not
actively have to compute something. Another property you can use is `shape` which
gives you the size along each axis of the array:

```python
print(matrix.shape)  # prints (3, 3) because matrix is a 3x3 array
print(vector.shape)  # prints (3, ) because vector is a 3x1 array
```

The order of the shapes is similar to the ordering of matrix axes: first the
length in *i*-direction (downwards) is given, then the *j*-direction (left-to-right).
You can also use the `len` function to get the length of the first axis:

```python
m = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(len(m))  # prints 4
```

### Indexing arrays

Indexing an array is very similar to indexing lists:

```python
print(vector[0])  # prints the first element (i.e. 1)
print(vector[1:])  # prints elements starting from index 1 (i.e. array([2, 3]))
print(matrix[0])  # prints the first element (i.e. array([1, 2, 3]))
print(matrix[1:])  # prints elements starting from index 1 (i.e. array([[4, 5, 6], [7, 8, 9]]))
```

In the `matrix[0]` example, you obtain a new array, that you can index again, e.g.:

```python
print(matrix[0][1])  # prints the second element of the first element (array([2, 3])[1], i.e. 3) 
```

You can write this more cleanly as:

```python
print(matrix[0, 1])
```

Of course you can also slice an array in this way. If you want to get all elements
along a certain axis, you can use `:` as an index, for example, 

```python
print(matrix[:, 1])
```

will print all rows (indicated by the `:`) and the second column (indicated by the `1`).


---

###### Exercises

Given this code

```python
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
```
How do you obtain

* The first row of the matrix?
    
    <details><summary>Answer</summary><p>

    `m[0]`

    </p></details>

* The second row of the matrix?
    
    <details><summary>Answer</summary><p>

    `m[1]`

    </p></details>

* Every second row of the matrix?
    
    <details><summary>Answer</summary><p>

    `m[::2]`

    </p></details>

* Every second column of the matrix?
    
    <details><summary>Answer</summary><p>

    `m[:, ::2]`

    </p></details>

* The last column of the matrix?
    
    <details><summary>Answer</summary><p>

    `m[:, -1]`

    </p></details>

---


### Assigning at indices

Like with lists, you can assign directly to parts of an array, for example,

```python
matrix[:2, :2] = np.array([[10, 20], [30, 40]])
```

will replace the values in the top left of the matrix, such that the matrix
now contains    

```python
[
    [10, 20, 2],
    [30, 40, 6],
    [ 7,  8, 9]
]
```

Be careful to match the shape of the matrix on the right side of the `=`-sign
with the indices on the left: if you are assigning to a 2×2 block, make sure
the array on the right is the same size. NumPy will try to 'broadcast' the array
on the righthandside such that it fits the part you want to assign to, for example for

```python
matrix[:2, :2] = np.array([[10, 20]])
```

the righthandside has shape (1, 2), but the lefthandside has
(2, 2). In that case, NumPy will stretch the righthandside to fit that area of the array:

```python
[
    [10, 20, 2],
    [10, 20, 6],
    [ 7,  8, 9]
]
```

If the righthandside has shape (2, 1), for example,

```python
matrix[:2, :2] = np.array([[10], [20]])
```

the result will be:

```python    
[
    [10, 10, 2],
    [20, 20, 6],
    [ 7,  8, 9]
]
```

If you only put one value on the righthandside, NumPy will just put that value
in every element that you are assigning to. For example, 

```python
matrix[:2, :2] = 42
```

gives the result:

```python
[
    [42, 42, 2],
    [42, 42, 6],
    [ 7,  8, 9]
]
```
Be careful: like lists, arrays are mutable variables, which means they assign
*by reference*, i.e.:

```python
a = np.array([1, 2, 3])
b = a
a[0] = 0  # also changes b!
print(b)  # prints [0, 2, 3]
```

---

###### Exercises

Given the matrix 
```python
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```
write code that

* Replaces the rightmost column with zeroes
    <details><summary>Answer</summary><p>

    `m[:, -1] = 0`

    </p></details>

* Replaces every second row with ones
    <details><summary>Answer</summary><p>

    `m[::2] = 1`

    </p></details>

* Puts a `2` in every element on the diagonal
    <details><summary>Answer</summary><p>

    ```python
    for i in range(len(m)):
        m[i, i] = 2
    ```

    </p></details>

---


### Reshaping and transposing

Sometimes it can be useful to change the shape of an array. The simplest example
of this is 'flattening' using the `flatten()` method, which turns any array into a one-dimensional array.

```python
m = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(m.flatten())  # prints [1, 2, 3, 4, 5, 6, 7, 8]
```

You can also reshape an array:

```python
m = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(m.shape)  # prints (4, 2)
k = m.reshape(2, 4)
print(k.shape)  # prints (2, 4)
print(k)  # prints [[1, 2, 3, 4], [5, 6, 7, 8]]
```

The `reshape()` method simply flattens an array, and reshapes the array to the shape
that you supply to the method. You can let NumPy figure out what the length of one axis should be, with `-1`:

```python
v = np.array([1, 2, 3])
v.reshape(-1, 1)  # means: the last axis should have length 1, 
                  # and NumPy will figure out the length of the first axis (3 in this case)
print(v)  # prints [[1], [2], [3]]
```

Finally, arrays have a `transpose()` method that reverses the order of the axis. For a 2D matrix this gets you the transpose of the matrix.
For higher dimensionality arrays, it is also possible to supply a new order of the axes.

```python
tensor = np.array([
    [[1, 2], [3, 4], [5, 6], [7, 8]],
    [[1, 2], [3, 4], [5, 6], [7, 8]],
    [[1, 2], [3, 4], [5, 6], [7, 8]],
])
print(tensor.shape)  # prints (3, 4, 2)

tensor2 = tensor.transpose()
print(tensor2.shape)  # prints (2, 4, 3)

tensor3 = tensor.transpose(0, 2, 1)  # The order of the axis will become 0, 2, 1
print(tensor3.shape)  # prints (3, 2, 4)
```

Note that none of the methods change the arrays on which they act, which means that
you have to put the result into a new variable if you want to use it.


---

###### Exercises

* Reshape the array `m = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])` to the shape `(3, 3)`

    <details><summary>Answer</summary><p>

    This results in a `ValueError` because the shape does not fit the array.

    </p></details>

* What happens when you transpose a vector like np.array([1, 2, 3, 4])? Why?

    <details><summary>Answer</summary><p>

    `transpose()` reverses the order of the axes. For a 1D array reversing the order has no effect.

    </p></details>

* Given the matrix `np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`, write code that rotates the elements 90 degrees anticlockwise, such that you obtain a matrix

    ```python
    [
        [3, 6, 9],
        [2, 5, 8],
        [1, 4, 7]
    ]
    ```

    <details><summary>Answer</summary><p>

    ```python
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rotated_a = a.transpose()[::-1]
    ```

    </p></details>

---




### Array products

#### Elementwise products

Any NumPy array can be multiplied with a scalar using the `*`-operator, for example `2 * np.array([1, 2, 3])` and `np.array([1, 2, 3]) * 2` yield `np.array([2, 4, 6])`. You can also do multiplication of two arrays. The multiplication will be elementwise, i.e. `np.array([1, 2, 3]) * np.array([2, 3, 4])` yields `np.array([2, 6, 12])`. The results are again subject to broadcasting, so be careful. Some examples: 

```python
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
c = np.array([[1, 2, 3],[2, 3, 4]])
```

* `a * b` yields `[2, 6, 12]`, i.e. each element in `a` is multiplied with the corresponding element in `b`;
* `a * c` and `c * a` yield `[[1, 4, 9], [2, 6, 12]]`, i.e. `a` is broadcasted to `[[1, 2, 3], [1, 2, 3]]`, and then multiplied elementwise;
* `a * b * c`, `a * c * b`, `b * a * c`, `b * c * a`, `c * a * b`, and `c * b * a`, all result in `[[2, 12, 36], [4, 18, 48]]`, i.e. `a` and `b` are broadcasted to the same size as `c`, and then multiplied elementwise.


#### Dot products

NumPy also supports dot products, for example the inner product of two vectors:

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(v1.dot(v2))  # prints 32
```

The inner product also can be applied to two matrices, as you are probably used to from your linear algebra courses. In that case you need to multiply N ×M arrays with M×N arrays to obtain a N×N array, for example:

```python
a1 = np.array([[1, 2, 3], [4, 5, 6]])
a2 = np.array([[1, 2], [3, 4], [5, 6]])
a1.dot(a2)  # yields [[22, 28], [49, 64]]
a2.dot(a1)  # yields [[9, 12, 15], [19, 26, 33], [39, 40, 51]]
```

If you want to multiply a N×M matrix with a vector, you need to make sure that the vector is represented by a length N array, for example:

```python
v1 = np.array([1, 2, 3])
a1.dot(v1)  # yields [14, 32]
```

Alternatively, you can do the same thing using an N×1 matrix:

```python
v1 = np.array([[1], [2], [3]])
# or v1 = v1.reshape(-1, 1)
a1.dot(v1)
```

---

###### Exercises

1. What happens when you try to multiply a 2×3 array with a 3×2 array elementwise? Can you explain why?
    <details><summary>Answer</summary><p>

    You get a `ValueError` because the shapes are incompatible: NumPy cannot broadcast one array to that of the other.

    </p></details>

2. Calculate the dot-product between the matrix

    <a href="https://www.codecogs.com/eqnedit.php?latex=\left(\begin{matrix}1&space;&&space;0&space;&&space;1\\\\2&space;&&space;1&space;&&space;2\\\\1&space;&&space;0&space;&&space;1\end{matrix}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left(\begin{matrix}1&space;&&space;0&space;&&space;1\\\\2&space;&&space;1&space;&&space;2\\\\1&space;&&space;0&space;&&space;1\end{matrix}\right)" title="\left(\begin{matrix}1 & 0 & 1\\\\2 & 1 & 2\\\\1 & 0 & 1\end{matrix}\right)" /></a>

    and its transpose.
    <details><summary>Answer</summary><p>

    ```python
    a = np.array([[1, 0, 1], [2, 1, 2], [1, 0, 1]])
    b = a.dot(a.transpose())  # yields [[2, 4, 2], [4, 9, 4], [2, 4, 2]]
    ```

    </p></details>

---



### Constructing matrices

NumPy has a few default matrices that you can obtain using the functions `np.ones()`, `np.zeros()`, and `np.eye()`. The first two generate matrices filled with ones and zeros. Their argument is a shape tuple, for example `np.zeros((4, 2))` gives you `[[0, 0], [0, 0], [0, 0], [0, 0]]`. `np.eye()` generates identity matrices, for example `np.eye(3)` generates `[[1, 0, 0], [0, 1, 0], [0, 0, 1]]`.

These construction functions are useful when you are filling large matrices using code. In the previous chapter, you have seen that you can construct lists using the `append()` method to add elements to a list. For NumPy arrays such functionality does not exist, because this method is generally quite slow. It is much faster to generate an array of a certain size using `np.ones()` or `np.zeros()`, and then fill the elements. For example

```python
a = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        a[i, j] = i * j
```

fills an empty 5×5 matrix.

There are also two functions that generate template vectors. `np.arange()` is similar to the function `range()`, but returns an `ndarray` instead of a range object, and also allows decimal values as arguments:

```python
np.arange(10)  # returns ≈
np.arange(5, 10)  # returns array([5, 6, 7, 8, 9])
np.arange(5.2, 10.2)  # returns array([5.2, 6.2, 7.2, 8.2, 9.2])
np.arange(0, 10, 2.5)  # returns array([0, 2.5, 5, 7.5])
```

The `np.linspace()` function does something similar, but instead of defining the start, end (not included), and step, you define that start, end (included!), and number of elements you want. For example:

```python
np.linspace(0, 10, 11)  # returns array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), notice that 10 is included
np.linspace(2, 5, 7)  # returns array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
np.linspace(0, 1, 6)  # returns array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
```


### Data types in NumPy

NumPy has its own variant of numeric types. It does not use the built-in Python types `int` and `float`, but has types for certain precisions, which is the number of bytes used to store a value. For example, NumPy has types to store integers called `np.int8`, `np.int16`, `np.int32`, and `np.int64`. `np.int8` can store numbers between -128 and 127, while the `np.int64` type can store numbers between -9223372036854775808 and 9223372036854775807. A variant of the integer types called 'unsigned integers' can only store positive values, and are indicated by `np.uint8`, `np.uint16`, `np.uint32`, and `np.uint64`. Their ranges all start at zero, which results in the `np.uint8` being able to store values between 0 and 255. Floats are represented by either `np.float32` or `np.float64`, and are always signed, i.e. they can also store negative values.

Naturally, an `np.int64` takes up eight times as much space in memory compared to an `np.int8`. When you are having a lot of data (large medical images for example) in memory, it becomes important what type you use. That is why every construction function mentioned in the previous section has a `dtype` parameter in which you can set the datatype, for example:

```python
a = np.arange(0, 10)  # returns array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.arange(0, 10, dtype=np.float32)  # returns array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
```

You can check out the datatype of an array using the `dtype` attribute:

```python
print(a.dtype)  # prints dtype('int64')
print(b.dtype)  # prints dtype('float32')
```

You can also change the data type of any existing array using the `as_type()` method:

```python
c = a.as_type('float64')
print(c.dtype)  # prints dtype('float64')
```


### Mathematical functions

NumPy implements many mathematical functions. In general, they do what you expect. One thing to keep in mind is that if you supply an array instead of a scalar, the function will be evaluated elementwise.

| syntax                                                                          | meaning                                                                               |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------                   |
| `np.sin()`, `np.cos()`, `np.tan()`, `np.arcsin()`, `np.arccos()`, `np.arctan()` | Trigonometric functions, angles are measured in radians                               |
| `np.exp()`                                                                      | Exponential with base *e*                                                             |
| `np.log()`                                                                      | Logarithm with base *e*                                                               |
| `np.log10()`                                                                    | Logarithm with base 10                                                                |
| `np.abs()`                                                                      | Returns the absolute value \|x\|                                                      |
| `np.floor()`                                                                    | Round downwards, i.e. `np.floor(3.5)` yields `3`                                      |
| `np.ceil()`                                                                     | Round upwards, i.e. `np.ceil(3.5)` yields `4`                                         |
| `np.max()`, `np.min()`                                                          | Return the minimum or maximum of a list or array, i.e. `np.min([3, 2, 7])` yields `2` |
| `np.argmax()`, `np.argmin()`                                                    | Return the index of the minimum or maximum of a list or array                         |
| `np.mean()`                                                                     | Return the mean of a list or array                                                    |
| `np.median()`                                                                   | Return the median of a list or array                                                  |
| `np.std()` , `np.var()`                                                         | Return the standard deviation or variance of a list or array                          |
| `np.cov()`                                                                      | Return the covariance between two lists or arrays of the same shape                   |
| `np.cross()`                                                                    | Returns the cross product of two 3D vectors                                           |

The functions `np.mean()`, `np.median()`, `np.std()`, `np.var()` have an optional `axis` argument that can specify along which axis of an ND array you calculate the statistic, for example:

```python
a = np.array([[1, 1, 1], [3, 0, 0]])
print(a.mean(axis=0))  # prints [2, 0.5, 0.5]
print(a.mean(axis=1))  # prints [1, 1]
```


### Linear algebra

NumPy has a sub-library for linear algebra, called `linalg`. Examples:

* You can calculate the inverse of a matrix using the `np.linalg.inv()` function. 
* Eigenvalues and eigenvectors can be calculated using `np.linalg.eig()`. 
* The matrix determinant can be found with `np.linalg.det()`.
* The matrix trace can be found with `np.linalg.trace()`.
* The norm of a vector be found with `np.linalg.norm()`. The order is 2 by default, but can be set with the `ord` parameter.


---

###### Exercises

Given the matrix `a = np.array([[1, 2, 0], [1, 3, 1], [2, 1, 2]])` and vector `y = array([2, 4, 1])`, calculate:

* The matrix eigenvalues and eigenvectors
    <details><summary>Answer</summary><p>

    `np.linalg.eig(a)`

    </p></details>

* The solution of a x = y
    <details><summary>Answer</summary><p>

    `np.linalg.inv(a).dot(y)`

    </p></details>

* The Frobenius norm of `a`
    <details><summary>Answer</summary><p>

    `np.linalg.norm(a)` or `np.linalg.norm(a, ord=2)`

    </p></details>

* Calculate the length (magnitude) of `y`
    <details><summary>Answer</summary><p>

    `np.linalg.norm(y)` or `np.linalg.norm(y, ord=2)`

    </p></details>

---


### Loading and saving data

The simplest way to save your data is to save them to a binary NumPy file with the `save()` function. For example, you can save an array as follows:

```python
a = np.array([[1, 2, 0], [1, 3, 1], [2, 1, 2]])
np.save('my_file.npy', a)
```

which saves the array `a` in binary format to the file `my_file.npy`. 

---

###### Exercises

1. Make two vector arrays with 1,000 elements. Convert the first to 8-bit integers, and the second to 64-bit integers. Save them both to different `*.npy` files. How much space do these files take on your hard drive? Can you explain why?
    
    <details><summary>Answer</summary><p>

    The 8-bit array should take about 8,000 bits, or 1000 bytes, so roughly 1 kB. the 64-bit array should take about 64,000 bits, or 8000 bytes, or 8 kB. Because of metainformation, the actual `*.npy` files will take more space on your hard drive, but they should roughly show the difference in size.

    </p></details>

---


### Random numbers

NumPy can generate arrays of random numbers with `np.random` sublibrary. Those arrays can draw from uniform and normal distributions. Here are some specific examples:

* `np.random.random_integers(low=5, high=10, size=(3, 2))` gives an array of shape 3x2 with random integers between 5 and 10. The integers are uniformly distributed.
* `np.random.uniform(low=4, high=9, size=(3, 2))` gives an array of shape 3x2 with uniformly distributed random numbers between 4 and 9.
* `np.random.normal(loc=1.5, scale=4.2, size=(3, 2))` gives an array of shape 3x2 with random integers drawn from a normal distribution with mean (loc) 1.5 and standard deviation (scale) 4.2.

Many more distributions are available. A full list can be found [here](https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.random.html#distributions).

Other useful functions in this sublibrary are `np.random.permutation()` which randomly shuffles the contents of an array or list, and `np.random.choice()`, which samples a random subset from a given array, for example

```python
a = np.arange(10)
np.random.choice(a, size=3)
```

generates a set of three random items in `a`.


## Matplotlib

Matplotlib is a library that lets you make plots and show images. The syntax of many of the functions provided by this library closely follows the syntax of the corresponding MATLAB functions. The Matplotlib documentation can be found [here](https://matplotlib.org/contents.html). Matplotlib is usually imported under the prefix `plt`, like so:

```python
import matplotlib.pyplot as plt
```

---

**Tip**
If you import multiple libraries in the same script, it is good practice to import them both *at the top of the script*. In general, it is bad practice to import libraries anywhere else than at the top of your script.
Therefore, if you import NumPy and Matplotlib, put the two import statements *both* at the top.

---

**Tip**
If you are using Matplotlib in a Jupyter notebook, putting the following code in *the first cell of your notebook*:

```python
%matplotlib inline
```

will work like magic: any figure you create will be displayed inline in your notebook. If you do this, you can also ignore all `plt.show()` commands in the remainder of this chapter, as your plots do not need to be shown in separate windows anymore.

---

Let's make a basic plot:

```python
x = np.linspace(-10, 10, 1000)
y = x ** 2
plt.plot(x, y)
```

This example makes an array of floating point values between -10 and 10, evaluates the quadratic function y = x^2, and then plots y against x. However, if you do this in a script, it will not show the plot yet. You need to type

```python
plt.show()
```

to show all the figures that you have plotted.


### Basic plotting

With the `plt.plot()` function, you can also plot multiple series, for example, if you have two mathematical functions you would like to plot, you can do this:

```python
x = np.linspace(-10, 10, 100)
y1 = x ** 2
y2 = x ** 3
y3 = x ** 4
plt.plot(x, y1, x, y2, x, y4)
```

By default, you will get a line plot, but you can also specify to plot dashed lines, or markers for the individual points using a format string for every line:

```python
x = np.linspace(-10, 10, 100)
y1 = x ** 2
y2 = x ** 3
y3 = x ** 4
plt.plot(x, y1, '-', x, y2, '.-', x, y3, '*')
```

You can also specify colors of the markers by preceding the marker with certain characters, for example `r--` gives you a red dashed line. A limited list of markers format strings can be found in the three tables below.

| Line formats  | Result                |
| ------------- | --------------------- |
| `-`           | solid line style      |
| `--`          | dashed line style     |
| `-.`          | dash-dot line style   |
| `:`           | dotted line style     |

| Point formats | Result                |
| ------------- | --------------------- |
| `.`           | point marker          |
| `,`           | pixel marker          |
| `o`           | circle marker         |
| `v`           | triangle_down marker  |
| `^`           | triangle_up marker    |
| `<`           | triangle_left marker  |
| `>`           | triangle_right marker |

| Color formats | Result                |
| ------------- | --------------------- |
| `b`           | blue                  |
| `g`           | green                 |
| `r`           | red                   |
| `c`           | cyan                  |
| `m`           | magenta               |
| `y`           | yellow                |
| `k`           | black                 |
| `w`           | white                 |

Alternatively, you can also have multiple calls to the `plt.plot()` function, like this:

```python
x = np.linspace(-10, 10, 100)
y1 = x ** 2
y2 = x ** 3
y3 = x ** 4
plt.plot(x, y1, '-')
plt.plot(x, y2, '.-')
plt.plot(x, y3, '*')
```

which gives the same result.


### Figures and axes

If you call the `plt.plot()` function multiple times, the lines and points will just be plotted in the same figure. If you want to make a new figure, you have to call the `plt.figure()` in between calls to the `plt.plot()` function. This will show each line in its own figure:

```python
x = np.linspace(-10, 10, 100)
y1 = x ** 2
y2 = x ** 3
y3 = x ** 4
plt.plot(x, y1, '-')
plt.figure()
plt.plot(x, y2, '.-')
plt.figure()
plt.plot(x, y3, '*')
plt.show()
```

If you want to show multiple plots side by side in one figure, you can make subplots. Often, you want to make a new figure, with a certain layout of subplots using the `plt.subplots()` function which returns a new figure and a list of axes. For example, to make a 2x3 array of subplots, you can use:

```python
fig, ax = plt.subplots(2, 3)
x = np.linspace(-10, 10, 100)
ax[0, 0].plot(x, x)
ax[0, 1].plot(x, x ** 2)
ax[0, 2].plot(x, x ** 3)
ax[1, 0].plot(x, x ** 4)
ax[1, 1].plot(x, x ** 5)
ax[1, 2].plot(x, x ** 6)
plt.show()
```

The `ax` variable is in fact a NumPy array filled with subplots that you can index like any array, hence `ax[0, 0]` is the subplot on the left of the top row, and `ax[1, 1]` is in the middle of the bottom row. 

If there is overlap between the subplots, you can call `plt.tight_layout()` before `plt.show()`. `plt.subplots()` also has an optional `figsize` parameter, that allows you to make a figure of a certain size, specified as a tuple with the width and height in inches: `fig, ax = plt.subplots(2, 3, figsize=(6, 4))`.


### Titles, legends, labels

Subplots can have labels on their x- and y-axis. Each subplot can also have a title, and a legend for the sequences that are plotted in it. The following example should give you an idea of what is possible:

```python
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
x = np.arange(-10, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

ax[0].plot(x, y1)
ax[0].set_title('Sine')
ax[0].set_xlabel('x')
ax[0].set_ylabel('sin(x)')

ax[1].plot(x, y2)
ax[1].set_title('Cosine')
ax[1].set_xlabel('x')
ax[1].set_ylabel('cos(x)')

plt.tight_layout()

plt.show()
```

### Other plots and diagrams

In addition to line/point plots using the `plt.plot()` function, Matplotlib supports boxplots, histograms, and plots with errorbars. Have a look at the following examples:

```python
data1 = np.random.normal(loc=0.0, scale=2.0, size=(1000,))
data2 = np.random.normal(loc=2.0, scale=1.5, size=(1000,))
plt.boxplot([data1, data2])
plt.show()
```

```python
plt.hist([data1, data2], bins=10)  # bins is the number of columns in the histogram
plt.show()
```

```python
x = np.arange(0, 11)
y = [0, 2, 1.5, 3, 4.5, 5, 5.3, 6, 8, 9.5, 9.9]
errors = np.random.rand(*x.shape)
plt.errorbar(x, y, yerr=[errors, errors])
plt.show()
```

Of course, you can also plot boxplots, histograms, and errorbar plots in subplots, by using the `boxplot()`, `hist()`, and `errorbar()` methods of an axis object.


### Showing images

You can show any 2D matrix as an image, using Matplotlib's `imshow()` function. For example, you can show an array like this:

```python
a = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
plt.imshow(a)
plt.show()
```

As you can see, Matplotlib has chosen some colors for you. If you don't like these, you can change them with the `cmap` parameter. For example, to show grayscale images, you can use:

```python
a = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
plt.imshow(a, cmap='Greys')
plt.show()
```

There are more `cmap` options available. A list can be found [here](https://matplotlib.org/users/colormaps.html).

Colormaps are nice, but they *do not* allow you to actually show RGB color images, like the ones you make with a digital camera. However, `imshow` automatically shows RGB channels if it receives a 3D array, where the last axis has length 3. This last axis is then treated as the intensity of the red, green, and blue channels. Such an array would look like this:

<a href="https://www.codecogs.com/eqnedit.php?latex=\left(\begin{matrix}&space;(r,&space;g,&space;b)&space;&&space;\cdots&space;&&space;(r,&space;g,&space;b)\\\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots\\\\&space;(r,&space;g,&space;b)&space;&&space;\cdots&space;&&space;(r,&space;g,&space;b)&space;\end{matrix}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left(\begin{matrix}&space;(r,&space;g,&space;b)&space;&&space;\cdots&space;&&space;(r,&space;g,&space;b)\\\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots\\\\&space;(r,&space;g,&space;b)&space;&&space;\cdots&space;&&space;(r,&space;g,&space;b)&space;\end{matrix}\right)" title="\left(\begin{matrix} (r, g, b) & \cdots & (r, g, b)\\\\ \vdots & \ddots & \vdots\\\\ (r, g, b) & \cdots & (r, g, b) \end{matrix}\right)" /></a>

However, it is important that the values in the matrix are either **floats** between 0 and 1, or **unsigned 8-bit integers** (i.e. np.uint8) between 0 and 255, otherwise what will be displayed will be a lot of nonsense.

```python
color_image = np.array([
    [[1, 0, 0], [1, 1, 0], [0, 0, 0]],        # red,   yellow,  black
    [[0, 1, 0], [0, 1, 1], [1, 1, 1]],        # green, cyan,    white
    [[0, 0, 1], [1, 0, 1], [0.5, 0.5, 0.5]]   # blue,  magenta, gray
], dtype='float32')
plt.imshow(color_image)
plt.show()
```

We will come back to this in the next section when we are going to give a short introduction to image analysis libraries.

---

###### Exercises

1. Make a figure with five subplots, each showing a histogram of a Gamma distribution, having shape parameters 1, 2, 3, 4, and 5, and each having scale 3.0.

    <details><summary>Answer</summary><p>

    ```python
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        sample = np.random.gamma(shape=i, scale=3, size=1000)
        ax[i].hist(sample,  bins=100)
    plt.show()
    ```

    </p></details>

2. Make a 3x3 image with a red cross on a blue background and show it with `plt.imshow()`.

    <details><summary>Answer</summary><p>

    ```python
    image = np.array([
        [[0, 0, 1], [1, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 0, 1], [1, 0, 0], [0, 0, 1]]
    ], dtype='float32')
    plt.imshow(image)
    ```

    </p></details>

---



## SciPy

SciPy is the final library we are going to discuss in this chapter, but this time we will let you read the documentation on this package yourself. You will need this for the exercises at the end, but you may also need it for a project. The documentation can be found [here](https://scipy.org/docs.html).

SciPy has a number of sub-libraries for statistics, optimization, interpolation, some more linear algebra, and image analysis. We are going to focus on the latter one, which is called `scipy.ndimage` and adds support for N-dimensional images: images of dimensions two and up. However, the other sub-libaries may also be relevant and useful in your own projects.

Let's look at an example in the SciPy documentation: the `scipy.ndimage.median_filter()` function. The median filter replaces each pixel in an image with the median of its neighborhood. If we look at the documentation, we find out that the *prototype* of this function is:

```python
scipy.ndimage.median_filter(input, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
```

The function has one mandatory argument named `input`. Below the prototype specification, we can find that this parameter is the input array, i.e. the input image that needs to be filtered with a median filter. Next, we find that there are optional argument for `size`, `footprint`, `output`,  `mode`, `cval`, and `origin`. Note that the documentation explicitly states to define either the `size` or the `footprint`. 


---

###### Exercises

For each of the following questions, you can use the following code to load an image and turn it into grayscale:

```python
image = plt.imread('/path/to/some/image')
image = np.mean(image[:, :, :3], axis=2)
```

or generate an image of a white disk on a black background using the code

```python
si, sj = 63, 63
image = np.zeros((si, sj))
for i in range(si):
    for j in range(sj):
        if (i - si // 2) ** 2 + (j - sj // 2) ** 2 < (si // 3) ** 2:
            image[i, j] = 1

image += np.ceil(np.random.rand(si, sj) - 0.9)
plt.imshow(image)
plt.show()
```

1. Use the median filter (with the size set to `(5, 5)`) on the grayscale version of the image that you loaded previously, and show the result next to the original image.

    <details><summary>Answer</summary><p>

    ```python
    after_median_filter = scipy.ndimage.median_filter(image, size=(5, 5))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(after_median_filter, cmap='gray')  # If you do not use the gray color map, Matplotlib will use its default color map.
    plt.show()
    ```

    </p></details> 

2. Use the median filter on the image, but this time test all multiples of 5 between 5 and 50 for `size`. Show each resulting image in its own subplot.

    <details><summary>Answer</summary><p>

    ```python
    sizes = range(5, 51, 5)

    fig, ax = plt.subplots(1, len(sizes))

    for i, size in enumerate(sizes):
        after_median_filter = scipy.ndimage.median_filter(image, size=(size, size))
        ax[i].imshow(after_median_filter, cmap='gray')   

    plt.show()
    ```

    </p></details> 

3. Using `SciPy`, show the image and a Gaussian filtered version of the grayscale image. Use a scale (sigma) of 5 pixels.

    <details><summary>Answer</summary><p>
    The documentation shows a function called `gaussian_filter()`, which has two obligatory parameters: `input` for the image, and `sigma` for the scale (the `sigma` in the Gaussian function). The result will show a blurred image. If you increase the sigma, the image will appear to be more blurred.

    ```python
    gaussian_filtered = scipy.ndimage.gaussian_filter(image, sigma=5)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(gaussian_filtered, cmap='gray')
    plt.show()
    ```

    </p></details> 

4. Compute the Gaussian gradient magnitude of the image at a scale of 5 pixels.

    <details><summary>Answer</summary><p>
    Very similar to the previous answer. In this case, the gradient of the image is computed, which will show the edges of the image in each direction, after which the magnitude (L2-norm) of the gradient vector is computed.

    ```python
    gradient_magnitude = scipy.ndimage.gaussian_gradient_magnitude(image, sigma=5)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(gradient_magnitude, cmap='gray')
    plt.show()
    ```

    </p></details> 

5. Rotate the image 45° anticlockwise.

    <details><summary>Answer</summary><p>
    Very similar to the previous answer. In this case, the gradient of the image is computed, which will show the edges of the image in each direction, after which the magnitude (L2-norm) of the gradient vector is computed.

    ```python
    rotated = scipy.ndimage.rotate(image, angle=45)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(rotated, cmap='gray')
    plt.show()
    ```

    </p></details> 

6. Threshold the image at 50% of the maximum intensity in the image.

    <details><summary>Answer</summary><p>
    This is a trick question, as you do not need SciPy to do this. You can use NumPy's ability to show where an array is larger than a threshold:

    ```python
    threshold = 0.5 * image.max()
    thresholded = image > threshold
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(thresholded, cmap='gray')
    plt.show()
    ```

    In the second line, the `>` is evaluated before assignment to `thresholded`, so it is executed as `thresholded = (grayscale > threshold)`. The NumPy data type of `thresholded` is `bool`.
    </p></details>


