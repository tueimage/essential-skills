# Numerical and scientific computing in Python

So-far we have seen basic functionality of Python required to write your own
scientific programs. What has been lacking is a way to implement numeric code.

In this chapter we will look at four libraries that are often used for working
with Python in numeric and scientific computing: NumPy, SciPy, Matplotlib, and
Skimage. We start with the basics of NumPy, to look at some linear algebra and
ways to import  and export data, and then will move onto plotting that data
with Matplotlib.



## NumPy

NumPy is a library: a collection of classes, constants, and functions that you can import to
use in your own code. NumPy supports multi-dimensional arrays and matrices, linear
algebra operations, and has a large collection of functions for mathematical operations.

To start out, first import the NumPy library.

```python
import numpy as np
```

This statement imports the library. If you use a function, class, or constant from the library, you
need to prefixe it with `np.`, for example if you want to use the square root function `sqrt`
you type

```python
print(np.sqrt(9))  # prints 3.0
```

## N-dimensional arrays

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

$$\left(\begin{matrix}1 & 2 & 3\\\\4 & 5 & 6\\\\7 & 8 & 9\end{matrix}\right)$$

to the variable `matrix`. As you can see, the matrix is defined as a list of lists. 
For a three-dimensional array, you would need a list of lists of lists. For a simple
vector, you can just use a single list

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
gives you the size along each axis of the array

```python
print(matrix.shape)  # prints (3, 3) because matrix is a 3x3 array
print(vector.shape)  # prints (3, ) because vector is a 3x1 array
```

The order of the shapes is similar to the ordering of matrix axes: first the
length in *i*-direction (downwards) is given, then the *j*-direction (left-to-right).
You can also use the `len` function to get the length of the first axis.

```python
m = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(len(m))  # prints 4
```

## Indexing arrays

Indexing an array is very similar to indexing lists:

```python
print(vector[0])  # prints the first element (i.e. 1)
print(vector[1:])  # prints elements starting from index 1 (i.e. array([2, 3]))
print(matrix[0])  # prints the first element (i.e. array([1, 2, 3]))
print(matrix[1:])  # prints elements starting from index 1 (i.e. array([[4, 5, 6], [7, 8, 9]]))
```

In the `matrix[0]` example, you obtain a new array, that you can index again, e.g.

```python
print(matrix[0][1])  # prints the second element of the first element (array([2, 3])[1], i.e. 3) 
```

You can write this more cleanly as

```python
print(matrix[0, 1])
```

Of course you can also slice an array in this way. If you want to get all elements
along a certain axis, you can use `:` as an index, for example

```python
print(matrix[:, 1])
```

will print all rows (indicated by the `:`) and the second column (indicated by the `1`).


#### Exercises

Given this code

```python
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
```
How do you obtain

* the first row of the matrix?
    * `m[0]`

* the second row of the matrix?
    * `m[1]`

* every second row of the matrix?
    * `m[::2]`

* every second column of the matrix?
    * `m[:, ::2]`

* the last column of the matrix?
    * `m[:, -1]`


## Assigning at indices

Like with lists, you can assign directly to parts of an array, for example

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
with the indices on the left: if you are assigning to a 2x2 block, make sure
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

If the righthandside has shape (2, 1), for example

```python
matrix[:2, :2] = np.array([[10], [20]])
```

the result will be

```python    
[
    [10, 10, 2],
    [20, 20, 6],
    [ 7,  8, 9]
]
```

If you only put one value on the righthandside, NumPy will just put that value
in every element that you are assigning to:

```python
matrix[:2, :2] = 42
```

gives the result

```python
[
    [42, 42, 2],
    [42, 42, 6],
    [ 7,  8, 9]
]
```
Be careful: like lists, arrays are mutable variables, which means they assign
*by reference*, i.e.

```python
a = np.array([1, 2, 3])
b = a
a[0] = 0  # also changes b!
print(b)  # prints [0, 2, 3]
```

#### Exercises

Given the matrix 
```python
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```
write code that

* Replaces the rightmost column with zeroes
* `m[:, -1] = 0`

* Replaces every second row with ones
* `m[::2] = 1`

* Puts a `2` in every element on the diagonal
* Code:
```python
for i in range(len(m)):
    m[i, i] = 2
```

## Reshaping and transposing

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
that you supply to the method.

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




#### Exercises

* Reshape the array `m = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])` to the shape `(3, 3)`

    * This results in a `ValueError` because the shape does not fit the array.

* What happens when you transpose a vector like np.array([1, 2, 3, 4])? Why?

    * `transpose()` reverses the order of the axes. For a 1D array reversing the order has no effect.

* Given the matrix `np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`, write code that rotates the elements 90 degrees anticlockwise, such that you obtain a matrix

    ```python
    [
        [3, 6, 9],
        [2, 5, 8],
        [1, 4, 7]
    ]
    ```

    * Code:
    ```python
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rotated_a = a.transpose()[::-1]
    ```



## Scalar and dot products

## Linear algebra

## Statistics

## 

