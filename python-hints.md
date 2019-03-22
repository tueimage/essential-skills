## Some generic Python tips

### Use comments, but do not overuse them

It is a good idea to put comments in your code when there are ambiguities. This is nice for other people who use your code, but it can also serve as a reminder when you have to use code that you wrote a few weeks ago. Do not overuse comments. It is useless to put comments every other line to explain things that are perfectly clear from the code alone. Only use them when the code is confusing. In general, if you design your code to be clear, you need fewer comments.

### Use functions for structuring your code and avoid repetition

It is good practice to write your code in functions exclusively. Try to divide the task you try to solve in logical subtasks that have a logical input and output. For example, if you write code to filter an image with a Gaussian kernel, it makes sense to define a function that creates the kernel and a function that convolves the image with the kernel.

In general, functions should do a single logical thing. Ideally they should be no more than ten statements (= ten lines) long to avoid complexity.

### Use the existing modules

Do not try to re-invent the wheel, it is already out there. It is nice to be able to implement complex algorithms yourself, but it is not the most productive way. If someone already did what you try to do and their code is available online, use their code! It is there fore a reason. In most cases, a good internet search containing the problem statement and the word 'python' will give you code bases to do what you want.

### Use the documentation of the packages you use

Most packages have online documentation, which is often very good. Use it!

### Google and ask

If you do not understand something even after reading the documentation, chances are someone asked the same question on a site like StackExchange or Reddit. Googling the problem you have (or even just the error message) should be the first step of solving a problem. If you cannot find the answer to your question, try asking it yourself on StackExchange. You will be surprised how many people want to help you if you ask a good question.

### Keep to the conventions

If you look at the code in this module, you may notice that we keep to a very strict coding convention called PEP8. PEP8 is a set of formatting rules that most Python developers follow. It is much easier to read code that uses the conventions you use yourself, and if you use PEP8 conventions, you will be able to read most of the code online much faster.

The conventions are summarized [here](https://www.python.org/dev/peps/pep-0008/). A quick summary:

* Use four spaces for indentation. Do not use tabs.
* Put a maximum of one statement on every line.
* Use lower case letters for function and variable names. Separate words by underscores, e.g. `my_awesome_function()`.
* Use CamelCase for class names, e.g. `MyAwesomeClass`.
* Use ALL CAPS for constant values, e.g. `THIS_STAYS_CONSTANT`.
* Separate functions and/or classes by two blank lines.
* Do not use more than one blank line otherwise.
* Put spaces around every operator, e.g. `1 + 1 == 2`, `x = 3`, not `1+1==2` and `x=3`.
* Do not put spaces around `=` when defining keyword arguments in functions, e.g. `translate_point(translation=[1, 2], point=[3, 4])`.
* Put all `import` statements at the top of a file. It is best not to import packages within functions or classes.

### Use doc strings

When your project becomes larger, it is useful to use doc strings for documentation. Doc strings are strings at the start of classes, methods, and functions that describe what they do. For complicated functions, it also worthwhile to mention which arguments are expected, what their types are, and what the function returns. Often the following format is used:

```python
def translate_point(translation, point):
    """
    Translates a point.

    Args:
        translation (list): The translation vector.
        point (list): The point.
    Returns:
        list: The translated point.
    """
    new_point = []
    for a, b in zip(translation, point):
        new_list.append(a + b)
    return new_point
```

### Separate large code bases into modules

When you have a lot of code it can become quite a hassle to maintain an overview of all your functions and classes. In that case, it is best to split-up your code over multiple `*.py` files. In Python, such a file is called a module. For example, when you are developing a deep learning method for liver segmentation in MR images, you can have a module for loading the image and segmentations and creating a training set out of them, a module in which the network is defined, a module that trains the network, and a module that provides functions for validation. You can import a module to get access to the functions and classes in that module. For example, if you have a file `network.py` that looks like this

```python
# network.py

def create_segmentation_network():
    cnn = ...
    # ... etc.
    return cnn
```

you can get to the `create_network()` function from another file in the same folder using

```python
# main.py

from . import network

my_network = network.create_segmentation_network()
```

You can read the `from . import network` statement as 'import the module network.py from the current folder'. Alternatively, you can also directly import a function from a module, like this:

```python
# main.py

from .network import create_segmentation_network

my_network = create_segmentation_network()
```

