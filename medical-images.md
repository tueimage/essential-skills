# Working with medical images in Python

In this chapter, a few modules and methods for working with medical images are
discussed. These packages are not included with the Anaconda installation you
(may) have installed previously. Therefore this chapter will also discuss how to
install packages in Python.


## Installing extra Python packages

Anaconda comes with the `pip` package manager. You can run this from a Terminal window (on Linux or macOS) or a Command Prompt window on Windows, provided you have added the Anaconda distribution to your PATH during installation. Alternatively, you can open a prompt or terminal from the Anaconda Navigator by clicking on `Environments` in the left side bar, clicking on the green triangle, and then 'Open Terminal'.

![Opening a Terminal window from the Anaconda Navigator](figures/anaconda_install_pip.png)

In the Terminal, the Windows Prompt, or the Anaconda Terminal, you can use `pip` to install packages. For example, to install the package `SimpleITK`, use

```bash
pip install --user SimpleITK
```

We are going to use SimpleITK to load `*.mhd` files later.

To be able to DICOM files, also install the package `pydicom` this way.

To view image files in Matplotlib, you can install our group's custom image viewer package using

```bash
pip install git+https://github.com/tueimage/slycer
```

With these three packages you are all set for the remainder of this chapter.


## Working with `*.mhd` files

`*.mhd` files are used in Elastix and the ITK software packages. These files can 
be opened using the SimpleITK package, which is a rather schizophrenic translation of
ITK to Python. The functions in this package do *not* adhere to Python conventions. For example, all functions have capitalized camel case names (i.e. `ReadImage` instead of `read_image`). 

```python
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

itk_image = sitk.ReadImage('example_data/chest_xray.mhd')
image_array = sitk.GetArrayFromImage(itk_image)

# print the image's dimensions
print(image_array.shape)

# plot the image
plt.imshow(image_array, cmap='gray')
plt.show()
```

This code loads the image and then retrieves the image array as a NumPy array.

`*.mhd` files themselves are pure text header files that contain properties of the images.
For example, for the example `chest_xray.mhd` file, the parameters read

```
ObjectType = Image
NDims = 2
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 1
Offset = 0 0
CenterOfRotation = 0 0
ElementSpacing = 1 1
DimSize = 1024 1024
ElementType = MET_DOUBLE
ElementDataFile = chest_xray.raw
```

This shows that this chest X-ray is a 2D image, consisting of uncompressed binary data, with 1 mm x 1 mm pixels (`ElementSpacing`) and a size of 1024 by 1024. The `MET_DOUBLE` type will be converted to the `numpy.float64` dtype. SimpleITK image objects like `itk_image` have some methods to get and set these parameters. Because the names of the methods and parameters in the header file *do not match at all*, we give a summary of the most important ones below. Each of the `Get*` methods has a similar `Set*` method to change the parameter, e.g. `itk_image.SetOrigin([1, 0])`.

| Method name      | Accessed parameter |
| ---------------- | ------------------ |
| `GetDimension()` | `NDims`            |
| `GetSize()`      | `DimSize`          |
| `GetOrigin()`    | `CenterOfRotation` |
| `GetSpacing()`   | `ElementSpacing`   |
| `GetDirection()` | `TransformMatrix`  |

### Writing `*.mhd` files

You can convert any NumPy array to an ITK image using the `GetImageFromArray()` function. You can write the image to disk using `sitk.WriteImage()`. Before you write the image, you can use the setter methods to change the `ElementSpacing` parameter.

```python
random_data = np.random.rand(100, 100)
random_itk_image = sitk.GetImageFromArray(random_data)
random_itk_image.SetSpacing([1.1, 0.98]) # Each pixel is 1.1 x 0.98 mm^2
sitk.WriteImage(random_itk_image, '/destination/path/for/image.mhd')
```

## Working with Dicom files

## Plotting 3D image files




