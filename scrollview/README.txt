# Scrolling views for 3D data in Matplotlib

This module defines a scrollable viewer object for viewing volumetric data in Matplotlib.
A minimal working example:

```python
image = np.load('path/to/image')
aspect_ratio = [2.3, 0.95, 0.95]

# Define viewers for every axis
viewer1 = slycer.ScrollView(im)
viewer2 = slycer.ScrollView(im.transpose(1, 0, 2))
viewer3 = slycer.ScrollView(im.transpose(2, 0, 1))

# Make three Matplotlib supblots, and populate them with the viewers objects
# The aspect ratios of the different axes need to be defined here as well.
fig, ax = plt.subplots(1, 3)
view1.plot(ax[0], cmap='gray', aspect=aspect_ratio[1]/aspect_ratio[2])
view2.plot(ax[1], cmap='gray', aspect=aspect_ratio[0]/aspect_ratio[2])
view3.plot(ax[2], cmap='gray', aspect=aspect_ratio[0]/aspect_ratio[1])
```

![An example of three `ScrollView` objects](example.png)

The code in this module was inspired by work by [Juan Nunez-Iglesias]([https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data)

