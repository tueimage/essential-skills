#! /usr/bin/env python
#
# This module defines a scrollable viewer object for viewing volumetric data in Matplotlib.
# A minimal working example:
# 
# image = np.load('path/to/image')
# aspect_ratio = [2.3, 0.95, 0.95] # Is the elementspacing in *.mhd files
# 
# # Define viewers for every axis
# viewer1 = slycer.ScrollView(im)
# viewer2 = slycer.ScrollView(im.transpose(1, 0, 2))
# viewer3 = slycer.ScrollView(im.transpose(2, 0, 1))
# 
# # Make three Matplotlib supblots, and populate them with the viewers objects
# # The aspect ratios of the different axes need to be defined here as well.
# fig, ax = plt.subplots(1, 3)
# view1.plot(ax[0], cmap='gray', aspect=aspect_ratio[1]/aspect_ratio[2])
# view2.plot(ax[1], cmap='gray', aspect=aspect_ratio[0]/aspect_ratio[2])
# view3.plot(ax[2], cmap='gray', aspect=aspect_ratio[0]/aspect_ratio[1])
#
# Author: Koen Eppenhof


import numpy as np


class ScrollView:
    """Viewer for 3D images that can scroll through the i-axis.

    Attributes:
        speed (int): How many slices the ScrollView moves per scroll event.
        fast_speed (int): How many slices the ScrollView moves per scroll event while holding the shift key.
        very_fast_speed (int): How many slices the ScrollView moves per scroll event while holding the alt key.
        volume (np.ndarray): The wrapped 3D array.
        slice_index (int): The current slice index (in the i-index)
    """

    def __init__(self, volume, slice=0, speed=1, fast_speed=10, very_fast_speed=1e6):
        """
        Args:
            volume (np.array): A 3D numpy array.
            slice (int): The slice plotted by default.
            speed (int): How many slices the ScrollView moves per scroll event.
            fast_speed (int): How many slices the ScrollView moves per scroll event while holding the shift key.
            very_fast_speed (int): How many slices the ScrollView moves per scroll event while holding the alt key.

        Raises:
            ValueError - When volume is not a 3D Numpy array.
        """
        if type(volume) is not np.ndarray:
            raise ValueError('Volume is not a Numpy array')
        if volume.ndim != 3:
            raise ValueError('Volume is not a 3D Numpy array')

        self._volume = volume
        self._slice_index = slice

        # Set the default speeds
        self.speed = speed
        self.fast_speed = fast_speed
        self.very_fast_speed = very_fast_speed

        # At init there are no keys held
        self._shift_held = False
        self._alt_held = False

        self._ax = None

    @property
    def volume(self):
        """The volume plotted by the ScrollView object."""
        return self._volume

    @volume.setter
    def volume(self, volume):
        """The volume plotted by the ScrollView object."""

        if type(volume) is not np.ndarray:
            raise ValueError('Volume is not a Numpy array')
        if volume.ndim != 3:
            raise ValueError('Volume is not a 3D Numpy array')

        self._volume = volume
        self._ax.volume = self._volume

        if self._ax.slice_index > self._ax.volume.shape[0]:
            self._ax.slice_index = self._ax.volume.shape[0] - 1
        self.slice_index = self._ax.slice_index

        self._ax.images[0].set_array(self._ax.volume[self._ax.slice_index])
        self._ax.set_title('{}/{}'.format(
            self._ax.slice_index, self._ax.volume.shape[0] - 1))
        self.figure.canvas.draw()

    @property
    def slice_index(self):
        """The index of the slice that is currently shown in the plot."""
        return self._slice_index

    @slice_index.setter
    def slice_index(self, slice_index):
        """The index of the slice that is currently shown in the plot."""
        self._slice_index = slice_index
        if self._ax is not None:
            self._ax.slice_index = self._slice_index
            self._ax.images[0].set_array(self._ax.volume[self._slice_index])
            self._ax.set_title(
                '{}/{}'.format(self._ax.slice_index, self._ax.volume.shape[0] - 1))
        self.figure.canvas.draw()

    def plot(self, ax, *args, **kwargs):
        """
        Put the plot on the axis.slice_index

        Args:
            ax (AxesSubplot): The subplot that the images is shown on
            *kwargs (Iterable): plt.imshow options, like cmap, vmin, vmax, etc.
        """
        self.figure = ax.get_figure()

        # Set the volume
        ax.volume = self._volume

        # Set the slice index
        if self.slice_index >= self.volume.shape[0]:
            self.slice_index = self.volume.shape[0] - 1
        ax.slice_index = self.slice_index
        ax.imshow(ax.volume[self.slice_index], *args, **kwargs)

        # Update the title and axis
        self._ax = ax
        ax.set_title('{}/{}'.format(ax.slice_index, ax.volume.shape[0] - 1))

        # Attach the event methods
        self.figure.canvas.mpl_connect(
            'scroll_event', lambda x: self._process_scroll(x))
        self.figure.canvas.mpl_connect(
            'key_press_event', lambda x: self._process_key_press(x))
        self.figure.canvas.mpl_connect(
            'key_release_event', lambda x: self._process_key_release(x))

    def _process_key_press(self, event):
        """Private method for registering alt and shift keys."""
        if event.key == 'shift':
            self._shift_held = True
            self._alt_held = False
        if event.key == 'alt':
            self._alt_held = True
            self._shift_held = False

    def _process_key_release(self, event):
        """Private method for registering alt and shift keys."""
        if event.key == 'shift':
            self._shift_held = False
            self._alt_held = False
        if event.key == 'alt':
            self._alt_held = False
            self._shift_held = False

    def _process_scroll(self, event):
        """Private method for registering scrolling."""
        # Get the right speed
        if self._shift_held:
            speed = self.fast_speed
        elif self._alt_held:
            speed = self.very_fast_speed
        else:
            speed = self.speed

        # Send the command to the view that is being scrolled
        if event.inaxes == self._ax:
            if event.button == 'up':
                self._previous_slice(event.inaxes, speed)
            elif event.button == 'down':
                self._next_slice(event.inaxes, speed)

    def _previous_slice(self, ax, speed):
        new_slice_index = ax.slice_index - speed
        ax.slice_index = max(new_slice_index, 0)
        self.slice_index = ax.slice_index
        ax.images[0].set_array(ax.volume[ax.slice_index])
        ax.set_title('{}/{}'.format(ax.slice_index, ax.volume.shape[0] - 1))
        self.figure.canvas.draw()

    def _next_slice(self, ax, speed):
        new_slice_index = ax.slice_index + speed
        ax.slice_index = min(new_slice_index, ax.volume.shape[0] - 1)
        self.slice_index = ax.slice_index
        ax.images[0].set_array(ax.volume[ax.slice_index])
        ax.set_title('{}/{}'.format(ax.slice_index, ax.volume.shape[0] - 1))
        self.figure.canvas.draw()
