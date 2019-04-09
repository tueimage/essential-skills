from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from unittest import TestCase
from scrollview import ScrollView


class TestScrollView(TestCase):

    @staticmethod
    def phantom(shape=(10, 20, 30)):
        im = np.zeros(shape)
        for i in range(len(im)):
            im[i, 0, :i] = 1
        return im

    def test_plot_in_single_axis(self):
        fig, ax = plt.subplots()
        ScrollView(self.phantom()).plot(ax)

    def test_plot_in_multi_axis(self):
        fig, ax = plt.subplots(1, 2)
        ScrollView(self.phantom()).plot(ax[0])
        ScrollView(self.phantom()).plot(ax[1])

    def test_exceptions(self):
        self.assertRaises(ValueError, ScrollView, 0)
        self.assertRaises(ValueError, ScrollView, np.random.rand(2, 2))
        s = ScrollView(self.phantom())
        with self.assertRaises(ValueError):
            s.volume = 0
        with self.assertRaises(ValueError):
            s.volume = np.random.rand(2, 2)

    def test_setting_volume(self):
        fig, ax = plt.subplots()
        s = ScrollView(self.phantom())
        s.plot(ax)
        s.volume = self.phantom((20, 30, 10))

    def test_setting_volume_with_exceeding_slice_index(self):
        fig, ax = plt.subplots()
        t = ScrollView(self.phantom((20, 30, 10)))
        t.plot(ax)
        t.slice_index = 19
        t.volume = self.phantom((10, 10, 10))
        self.assertEquals(t.slice_index, 9)

    def test_plotting_volume_with_exceeding_slice_index(self):
        fig, ax = plt.subplots()
        t = ScrollView(self.phantom((20, 2, 2)), slice=50)
        t.plot(ax)
        self.assertEquals(t.slice_index, 19)

    def test_key_presses(self):
        fig, ax = plt.subplots()
        t = ScrollView(self.phantom((20, 2, 2)))
        t.plot(ax)

        event = matplotlib.backend_bases.Event(name='test', canvas=None)
        event.key = 'shift'
        t._process_key_press(event)
        t._process_key_release(event)
        event.key = 'alt'
        t._process_key_press(event)
        t._process_key_release(event)

    def test_scrolling(self):
        fig, ax = plt.subplots()
        t = ScrollView(self.phantom((20, 2, 2)))
        t.plot(ax)

        event = matplotlib.backend_bases.Event(name='test', canvas=None)
        event.inaxes = ax

        event.button = 'down'
        t._process_scroll(event)
        self.assertEqual(t.slice_index, 1)

        event.button = 'up'
        t._process_scroll(event)
        self.assertEqual(t.slice_index, 0)

    def test_scrolling_with_modifier(self):
        fig, ax = plt.subplots()
        t = ScrollView(self.phantom((20, 2, 2)))
        t.plot(ax)

        event = matplotlib.backend_bases.Event(name='test', canvas=None)
        event.inaxes = ax
        event.key = 'shift'

        t._process_key_press(event)
        event.button = 'down'
        t._process_scroll(event)
        self.assertEqual(t.slice_index, 10)
        t._process_key_release(event)

        event.key = 'alt'
        t._process_key_press(event)
        event.button = 'up'
        t._process_scroll(event)
        self.assertEqual(t.slice_index, 0)
        t._process_key_release(event)
