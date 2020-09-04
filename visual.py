import numpy as np
import matplotlib.pyplot as plt
import torch

from monai.data import DataLoader


class IndexTracker(object):
    """
    Image Slices Viewer
    Scroll through 2D image slices of a 3D array.
    https://matplotlib.org/3.1.0/gallery/event_handling/image_slices_viewer.html
    """
    def __init__(self, ax, X):
        self.ax = ax

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def show3(volume, label='use scroll wheel to navigate images', ax=None, do_show=True, seg=None):
    ax = ax or plt.gca()
    fig = plt.gcf()
    if seg is not None:
        volume = np.append(volume, seg * np.max(volume), axis=1)
    tracker = IndexTracker(ax, volume)
    ax.set_title(label)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    if do_show:
        plt.show()
    return fig


def show_element(dataset, number_of_examples=1):
    check_loader = DataLoader(dataset)
    for i, batch_data in enumerate(check_loader):
        image, label = (batch_data["image"][0][0], batch_data["label"][0][0])
        show3(np.append(image, label * torch.max(image), axis=1))
        if i+1 == number_of_examples:
            break
