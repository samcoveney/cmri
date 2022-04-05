"""Region Of Interest."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse


class create_mask:
    """Define a mask.

       https://stackoverflow.com/questions/68229211/matplotlib-ellipseselector-how-to-get-the-path
    """

    def __init__(self, img):

        # get pixel coordinates
        self.img_shape = img.shape
        X, Y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        self.MYDATA = np.vstack([X.ravel(), Y.ravel()])

        # create figure and selector
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(img.T, cmap = "gray")
        self.es = EllipseSelector(self.ax, self.onselect,
                                  interactive=True)
        plt.title("Click & drag, adjust, close plot.")
        plt.show()

    # selector callback method
    def onselect(self, eclick, erelease):
        ext = self.es.extents
        ellipse = Ellipse(self.es.center, ext[1]-ext[0], ext[3]-ext[2])
        mask = ellipse.contains_points(self.MYDATA.T)
        self.mask = mask.reshape(self.img_shape).astype(np.int32)

