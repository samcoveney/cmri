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


class select_point:
    """Select a coordinate by clicking on an image."""

    def __init__(self, img, vmin=None, vmax=None):

        self.img = img
        self.vmin = self.img.min() if vmin is None else vmin
        self.vmax = self.img.max() if vmax is None else vmax
        self.x = None
        self.y = None


    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.imshow(self.img.T, cmap="turbo", vmin=self.vmin, vmax=self.vmax)
        ax.set_title("Click on LV center, close when done")

        self.point = None

        def onclick(event):
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  (event.button, event.x, event.y, event.xdata, event.ydata))
            if self.point is not None:
                self.point[0].remove()
            self.point = plt.plot(event.xdata, event.ydata, 'o', color = 'red')
            self.x, self.y = event.xdata, event.ydata
            fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return self.x, self.y

