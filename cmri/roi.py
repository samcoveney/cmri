"""Region Of Interest."""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse
import matplotlib.lines as lines


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


class draggable_lines:
    """https://stackoverflow.com/questions/47439355/draggable-lines-select-one-another-in-matplotlib"""
    def __init__(self, ax, axg, axb, kind, data, MI):

        self.data = data
        self.midx = np.argsort(MI)
        self.MIS = MI[self.midx]

        self.ax = ax
        self.ax.plot(self.MIS, marker='o')
        self.c = ax.get_figure().canvas

        self.axg = axg
        self.axb = axb

        self.o = kind
        self.XorY = self.MIS.mean()

        if kind == "h":
            x = [0, data.shape[-1]]
            y = [self.XorY, self.XorY]

        elif kind == "v":
            x = [self.XorY, self.XorY]
            y = [0, data.shape[-1]]
        
        # images bracketing threshold
        im_good = self.data[:, :, self.midx[self.MIS < self.XorY]][..., -1]
        im_bad = self.data[:, :, self.midx[self.MIS > self.XorY]][..., 0]
        axg.imshow(im_good.T, cmap="turbo")
        axb.imshow(im_bad.T, cmap="turbo")

        self.line = lines.Line2D(x, y, picker=5)
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)

    def releaseonclick(self, event):
        if self.o == "h":
            self.line.set_ydata([event.ydata, event.ydata])
        else:
            self.line.set_xdata([event.xdata, event.xdata])
        self.c.draw_idle()

        if self.o == "h":
            self.XorY = self.line.get_ydata()[0]
        else:
            self.XorY = self.line.get_xdata()[0]
        self.c.draw_idle()

        # update plots
        im_good = self.data[:, :, self.midx[self.MIS < self.XorY]][..., -1]
        im_bad = self.data[:, :, self.midx[self.MIS > self.XorY]][..., 0]
        self.axg.imshow(im_good.T, cmap="turbo")
        self.axb.imshow(im_bad.T, cmap="turbo")
        self.c.draw_idle()


def select_outliers(data, MI):
    """Select outliers based on thresh-hold."""

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    axg = fig.add_subplot(gs[1, 0])
    axb = fig.add_subplot(gs[1, 1])
    Vline = draggable_lines(ax, axg, axb, "h", data, MI)
    plt.show()

    good_img = np.ones(data.shape[-1])
    bad = Vline.midx[Vline.MIS > Vline.XorY]
    good_img[bad] = 0

    return good_img

