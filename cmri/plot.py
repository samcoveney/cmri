"""Plotting functions"""

import sys
import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse
from matplotlib.collections import EllipseCollection
import matplotlib.lines as lines
from matplotlib.widgets import Button
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmri.utils as utils


def ellipsoid_shadows(evecs, evals):
    """Get 2d shadow of ellipsed onto xy plane"""

    # NOTE: attempt a 2D tensor plot with ellipses
    [Y, X] = np.meshgrid(range(evecs.shape[1]), range(evecs.shape[0]))
    XY = np.column_stack((X.ravel(), Y.ravel()))

    # equivalent to using 'norm' in fury plot
    Qn = evals / np.max(evals, axis = 3)[..., None] #[:, :, 0, 0:2])

    D = np.zeros([evecs.shape[0], evecs.shape[1], 3, 3])
    P = np.array([[1,0,0], [0,1,0], [0,0,0]])
    Q = np.zeros([evecs.shape[0], evecs.shape[1], 2])
    V = np.zeros([evecs.shape[0], evecs.shape[1], 2, 2])
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            tmp = evecs[i, j, 0] @ np.diag(Qn[i, j, 0]) @ evecs[i, j, 0].T
            tmp = P.T @ tmp @ P
            q, v = np.linalg.eigh(tmp[0:2, 0:2])  # note: ascending order
            Q[i, j] = q
            V[i, j] = v

    aa = np.degrees(np.arctan(V[..., -1, 1] / V[..., -1, 0]))
    ww, hh = Q[:, :, 1], Q[:, :, 0]

    return XY, ww, hh, aa


def tensor_plot_2d(evecs, evals, scalars, color, ax=None):
    """2d plot of tensors"""

    cmap, vmin, vmax = utils.get_colors(color)

    if ax is None:
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        make_plot = True
    else:
        make_plot = False

    XY, ww, hh, aa = ellipsoid_shadows(evecs, evals)
    ec = EllipseCollection(ww, hh, aa, units='xy', offsets=XY,  # NOTE: 'xy' may be bad as angle depends on aspect ratio
                           transOffset=ax.transData,
                           cmap=cmap, clim=[vmin, vmax], array=scalars.flatten())

    ax.add_collection(ec)
    ax.autoscale_view()
    ax.set_xlim(-0.5, evecs.shape[0]-0.5)
    ax.set_ylim(evecs.shape[1]-0.5, -0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #cbar = plt.colorbar(ec)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.1)
    cbar = plt.colorbar(ec, cax) #=self.ax)
    cbar.set_label('scalars')
    ax.set_aspect('equal')

    if make_plot:
        plt.show()
    else:
        return ax, cbar


class Multiplot(object):
    """Plotter for multiple images of the same size."""

    def __init__(self, scalars, evecs=None, evals=None):

        self.evecs, self.evals, self.scalars =\
        evecs, evals, scalars

        [self.Nx, self.Ny] = self.scalars[0]["values"].shape 
        x, y = np.mgrid[:self.Nx, :self.Ny]
        self.coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))

        # store the color info here
        for idx in range(len(self.scalars)):
            scalars = self.scalars[idx]["values"]
            name = self.scalars[idx]["name"]
            cmap, vmin, vmax = utils.get_colors(name)
            self.scalars[idx]["cmap"] = cmap
            if vmin is None: vmin = scalars.min()
            if vmax is None: vmax = scalars.max()
            self.scalars[idx]["vmin"] = vmin
            self.scalars[idx]["vmax"] = vmax
            self.scalars[idx]["vdif"] = vmax - vmin 

        # which color
        self.cidx = 0

        # setup plot
        plt.style.use('dark_background')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()

        #from matplotlib.gridspec import GridSpec
        #self.fig = plt.figure()
        #gs = GridSpec(1, 1) #, width_ratios=[1, 2], height_ratios=[4, 1])
        #self.ax = self.fig.add_subplot(gs[0])
        #plt.tight_layout()

        #self.fig.canvas.draw_idle()  # NOTE: perhaps pointless, draw happens in update

        #plt.show()
        #input("wait")
        
        if (self.evecs is not None) and (self.evals is not None): 
            self.ax, self.cbar = tensor_plot_2d(self.evecs, self.evals, self.scalars[self.cidx]["values"], self.scalars[self.cidx]["name"], self.ax)
        else:
            cmap, vmin, vmax = utils.get_colors(self.scalars[self.cidx]["name"])
            self.im = self.ax.imshow(self.scalars[self.cidx]["values"].T, cmap=cmap, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            self.cbar = plt.colorbar(self.im, cax) #=self.ax)
            self.cbar.set_label(self.scalars[self.cidx]["name"])
            self.ax.set_facecolor(color="white")

        self.cbar.set_label(self.scalars[self.cidx]["name"])
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)

        self.ax.set_xlim(-0.5, self.Nx-0.5)
        self.ax.set_ylim(self.Ny-0.5, -0.5)


    def key_press(self, event):
        """Changes scalars and color limits in the plot."""
        print('press:', event.key)

        sys.stdout.flush()
        if event.key == " ":
            self.cidx += 1
            if self.cidx == len(self.scalars):
                self.cidx = 0
            
            scalars = self.scalars[self.cidx]["values"].flatten()
            name = self.scalars[self.cidx]["name"]
            self.cbar.set_label(self.scalars[self.cidx]["name"])
            cmap = self.scalars[self.cidx]["cmap"]
            vmin = self.scalars[self.cidx]["vmin"]
            vmax = self.scalars[self.cidx]["vmax"]

            if (self.evecs is not None) and (self.evals is not None): 
                self.ax.collections[0].set_array(scalars)
                self.ax.collections[0].set_clim(vmin, vmax)
                self.ax.collections[0].set_cmap(cmap)
            else:
                self.im.set_data(scalars.reshape(self.Nx, self.Ny).T)
                self.im.set_clim(vmin, vmax)
                self.im.set_cmap(cmap)
            
        #if event.key in ["left", "right", "down", "up"]:
        if event.key in ["end", "pagedown", "down", "begin"]:
            vmin = self.scalars[self.cidx]["vmin"]
            vmax = self.scalars[self.cidx]["vmax"]
            vdif = self.scalars[self.cidx]["vdif"]
            factor = 0.05
            if event.key == "end":
                vmin = vmin - factor*vdif
            if event.key == "pagedown":
                vmin = vmin + factor*vdif
            if event.key == "down":
                vmax = vmax - factor*vdif
            if event.key == "begin":
                vmax = vmax + factor*vdif

            if vmax > vmin:
                self.scalars[self.cidx]["vmin"] = vmin
                self.scalars[self.cidx]["vmax"] = vmax

            if (self.evecs is not None) and (self.evals is not None): 
                self.ax.collections[0].set_clim(vmin, vmax)
            else:
                self.im.set_clim(vmin, vmax)

        self.fig.canvas.draw_idle()


    def run(self):
        """Just plotting"""

        plt.show()


class Select_outliers:
    """https://stackoverflow.com/questions/47439355/draggable-lines-select-one-another-in-matplotlib"""
    def __init__(self, data, data_f, MI, bvals=None):

        self.data = data
        self.data_f = data_f
        self.midx = np.argsort(MI)
        self.MIS = MI[self.midx]
        self.XorY = self.MIS.mean()

        # create grid of plots
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(3, 2)

        # plot for MI
        self.ax = fig.add_subplot(gs[0, :])
        if bvals is None:
            self.ax.plot(self.MIS, marker='o', c='black')
        else:
            self.ax.plot(self.MIS, c='black')
            self.ax.scatter(np.arange(data.shape[-1]), self.MIS, c=bvals)
        self.ax.set_ylabel("good ---> bad")
        self.c = self.ax.get_figure().canvas

        # plots for images
        self.axg = fig.add_subplot(gs[1, 0])
        self.axb = fig.add_subplot(gs[1, 1])
        self.axg_f = fig.add_subplot(gs[2, 0])
        self.axb_f = fig.add_subplot(gs[2, 1])
        self.axg.set_title("good obs")
        self.axg_f.set_title("good pred")
        self.axb.set_title("bad obs")
        self.axb_f.set_title("bad pred")
        for a in [self.axg, self.axg_f, self.axb, self.axb_f]:
            a.set_axis_off()
        self.update_plots()

        # line for threshold
        x = [0, data.shape[-1]]
        y = [self.XorY, self.XorY]
        self.line = lines.Line2D(x, y, picker=5)
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)

    def releaseonclick(self, event):
        self.line.set_ydata([event.ydata, event.ydata])
        self.XorY = self.line.get_ydata()[0]
        self.c.draw_idle()
        self.update_plots()

    def update_plots(self):
        # good plots (below threshold)
        im_good = self.data[:, :, self.midx[self.MIS < self.XorY]][..., -1]
        im_good_f = self.data_f[:, :, self.midx[self.MIS < self.XorY]][..., -1]
        vming = np.min([im_good.min(), im_good_f.min()])
        vmaxg = np.max([im_good.max(), im_good_f.max()])
        self.axg.imshow(im_good.T, cmap="turbo", vmin=vming, vmax=vmaxg)
        self.axg_f.imshow(im_good_f.T, cmap="turbo", vmin=vming, vmax=vmaxg)

        # bad plots (above threshold)
        try:
            im_bad = self.data[:, :, self.midx[self.MIS > self.XorY]][..., 0]
            im_bad_f = self.data_f[:, :, self.midx[self.MIS > self.XorY]][..., 0]
            vminb = np.min([im_bad.min(), im_bad_f.min()])
            vmaxb = np.max([im_bad.max(), im_bad_f.max()])
            self.axb.imshow(im_bad.T, cmap="turbo", vmin=vminb, vmax=vmaxb)
            self.axb_f.imshow(im_bad_f.T, cmap="turbo", vmin=vminb, vmax=vmaxb)
        except IndexError as e:
            self.axb.imshow(np.zeros_like(im_good.T), cmap="turbo")
            self.axb_f.imshow(np.zeros_like(im_good.T), cmap="turbo")

        self.c.draw_idle()

    def run(self):
        plt.show()

        good_img = np.ones(self.data.shape[-1], dtype=int)
        bad = self.midx[self.MIS > self.XorY]
        good_img[bad] = 0

        return good_img

