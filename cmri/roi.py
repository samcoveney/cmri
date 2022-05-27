"""Region Of Interest."""

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

from scipy import interpolate

from cmri import utils


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


def square_from_mask(mask, return_indices=False):
    """Get the indices for square crop from a general mask."""

    cond = (mask > 0)
    ymin = np.argmin(np.sum(mask, axis = 0) > 0)

    y_tmp = np.argwhere(np.sum(mask, axis = 0) > 0).flatten()
    x_tmp = np.argwhere(np.sum(mask, axis = 1) > 0).flatten()

    if return_indices == False:
        square_mask = np.zeros_like(mask, dtype=bool)
        square_mask[x_tmp[0]:x_tmp[-1]+1, y_tmp[0]:y_tmp[-1]+1] = True
        return square_mask
    else:
        return x_tmp[0], x_tmp[-1] + 1, y_tmp[0], y_tmp[-1] + 1


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

        self.fig.canvas.draw_idle()  # NOTE: perhaps pointless, draw happens in update
        
        if (self.evecs is not None) and (self.evals is not None): 
            self.ax, self.cbar = tensor_plot_2d(self.evecs, self.evals, self.scalars[self.cidx]["values"], self.scalars[self.cidx]["name"], self.ax)
        else:
            cmap, vmin, vmax = utils.get_colors(self.scalars[self.cidx]["name"])
            self.im = self.ax.imshow(self.scalars[self.cidx]["values"].T, cmap=cmap, vmin=vmin, vmax=vmax)
            self.cbar = plt.colorbar(self.im, ax=self.ax)
            self.cbar.set_label(self.scalars[self.cidx]["name"])

        self.cbar.set_label(self.scalars[self.cidx]["name"])
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)


    def key_press(self, event):
        """Changes scalars and color limits in the plot."""
        print('press:', event.key)

        sys.stdout.flush()
        if event.key == " ":
            print("pressed space")
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
       
            self.fig.canvas.draw_idle()
            
        if event.key in ["left", "right", "down", "up"]:
            vmin = self.scalars[self.cidx]["vmin"]
            vmax = self.scalars[self.cidx]["vmax"]
            vdif = self.scalars[self.cidx]["vdif"]
            factor = 0.05
            if event.key == "left":
                vmin = vmin - factor*vdif
            if event.key == "right":
                vmin = vmin + factor*vdif
            if event.key == "down":
                vmax = vmax - factor*vdif
            if event.key == "up":
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
    

class Select_point(Multiplot):
    """Select a coordinate by clicking on an image."""

    def __init__(self, scalars, evecs=None, evals=None):
        super().__init__(scalars, evecs, evals)

    def run(self):
        """Things specific to setting up and running this class."""

        # add callback
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)

        # coords of selection
        self.x, self.y = None, None
        self.point = None

        self.ax.set_title("Click on LV center, close when done")
        plt.show()

        return self.x, self.y

    def button_press_callback(self, event):

        if self.point is None:
            self.point, = self.ax.plot(event.xdata, event.ydata, 'o', color = 'white')
        else:
            self.point.set_xdata(event.xdata)
            self.point.set_ydata(event.ydata)

        self.x, self.y = event.xdata, event.ydata
        self.fig.canvas.draw()


class select_outliers:
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
        im_good = self.data[:, :, self.midx[self.MIS < self.XorY]][..., -1]
        im_good_f = self.data_f[:, :, self.midx[self.MIS < self.XorY]][..., -1]
        vming = np.min([im_good.min(), im_good_f.min()])
        vmaxg = np.max([im_good.max(), im_good_f.max()])
        im_bad = self.data[:, :, self.midx[self.MIS > self.XorY]][..., 0]
        im_bad_f = self.data_f[:, :, self.midx[self.MIS > self.XorY]][..., 0]
        vminb = np.min([im_bad.min(), im_bad_f.min()])
        vmaxb = np.max([im_bad.max(), im_bad_f.max()])
        self.axg.imshow(im_good.T, cmap="turbo", vmin=vming, vmax=vmaxg)
        self.axg_f.imshow(im_good_f.T, cmap="turbo", vmin=vming, vmax=vmaxg)
        self.axb.imshow(im_bad.T, cmap="turbo", vmin=vminb, vmax=vmaxb)
        self.axb_f.imshow(im_bad_f.T, cmap="turbo", vmin=vminb, vmax=vmaxb)

        self.c.draw_idle()

    def run(self):
        plt.show()

        good_img = np.ones(self.data.shape[-1])
        bad = self.midx[self.MIS > self.XorY]
        good_img[bad] = 0

        return good_img


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
    ax.set_ylim(-0.5, evecs.shape[1]-0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cbar = plt.colorbar(ec)
    cbar.set_label('scalars')
    ax.set_aspect('equal')

    if make_plot:
        plt.show()
    else:
        return ax, cbar


class Segment_surfs(Multiplot):
    """Segment image with two surfaces."""

    def __init__(self, scalars, evecs=None, evals=None):
        super().__init__(scalars, evecs, evals)

    def run(self):
        """Things specific to setting up and running this class."""

        # add callbacks
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        self.points, self.curve = [None, None], [None, None]
        self.pind = None # point index
        self.srf = 0 # index of surface
        self.epsilon = 10 # tolerance for selecting points
        self.ls = "-w" # curve style

        # for storing surface coordinates
        self.x, self.y = [None, None], [None, None]
        for srf in [0, 1]:
            self.x[srf] = np.array([], dtype=float)
            self.y[srf] = np.array([], dtype=float)

        # NOTE: will need two masks
        self.mask = [None, None]
        self.mask[0] = np.ones((self.Nx, self.Ny), dtype=bool)
        self.mask[1] = np.ones((self.Nx, self.Ny), dtype=bool)
        self.switch_surface()
        self.update()

        plt.show()

        # return the mask
        return self.mask[0] * ~self.mask[1]

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        print("event.button:", event.button)
        if event.inaxes is None:
            return

        if event.button == 3:  # right-click = place points

            if self.x[self.srf].size > 0:
                self.x[self.srf] = np.hstack([self.x[self.srf], event.xdata])
                self.y[self.srf] = np.hstack([self.y[self.srf], event.ydata])
            else:
                self.x[self.srf] = np.array([event.xdata])
                self.y[self.srf] = np.array([event.ydata])

            self.reorder()
            self.update()
            return

        if event.button == 1:  # left-click = adjust points
            self.pind = self.get_ind_under_point(event)    

        if event.button == 2:  # switch surface
            self.switch_surface()

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if event.button in [1, 3]:
            self.reorder()
            self.update()
            self.pind = None
        else:
            return

    def motion_notify_callback(self, event):
        'on mouse movement'
        if self.pind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        
        self.x[self.srf][self.pind] = event.xdata
        self.y[self.srf][self.pind] = event.ydata

        self.update()

    def update(self):

        # update existing points 
        if self.points[self.srf] is not None:
            self.points[self.srf].set_xdata(self.x[self.srf])
            self.points[self.srf].set_ydata(self.y[self.srf])
        
        # plot points for first time
        if self.points[self.srf] is None and self.x[self.srf].size > 0:
            self.points[self.srf], = self.ax.plot(self.x[self.srf], self.y[self.srf], 'ow')
    
        # update curve
        k = 3
        if self.x[self.srf].size >= k:
            X = np.r_[self.x[self.srf], self.x[self.srf][0]]
            Y = np.r_[self.y[self.srf], self.y[self.srf][0]]
            tck, u = interpolate.splprep([X, Y], s=0, per=True, k=k)
            xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

            # snap curves to nearest pixels
            if False:
                curve_coors = np.hstack((xi.reshape(-1, 1), yi.reshape(-1,1)))
                tmp = self.coors[np.argmin(cdist(curve_coors, self.coors), axis=-1)]
                xi, yi = tmp[:, 0], tmp[:, 1]

            # update existing curve
            if self.curve[self.srf] is not None:
                self.curve[self.srf].set_xdata(xi)
                self.curve[self.srf].set_ydata(yi)
            # plot curve for first time
            else:
                self.curve[self.srf], = self.ax.plot(xi, yi, self.ls)

        # create mask
        if self.curve[self.srf] is not None:
            tmp = np.vstack([xi, yi]).T
            poly_path=Path(tmp)

            self.mask[self.srf][:, :] = poly_path.contains_points(self.coors).reshape(self.Nx, self.Ny)

            mask_total = self.mask[0] * ~self.mask[1]
            alpha = (mask_total + 0.3) / (1.3) 

            if (self.evecs is not None) and (self.evals is not None): 
                self.ax.collections[0].set_alpha(alpha)
            else:
                self.im.set_alpha(alpha.T)

        self.ax.set_xlim(-0.5, self.Nx-0.5)
        self.ax.set_ylim(self.Ny-0.5, -0.5)

        # redraw canvas while idle
        self.fig.canvas.draw_idle()

    def reorder(self):
        """ensure points are ordered to create a non-crossing loop"""
        # re-arrange the points to make a circle
        com = np.array([self.x[self.srf].mean(), self.y[self.srf].mean()])
        x = self.x[self.srf] - com[0]
        y = self.y[self.srf] - com[1]

        angles = np.abs(np.arctan(y / x))
        angles[(y < 0) & (x < 0)] = np.pi + angles[(y < 0) & (x < 0)] # Q3
        angles[(y > 0) & (x < 0)] = np.pi - angles[(y > 0) & (x < 0)] # Q2
        angles[(y < 0) & (x > 0)] = 2*np.pi - angles[(y < 0) & (x > 0)] # Q4

        args = np.argsort(angles)
        self.x[self.srf] = self.x[self.srf][args]
        self.y[self.srf] = self.y[self.srf][args]

    def get_ind_under_point(self, event):
        """get the index of the vertex under point if within epsilon tolerance"""
        # NOTE: this seems very clunky...

        if self.x[self.srf].size > 0:
            # display coords
            #print('display x is: {0}; display y is: {1}'.format(event.x,event.y))
            t = self.ax.transData.inverted()
            tinv = self.ax.transData 
            xy = t.transform([event.x, event.y])
            xr = np.reshape(self.x[self.srf], (np.shape(self.x[self.srf])[0],1))
            yr = np.reshape(self.y[self.srf], (np.shape(self.y[self.srf])[0],1))
            xy_vals = np.append(xr, yr, 1)
            xyt = tinv.transform(xy_vals)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.hypot(xt - event.x, yt - event.y)
            indseq, = np.nonzero(d == d.min())
            ind = indseq[0]

            if d[ind] >= self.epsilon:
                ind = None
        else:
            ind = None
        
        print(ind)
        return ind

    def switch_surface(self):
        """Cycle through modes"""

        #print("self.srf:", self.srf)

        self.srf = (self.srf + 1) % 2
        if self.srf == 0:
            self.ax.set_title("epi")
            self.ls = "-w"
        if self.srf == 1:
            self.ax.set_title("endo")
            self.ls = "--w"
        self.fig.canvas.draw_idle()


class Segment_aha(Multiplot):
    """Place AHA template over image."""

    def __init__(self, scalars, evecs=None, evals=None):
        super().__init__(scalars, evecs, evals)

    def run(self, segs):
        """Things specific to setting up and running this class."""

        # add callbacks
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)

        self.segs = segs  # number of segments
        self.srf = 0  #  0: LVcent, 1: insertion point
        # NOTE: probably would be better as array(s), one for LVcent, one for insertion
        self.x, self.y = [None, None], [None, None]
        for srf in [0, 1]:
            self.x[srf] = np.array([], dtype=float)
            self.y[srf] = np.array([], dtype=float)

        self.epsilon = 10
        # NOTE: will need two of these, one per surface, store as list
        self.points, self.curve = [None, None], [None, None]
        self.circle1 = None
        self.lines = [None for i in range(6)]
        self.ls = "-w" # curve style

        self.update()

        plt.title("left-click: LV center, right-click: insertion point")
        plt.show()

        # calculate the segment labels 
        vec = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]]) # from LVcent to insertion point
        vec_other = self.coors - np.array([self.x[0], self.y[0]]).T
        vec /= np.linalg.norm(vec)
        vec_other /= np.linalg.norm(vec_other, axis=1)[:, None]

        # NOTE: array dim 0 is on up-down axis
        
        # rotate coordinates, to align x-axis with vec
        ang = np.angle(vec[1] + vec[0]*1j)[0]
        rot = np.array([[np.cos(ang), -np.sin(ang)], [+np.sin(ang), np.cos(ang)]])
        vec_other = (rot @ vec_other.T).T

        # calculate angles using complex numbers
        angles = np.angle(vec_other[:, 1] + vec_other[:, 0] * 1j)
        angles = np.rad2deg(angles)
        angles[angles < 0] = 360 + angles[angles < 0] # adjust range to 0..360

        # NOTE: return the segment labels 
        labels = np.ceil(angles / (360 / self.segs))

        return labels.reshape(self.Nx, self.Ny).astype(int)

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        print("event.button:", event.button)
        if event.inaxes is None:
            return

        if event.button in [1,3]:  # any-click = place points

            if event.button == 1:
                self.srf = 0
            if event.button == 3:
                self.srf = 1

            self.x[self.srf] = np.array([event.xdata])
            self.y[self.srf] = np.array([event.ydata])

            self.update()
            return

    def update(self):

        # update existing points 
        if self.points[self.srf] is not None:
            self.points[self.srf].set_xdata(self.x[self.srf])
            self.points[self.srf].set_ydata(self.y[self.srf])
        
        # plot points for first time
        if self.points[self.srf] is None and self.x[self.srf].size > 0:
            self.points[self.srf], = self.ax.plot(self.x[self.srf], self.y[self.srf], 'ow', lw=3)
    
        # FIXME: update curve - will need circle, and boundaries, representing the AHA model
        # can only draw if both points are defined
        if self.x[0].size > 0 and self.x[1].size > 0:

            vec = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
            radius = np.linalg.norm(vec)
            
            # rotation matrix
            deg = np.deg2rad(int(360 / self.segs))
            rot = np.array([[np.cos(deg), -np.sin(deg)], [+np.sin(deg), np.cos(deg)]])
            ls = "--" # linestyle for separation of segment 1 and 2

            # if plotting for the first time
            if self.circle1 is None:
                self.circle1 = plt.Circle((self.x[0], self.y[0]), radius, color="white", fill=False, clip_on=True, lw=3) # FIXME: are coordinates correct?
                self.ax.add_patch(self.circle1)
                #self.lines[0] = self.ax.plot((self.x[0], self.x[1]), (self.y[0], self.y[1]), color="white")

                for idx in range(0, self.segs):
                    if idx > 0:
                        vec = rot @ vec
                        ls = "-"
                    self.lines[idx], = self.ax.plot((self.x[0], self.x[0] + vec[0]), (self.y[0], self.y[0] + vec[1]), color="white", lw=3, ls=ls)

            else:
                # FIXME: else update the currently plotted patch
                self.circle1.set_radius(radius)
                self.circle1.set_center((self.x[0], self.y[0]))
                #self.ax.patches[0].set_radius(radius)
                #self.ax.patches[0].set_center((self.x[0], self.y[0]))
                for idx in range(0, self.segs):
                    if idx > 0:
                        vec = rot @ vec
                        ls = "-"
                    self.lines[idx].set_xdata((self.x[0], self.x[0] + vec[0]))
                    self.lines[idx].set_ydata((self.y[0], self.y[0] + vec[1]))

            self.ax.set_xlim(-0.5, self.Nx-0.5)
            self.ax.set_ylim(self.Ny-0.5, -0.5)

        # redraw canvas while idle
        self.fig.canvas.draw_idle()

