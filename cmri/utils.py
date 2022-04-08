"""Some simple utility functions."""

import os
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap


class Filename():
    """For conveniently creating filenames.

       flnm = Filename(args.filename)
       flnm.new("registered_" + str(sliceNum))
    """

    def __init__(self, filename, bfile, ext):

        self.filename = filename
        self.bfile = bfile
        self.ext = ext
        self.head, tail = os.path.split(filename)
        self.tail = tail.split('.')[0]

        self.bval = os.path.join(self.head, self.bfile + ".bval")
        self.bvec = os.path.join(self.head, self.bfile + ".bvec")

    def new(self, x, ext=None):
        if ext is None:
            ext = self.ext
        return os.path.join(self.head, self.tail + "_" + x + "." + ext)


def progress_bar(iteration, total, prefix='', suffix='', decimals=0, length=20, fill='#'):
    """Print progress bar."""

    percent = ("{0:3." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def cyclic_turbo(deg=70):
    """Create a cyclic turbo colormap with violet on ends.
       Violet begins appearing after +/-deg for range -90 to 90 degrees."""

    # setup colors 
    turbo = cm.get_cmap('turbo', 256)

    newcolors = turbo(np.linspace(0, 1, 256))
    num = np.ceil(((90-deg)/90)*256).astype(int)  # begin change around +/-deg

    violet = np.array([148/256, 0/256, 211/256, 1])
    #violet = (newcolors[num] + newcolors[-num-1]) / 2.0

    for idx in range(0, num):
        newcolors[idx] = newcolors[num, :]*(idx/num) + violet*(1.0 - idx/num)

    for idx in range(0, num):
        newcolors[-idx-1] = newcolors[-num-1, :]*(idx/num) + violet*(1.0 - idx/num)

    cturbo = ListedColormap(newcolors)
    
    return cturbo


def cyclic_bwr():
    """Create a cyclic turbo colormap with violet on ends.
       Violet begins appearing after +/-deg for range -90 to 90 degrees."""

    # setup colors 
    bwr = cm.get_cmap('bwr', 256)
    bwr = bwr(np.linspace(0, 1, 256))
    blue, red, white = bwr[63], bwr[-64], bwr[127]
    violet = (blue + red) / 2.0

    newcolors = np.zeros((256, 4))
    newcolors[..., -1] = 1
    num = 64

    for idx in range(1, 65):
        newcolors[idx - 1] = blue*(idx/num) + white*(1.0 - idx/num)

    for idx in range(1, 65):
        newcolors[64 + idx - 1] = violet*(idx/num) + blue*(1.0 - idx/num)

    for idx in range(1, 65):
        newcolors[128 + idx - 1] = red*(idx/num) + violet*(1.0 - idx/num)

    for idx in range(1, 65):
        newcolors[192 + idx - 1] = white*(idx/num) + red*(1.0 - idx/num)

    print(newcolors)
    new = ListedColormap(newcolors)
    print(new)
    
    return new
