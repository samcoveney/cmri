"""Some simple utility functions."""

import os

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


def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 20, fill = '#'):
    """Print progress bar."""

    percent = ("{0:3." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

