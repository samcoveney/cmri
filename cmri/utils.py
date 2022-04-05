"""Some simple utility functions."""

import os

class Filename():
    """For conveniently creating filenames.

       flnm = Filename(args.filename)
       flnm.new("registered_" + str(sliceNum))
    """

    def __init__(self, filename):

        self.filename = filename
        self.head, tail = os.path.split(filename)
        self.tail, self.ext = tail.split('.')[0], '.'.join(tail.split('.')[1:]) 

    def new(self, x):
        return os.path.join(self.head, self.tail + "_" + x + "." + self.ext)

    def bval(self):
        return os.path.join(self.head, self.tail + ".bval")

    def bvec(self):
        return os.path.join(self.head, self.tail + ".bvec")


def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 20, fill = '#'):
    """Print progress bar."""

    percent = ("{0:3." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

