from __future__ import division, print_function

import os
import shutil
import theano
import theano.tensor as T

from blocks.extensions.saveload import Checkpoint

from sample import generate_samples


class SampleCheckpoint(Checkpoint):
    def __init__(self, image_size, channels, save_subdir, **kwargs):
        super(SampleCheckpoint, self).__init__(path=None, **kwargs)
        self.image_size = image_size
        self.channels = channels
        self.save_subdir = save_subdir
        self.iteration = 0
        self.epoch_src = "{0}/sample.png".format(save_subdir)

    def do(self, callback_name, *args):
        """Sample the model and save images to disk
        """
        generate_samples(self.main_loop.model, self.save_subdir, self.image_size, self.channels)
        if os.path.exists(self.epoch_src):
            epoch_dst = "{0}/epoch-{1:03d}.png".format(self.save_subdir, self.iteration)
            self.iteration = self.iteration + 1
            shutil.copy2(self.epoch_src, epoch_dst)
            os.system("convert -delay 5 -loop 1 {0}/epoch-*.png {0}/training.gif".format(self.save_subdir))


