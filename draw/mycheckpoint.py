
from __future__ import division, print_function

import os
import shutil
import theano
import theano.tensor as T


import blocks.extras


from blocks.extensions.saveload import Checkpoint

from sample import generate_samples
from blocks.serialization import secure_pickle_dump


LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"


class MyCheckpoint(Checkpoint):
    def __init__(self, image_size, save_subdir, **kwargs):
        super(MyCheckpoint, self).__init__(**kwargs)
        self.image_size = image_size
        self.save_subdir = save_subdir
        self.iteration = 0
        self.epoch_src = "{0}/sample.png".format(save_subdir)

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        from_main_loop, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if len(from_user):
                path, = from_user
#            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
#            self.main_loop.log.current_row[SAVED_TO] = (
#                already_saved_to + (path,))
#            secure_pickle_dump(self.main_loop, path)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                p = getattr(self.main_loop, attribute)
                if p:
                    secure_pickle_dump(p, filenames[attribute])
                else:
                    print("Empty %s",attribute)
            generate_samples(self.main_loop.model, self.save_subdir, self.image_size)
            if os.path.exists(self.epoch_src):
                epoch_dst = "{0}/epoch-{1:03d}.png".format(self.save_subdir, self.iteration)
                self.iteration = self.iteration + 1
                shutil.copy2(self.epoch_src, epoch_dst)
                os.system("convert -delay 5 -loop 1 {0}/epoch-*.png {0}/training.gif".format(self.save_subdir))

        except Exception:
            self.main_loop.log.current_row[SAVED_TO] = None
            raise


