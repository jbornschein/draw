from __future__ import division, print_function

from six.moves import cPickle
from blocks.extensions.saveload import Checkpoint, SAVED_TO
from blocks.serialization import secure_dump

class PartsOnlyCheckpoint(Checkpoint):
    def do(self, callback_name, *args):
        """Pickle the save_separately parts (and not the main loop object) to disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            ### this line is disabled from superclass impl to bypass using blocks.serialization.dump
            ### because pickling main thusly causes pickling error:
            ### "RuntimeError: maximum recursion depth exceeded while calling a Python object"
            # secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_dump(getattr(self.main_loop, attribute),
                            filenames[attribute], cPickle.dump)
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
