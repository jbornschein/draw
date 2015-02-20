
from __future__ import division

import logging

from blocks.extensions import TrainingExtension
from distutils.version import LooseVersion


logger = logging.getLogger(__name__)


try:
    import progressbar

    if LooseVersion(progressbar.__version__) < LooseVersion("2.3"):
        logger.warning("Missing required dependency: progressbar2 (>= 2.3), disabling ProgressBar extension")
        PROGRESSBAR_AVAILABLE = False
    else:
        PROGRESSBAR_AVAILABLE = True
except ImportError: 
    PROGRESSBAR_AVAILABLE = False


class ProgressBar(TrainingExtension):
    """Display a progress bar during training.

    This extension tries to obtain the number of mini-batches processed during
    a training epoch and will display progress bar. Not all IterationSchemes 
    provide the necessary atributes 

    Note   
    ----
    This extension should be run before other extensions that print to the screen 
    at the end or at the beginning of the epoch (e.g. the Printing exctension). 
    Placing PrpgressBar before these extension will ensure you won't get 
    intermingled output on your terminal.
    """
    def __init__(self, **kwargs):
        super(ProgressBar, self).__init__(**kwargs)

        self.bar = None
        self.epoch_counter = 0
        self.batch_counter = 0

        if not PROGRESSBAR_AVAILABLE:
            logger.warning("")

    def before_epoch(self):
        self.epoch_counter += 1
        self.batch_counter = 0
        
        if not PROGRESSBAR_AVAILABLE:
            return 

        iteration_scheme = self.main_loop.data_stream.iteration_scheme
        if hasattr(iteration_scheme, 'batches_per_epoch'):  
            batches_per_epoch = iteration_scheme.batches_per_epoch
        elif hasattr(iteration_scheme, 'num_examples') and hasattr(iteration_scheme, 'batch_size'):
            batches_per_epoch = iteration_scheme.num_examples // iteration_scheme.batch_size
        else:
            logger.warning("Disabling ProgressBar: The training iteration scheme does not provide the necessary attributes to calculate batches_per_epoch")
            return
      
        widgets = [ "Epoch {}, step ".format(self.epoch_counter),
                    progressbar.Counter(), ' (', progressbar.Percentage(), ') ',
                    progressbar.Bar(), ' ', progressbar.Timer(), ' ', progressbar.ETA()]
        self.bar = progressbar.ProgressBar(widgets=widgets, maxval=batches_per_epoch)

    def after_epoch(self):
        if PROGRESSBAR_AVAILABLE and self.bar:
            self.bar.finish()

    def before_batch(self, batch):
        if PROGRESSBAR_AVAILABLE and self.bar:
            if self.batch_counter == 0:
                self.bar.start()
            self.batch_counter += 1
            self.bar.update(self.batch_counter)

