import six
import gc

import chainer
from chainer import training 
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module


class MyUpdater(training.updaters.StandardUpdater):
    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0

        self.loss_scale = loss_scale
        if loss_scale is not None:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.set_loss_scale(loss_scale)

    def update_core(self):
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        loss = optimizer.target(**in_arrays)
        optimizer.target.cleargrads()
        loss.backward()
#        loss.unchain_backward()
        #del loss
        #gc.collect()
        optimizer.update()
        #if isinstance(in_arrays, tuple):
        #    optimizer.update(loss_func, *in_arrays)
        #elif isinstance(in_arrays, dict):
        #    optimizer.update(loss_func, **in_arrays)
        #else:
        #    optimizer.update(loss_func, in_arrays)
