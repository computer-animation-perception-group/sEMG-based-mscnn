from __future__ import division
import logging
from logbook import Logger
import mxnet as mx
import numpy as np
from functools import partial
from contextlib import contextmanager
from nose.tools import assert_equal
import re
import time
import os
from .symbol_semimyo import get_symbol
from . import ROOT, constant, utils


logger = Logger('sigr')


class SymbolMixin(object):

    def __init__(self, **kargs):
        symbol = kargs.pop('symbol', None)
        symbol_kargs = kargs.pop('symbol_kargs', {}).copy()
        if symbol is None:
            symbol = get_symbol(**symbol_kargs)
        super(SymbolMixin, self).__init__(symbol=symbol, **kargs)


class EvalMixin(object):

    def __init__(self, **kargs):
        self.num_eval_epoch = kargs.pop('num_eval_epoch', 1)
        super(EvalMixin, self).__init__(**kargs)

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_batch_end_callback=None, initializer=mx.init.Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None):
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                              eval_metric=eval_metric,
                                                              locals=locals())
                    for callback in mx.module.base_module._as_list(batch_end_callback):
                        callback(batch_end_params)

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            if epoch_end_callback is not None:
                arg_params, aux_params = self.get_params()
                for callback in mx.module.base_module._as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if eval_data and (epoch + 1) % self.num_eval_epoch == 0:
                res = self.score(eval_data, validation_metric,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()


class MxNetModule(mx.mod.Module):

    def __init__(self, **kargs):
        super(MxNetModule, self).__init__(
            **{k: kargs[k] for k in kargs
               if k in ['symbol', 'data_names', 'label_names', 'logger',
                        'context', 'work_load_list']})


class Base(EvalMixin, SymbolMixin, MxNetModule):

    @property
    def num_semg_row(self):
        return self.symbol.num_semg_row

    @property
    def num_semg_col(self):
        return self.symbol.num_semg_col

    @property
    def data_shape_1(self):
        return self.symbol.data_shape_1

    def __init__(self, **kargs):
        parent = kargs.pop('parent', {})
        kargs['data_names'] = kargs.get('data_names', getattr(parent, '_data_names', ('data',)))
        kargs['label_names'] = kargs.get('label_names', getattr(parent, '_label_names', ('softmax_label',)))
        super(Base, self).__init__(**kargs)

    def init_params(self, initializer=mx.init.Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'

        if self._arg_params is None:
            param_arrays = [mx.nd.zeros(x[0].shape) for x in self._exec_group.param_arrays]
            self._arg_params = {name: arr for name, arr in zip(self._param_names, param_arrays)}

        if self._aux_params is None:
            aux_arrays = [mx.nd.zeros(x[0].shape) for x in self._exec_group.aux_arrays]
            self._aux_params = {name: arr for name, arr in zip(self._aux_names, aux_arrays)}

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        assert cache_arr.shape == arr.shape, '{} {} {}'.format(name, cache_arr.shape, arr.shape)
                        cache_arr.copyto(arr)
                else:
                    assert allow_missing, name
                    initializer(name, arr)
            else:
                initializer(name, arr)

        for name, arr in self._arg_params.items():
            _impl(name, arr, arg_params)

        for name, arr in self._aux_params.items():
            _impl(name, arr, aux_params)

        self.params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec_group.set_params(self._arg_params, self._aux_params)


class AdaBNMixin(object):

    def __init__(self, **kargs):
        parent = kargs.get('parent', {})
        self.downsample = kargs.pop('downsample', getattr(parent, 'downsample', False))
        self.adabn = kargs.pop('adabn', getattr(parent, 'adabn', False))
        self.num_adabn_epoch = kargs.pop('num_adabn_epoch',
                                         getattr(parent, 'num_adabn_epoch', constant.NUM_ADABN_EPOCH))
        super(AdaBNMixin, self).__init__(**kargs)

    @contextmanager
    def _restore_eval_data(self, eval_data):
        shuffle = eval_data.shuffle
        eval_data.shuffle = True
        downsample = eval_data.downsample
        eval_data.downsample = self.downsample
        last_batch_handle = eval_data.last_batch_handle
        eval_data.last_batch_handle = 'roll_over'
        try:
            yield
        finally:
            eval_data.shuffle = shuffle
            eval_data.downsample = downsample
            eval_data.last_batch_handle = last_batch_handle

    def _update_adabn(self, eval_data):
        '''Update moving mean and moving var with eval data'''
        from time import time
        start = time()
        with self._restore_eval_data(eval_data):
            for _ in range(self.num_adabn_epoch):
                eval_data.reset()
                for nbatch, eval_batch in enumerate(eval_data):
                    self.forward(eval_batch, is_train=True)
                    for out in self.get_outputs():
                        #  Cause memory leak (though not increase after this _update_adabn) without this wait
                        #  TODO: fixme
                        out.wait_to_read()
                    #  for name, block in zip(self._exec_group.aux_names, self._exec_group.aux_arrays):
                        #  if 'moving' in name:
                            #  for a in block:
                                #  a.wait_to_read()
        logger.debug(
            'AdaBN with {} epochs takes {} seconds',
            self.num_adabn_epoch,
            time() - start
        )

    def _try_update_adabn(self, eval_data, reset):
        assert self.binded and self.params_initialized
        if self.adabn:
            self._update_adabn(eval_data)
        if not reset and self.adabn:
            eval_data.reset()

    def score(
        self,
        eval_data,
        eval_metric,
        num_batch=None,
        batch_end_callback=None,
        reset=True,
        epoch=0
    ):
        self._try_update_adabn(eval_data, reset)
        return super(AdaBNMixin, self).score(
            eval_data,
            eval_metric,
            num_batch,
            batch_end_callback,
            reset,
            epoch
        )

    def predict(
        self,
        eval_data,
        num_batch=None,
        merge_batches=True,
        reset=True,
        always_output_list=False
    ):
        self._try_update_adabn(eval_data, reset)
        return super(AdaBNMixin, self).predict(
            eval_data,
            num_batch,
            merge_batches,
            reset,
            always_output_list
        )


class Module(AdaBNMixin, Base):

    @classmethod
    def parse(cls, text, **kargs):
        if text == 'convnet':
            return cls(**kargs)
        from .base_module import BaseModule
        from .sklearn_module import SklearnModule
        assert SklearnModule
        return BaseModule.parse(text, **kargs)

    def __init__(self, **kargs):
        self.kargs = kargs.copy()
        self.for_training = kargs.pop('for_training')
        self.snapshot_period = kargs.pop('snapshot_period', 1)
        symbol_kargs = kargs.pop('symbol_kargs', {}).copy()
        symbol_kargs.update(
            for_training=self.for_training
        )
        kargs['symbol_kargs'] = symbol_kargs
        super(Module, self).__init__(
            data_names=['semg'],
            label_names=['gesture', 'glove'],
            **kargs
        )

    def _get_init(self):
        return Init(factor_type='in', magnitude=2, mod=self)

    def fit(self, **kargs):
        num_epoch = kargs.pop('num_epoch')
        num_train = kargs.pop('num_train')
        batch_size = kargs.pop('batch_size')
        epoch_size = num_train / batch_size
        lr_step = kargs.pop('lr_step')
        lr = kargs.pop('lr', constant.LR)
        wd = kargs.pop('wd', constant.WD)
        snapshot = kargs.pop('snapshot')
        params = kargs.pop('params', None)
        ignore_params = kargs.pop('ignore_params', [])
        fix_params = kargs.pop('fix_params', [])
        decay_all = kargs.pop('decay_all', False)
        lr_factor = kargs.pop('lr_factor', ()) or constant.LR_FACTOR

        checkpoint = []
        if snapshot:
            def do_checkpoint(prefix, period=1):
                def _callback(iter_no, sym, arg, aux):
                    #  Always save the first epoch
                    if iter_no == 0 or (iter_no + 1) % period == 0:
                        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
                period = int(max(1, period))
                return _callback
            checkpoint.append(do_checkpoint(snapshot, period=self.snapshot_period))

        if params:
            logger.info('Load params from {}', params)
            init = Load(
                params,
                default_init=self._get_init(),
                ignore=ignore_params,
                mod=self
            )
        else:
            init = self._get_init()

        #  self.monitor = mx.mon.Monitor(
            #  50,
            #  lambda d: mx.nd.norm(d) / np.sqrt(d.size),
            #  #  pattern='subject_.*fc\d*_(?:output|weight)',
            #  pattern='.*(gesture_last_fc|subject_last_fc|bottleneck)_(backward_data|backward_weight|_scale).*',
            #  sort=True
        #  )
        #  self.monitor_installed = False

        def nbatch(param):
            nbatch.value = param.nbatch

        batch_end_callback = [mx.callback.Speedometer(batch_size, 50), nbatch]

        #  def debug(params):
            #  data = params.locals['data_batch'].data[0].asnumpy()
            #  label = params.locals['data_batch'].label[0].asnumpy()
            #  import joblib as jb
            #  jb.dump(dict(data=data, label=label), 'dump')
            #  import sys
            #  sys.exit(0)

        #  batch_end_callback.append(debug)

        return super(Module, self).fit(
            eval_metric=Accuracy(0, 'g'),
            optimizer='Sigropt',
            optimizer_params=dict(
                learning_rate=lr,
                momentum=0.9,
                wd=wd,
                lr_scheduler=Scheduler(
                    lr_step=lr_step,
                    factor=lr_factor,
                    epoch_size=epoch_size
                ),
                get_nbatch=lambda: getattr(nbatch, 'value', -1),
                clip_gradient=1,
                fix_params=fix_params,
                decay_all=decay_all
            ),
            initializer=init,
            num_epoch=num_epoch,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=checkpoint,
            **kargs
        )

    def forward_backward(self, data_batch):
        if hasattr(self, 'monitor'):
            if not self.monitor_installed:
                for exe in self._exec_group.execs:
                    self.monitor.install(exe)
                self.monitor_installed = True
            self.monitor.tic()
        super(Module, self).forward_backward(data_batch)

    def update(self):
        super(Module, self).update()
        if hasattr(self, 'monitor'):
            self.monitor.toc_print()

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        opt = mx.optimizer

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = mx.model._create_kvstore(kvstore, len(self._context), self._arg_params)

        if isinstance(optimizer, str):
            batch_size = self._exec_group.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            idx2name = {}
            # Fix bug in mxnet for multi-gpu training
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i*len(self._context)+k: n
                                    for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = 1.0/batch_size
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if not update_on_kvstore:
            self._updater = opt.get_updater(optimizer)
        if kvstore:
            # copy initialized local parameters to kvstore
            mx.model._initialize_kvstore(kvstore=kvstore,
                                         param_arrays=self._exec_group.param_arrays,
                                         arg_params=self._arg_params,
                                         param_names=self._param_names,
                                         update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)

        self.optimizer_initialized = True


class Accuracy(mx.metric.EvalMetric):

    def __init__(self, index, name):
        super(Accuracy, self).__init__('accuracy[%s]' % name)
        if not isinstance(index, list):
            index = [index]
        self.index = index

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)

        for index in self.index:
            label = labels[index].asnumpy().astype('int32')
            assert label.ndim in (1, 2)
            if label.ndim == 1:
                pred_label = mx.nd.argmax_channel(preds[index]).asnumpy().astype('int32')
            else:
                pred_label = (preds[index].asnumpy() > 0.5).astype('int32')

            # mx.metric.check_label_shapes(label, pred_label)

            if label.ndim == 1:
                mask = label >= 0
                label = label[mask]
                pred_label = pred_label[mask]

            # mx.metric.check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label == label).sum()
            self.num_inst += pred_label.size


class Scheduler(mx.lr_scheduler.LRScheduler):

    def __init__(self, lr_step, factor, epoch_size):
        if not isinstance(lr_step, (tuple, list)):
            lr_step = list(range(lr_step, 1000, lr_step))
        step = [epoch_size * s for s in lr_step]
        super(Scheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        if self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= (
                    self.factor[self.cur_step_ind - 1]
                    if isinstance(self.factor, (tuple, list)) else self.factor
                )
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
        return self.base_lr


class Load(mx.init.Load):

    def __init__(self, params, *args, **kargs):
        self.ignore = kargs.pop('ignore', [])
        mod = kargs.pop('mod')
        if not os.path.exists(params) and os.path.exists(os.path.join(ROOT, params)):
            params = os.path.join(ROOT, params)
        super(Load, self).__init__(params, *args, **kargs)
        for name in list(self.param):
            for ignore in self.ignore:
                if re.match(ignore, name):
                    logger.info('Ignore param {}', name)
                    del self.param[name]
                    break
            if mod.adabn and (name.endswith('moving_mean')
                              or name.endswith('moving_var')):
                del self.param[name]

    def __call__(self, name, arr):
        if name in self.param and ('gamma' in name or 'beta' in name):
            self.param[name] = self.param[name].reshape(arr.shape)
        return super(Load, self).__call__(name, arr)


class Init(mx.init.Xavier):

    def __init__(self, *args, **kargs):
        self.mod = kargs.pop('mod')
        super(Init, self).__init__(*args, **kargs)

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if name.endswith('lambda'):
            self._init_zero(name, arr)
        elif name.endswith('_zero'):
            self._init_zero(name, arr)
        elif name.endswith('_one'):
            self._init_one(name, arr)
        elif name.startswith('sum'):
            self._init_gamma(name, arr)
        elif 'im2col' in name and name.endswith('weight'):
            self._init_im2col(name, arr)
        elif name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

    def _init_one(self, _, arr):
        arr[:] = 1

    def _init_proj(self, _, arr):
        '''Initialization of shortcut of kenel (2, 2)'''
        w = np.zeros(arr.shape, np.float32)
        for i in range(w.shape[1]):
            w[i, i, ...] = 0.25
        arr[:] = w

    def _init_im2col(self, _, arr):
        assert_equal(len(arr.shape), 4)
        assert_equal(arr.shape[0], arr.shape[1] * arr.shape[2] * arr.shape[3])
        arr[:] = np.eye(arr.shape[0]).reshape(arr.shape)


@mx.optimizer.register
class Sigropt(mx.optimizer.NAG):

    def __init__(self, get_nbatch, fix_params, decay_all, **kargs):
        self.decay_all = decay_all
        super(Sigropt, self).__init__(**kargs)
        self.get_nbatch = get_nbatch
        self.confuse = 0
        self.fix_params = fix_params

    @property
    def nbatch(self):
        return self.get_nbatch()

    def update(self, index, weight, grad, state):
        """Update the parameters.
        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters
        weight : NDArray
            weight ndarray
        grad : NDArray
            grad ndarray
        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.nd.NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)

        if lr:
            if state:
                mom = state
                mom *= self.momentum
                if wd:
                    # L1 reg
                    # grad += wd * mx.nd.sign(weight)
                    grad += wd * weight
                mom += grad
                grad += self.momentum * mom
                weight += -lr * grad
            else:
                assert self.momentum == 0.0
                if wd:
                    # weight += -lr * (grad + self.wd * mx.nd.sign(weight))
                    weight += -lr * (grad + self.wd * weight)
                else:
                    weight += -lr * grad

    def set_wd_mult(self, args_wd_mult):
        """Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        if not self.decay_all:
            return super(Sigropt, self).set_wd_mult(args_wd_mult)

        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (n.endswith('_weight')
                    or n.endswith('_gamma')
                    or n.endswith('_bias')
                    or n.endswith('_beta')) or 'zscore' in n:
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.list_attr(recursive=True)
            for k, v in attr.items():
                if k.endswith('_wd_mult'):
                    self.wd_mult[k[:-len('_wd_mult')]] = float(v)
        self.wd_mult.update(args_wd_mult)

    def _get_lr(self, index):
        """get learning rate for index.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        lr : float
            learning rate for this index
        """
        lr = 1

        if index not in self.lr_mult and index in self.idx2name:
            index = self.idx2name[index]
        assert isinstance(index, str)

        if self.fix_params:
            for pattern in self.fix_params:
                if re.match(pattern, index):
                    return 0

        return super(Sigropt, self)._get_lr(index) if lr else 0


class RuntimeMixin(object):

    def __init__(self, **kargs):
        args = []
        backup = kargs.copy()
        self.params = kargs.pop('params')
        super(RuntimeMixin, self).__init__(**kargs)
        self.args = args
        self.kargs = backup

    def predict(self, eval_data, *args, **kargs):
        if not getattr(self, 'incache', False):
            return _predict(utils.LazyProxy(self.Clone), eval_data)

        if not self.binded:
            self.bind(data_shapes=[('semg', (eval_data.batch_size, self.data_shape_1,
                                             self.num_semg_row, self.num_semg_col))],
                      for_training=False)
        if not self.params_initialized:
            self.init_params(Load(self.params, default_init=self._get_init(), mod=self))
        return super(RuntimeMixin, self).predict(eval_data, *args, **kargs)

    def predict_proba(self, eval_data, *args, **kargs):
        if not getattr(self, 'incache', False):
            return _predict_proba(utils.LazyProxy(self.Clone), eval_data)

        if not self.binded:
            self.bind(data_shapes=[('semg', (eval_data.batch_size, self.data_shape_1,
                                             self.num_semg_row, self.num_semg_col))],
                      for_training=False)
        if not self.params_initialized:
            self.init_params(Load(self.params, default_init=self._get_init(), mod=self))
        return super(RuntimeMixin, self).predict(eval_data, *args, **kargs)

    @property
    def Clone(self):
        return partial(type(self), *self.args, **self.kargs)


@utils.cached
def _predict(mod, val):
    mod.incache = True

    val.reset()
    true = val.gesture.copy()
    segment = val.segment.copy()
    val.reset()
    assert np.all(true == val.gesture.copy())
    assert np.all(segment == val.segment.copy())

    out = mod.predict(val).asnumpy()
    assert_equal(out.ndim, 2)
    #  assert_equal(out.shape[1], mod.num_gesture)
    pred = out.argmax(axis=1)
    assert_equal(true.shape, pred.shape)
    return pred, true, segment


def _predict_proba(mod, val):
    mod.incache = True

    val.reset()
    true = val.gesture.copy()
    segment = val.segment.copy()
    val.reset()
    assert np.all(true == val.gesture.copy())
    assert np.all(segment == val.segment.copy())

    out = mod.predict(val).asnumpy()
    return out, true, segment


class RuntimeModule(RuntimeMixin, Module):
    pass
