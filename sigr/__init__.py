import mxnet as mx
import numpy as np
import random

mx.random.seed(42)
np.random.seed(43)
random.seed(44)

import os

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CACHE = os.path.join(ROOT, '.cache')
#CACHE = '/home/weiwentao/exp/.cache/'


from contextlib import contextmanager


@contextmanager
def Context(log=None, parallel=False, level=None):
    from .utils import logging_context
    with logging_context(log, level=level):
        if not parallel:
            yield
        else:
            import joblib as jb
            from multiprocessing import cpu_count
            with jb.Parallel(n_jobs=cpu_count()) as par:
                Context.parallel = par
                yield


def _patch(func):
    func()
    return lambda: None


@_patch
def _patch_click():
    import click
    orig = click.option

    def option(*args, **kargs):
        if 'help' in kargs and 'default' in kargs:
            kargs['help'] += ' (default {})'.format(kargs['default'])
        return orig(*args, **kargs)

    click.option = option


from .data import s21 as data_s21


__all__ = ['ROOT', 'CACHE', 'Context', 'data_s21']
