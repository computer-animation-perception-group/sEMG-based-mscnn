from __future__ import division
import click
import mxnet as mx
from logbook import Logger
from pprint import pformat
import os
from .utils import packargs, Bunch
from .module_semimyo import Module
from .data import Preprocess, Dataset
from . import Context, constant


logger = Logger('semimyo')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--glove-loss-weight', type=float, required=True)
@click.option('--num-epoch', type=int, default=constant.NUM_EPOCH, help='Maximum epoches')
@click.option('--lr-step', type=int, multiple=True, default=constant.LR_STEP, help='Epoch numbers to decay learning rate')
@click.option('--lr-factor', type=float, multiple=True)
@click.option('--batch-size', type=int, default=constant.BATCH_SIZE, help='Batch size')
@click.option('--lr', type=float, default=constant.LR, help='Base learning rate')
@click.option('--wd', type=float, default=constant.WD, help='Weight decay')
@click.option('--gpu', type=int, multiple=True, default=[0])
@click.option('--log', type=click.Path(), help='Path of the logging file')
@click.option('--snapshot', type=click.Path(), help='Snapshot prefix')
@click.option('--root', type=click.Path(), help='Root path of the experiment, auto create if not exists')
@click.option('--params', type=click.Path(exists=True), help='Inital weights')
@click.option('--ignore-params', multiple=True, help='Ignore params in --params with regex')
@click.option('--adabn', is_flag=True, help='AdaBN for model adaptation, must be used with --num-mini-batch')
@click.option('--num-adabn-epoch', type=int, default=constant.NUM_ADABN_EPOCH)
@click.option('--num-conv-layer', type=int, default=constant.NUM_CONV_LAYER, help='Conv layers')
@click.option('--num-conv-filter', type=int, default=constant.NUM_CONV_FILTER, help='Kernels of the conv layers')
@click.option('--num-lc-layer', type=int, default=constant.NUM_LC_LAYER, help='LC layers')
@click.option('--num-lc-hidden', type=int, default=constant.NUM_LC_HIDDEN, help='Kernels of the LC layers')
@click.option('--lc-kernel', type=int, default=constant.LC_KERNEL)
@click.option('--lc-stride', type=int, default=constant.LC_STRIDE)
@click.option('--lc-pad', type=int, default=constant.LC_PAD)
@click.option('--num-fc-layer', type=int, default=constant.NUM_FC_LAYER, help='FC layers')
@click.option('--num-fc-hidden', type=int, default=constant.NUM_FC_HIDDEN, help='Kernels of the FC layers')
@click.option('--num-bottleneck', type=int, default=constant.NUM_BOTTLENECK, help='Kernels of the bottleneck layer')
@click.option('--dropout', type=float, default=constant.DROPOUT, help='Dropout ratio')
@click.option('--num-glove-layer', type=int, required=True)
@click.option('--num-glove-hidden', type=int, required=True)
@click.option('--num-mini-batch', type=int, default=constant.NUM_MINI_BATCH, help='Split data into mini-batches')
@click.option('--num-eval-epoch', type=int, default=1)
@click.option('--snapshot-period', type=int, default=constant.SNAPSHOT_PERIOD)
@click.option('--fix-params', multiple=True)
@click.option('--decay-all/--no-decay-all', default=constant.DECAY_ALL)
@click.option('--preprocess', callback=lambda ctx, param, value: Preprocess.parse(value))
@click.option('--dataset', type=click.Choice(['s21', 'csl',
                                              'dba', 'dbb', 'dbc',
                                              'ninapro-db1-matlab-lowpass',
                                              'ninapro-db1/caputo',
                                              'ninapro-db1',
                                              'ninapro-db1-raw/semg-glove',
                                              'ninapro-db1/g53',
                                              'ninapro-db1/g5',
                                              'ninapro-db1/g8',
                                              'ninapro-db1/g12']), required=True)
@click.option('--balance-gesture', type=float, default=0)
@click.option('--module', type=click.Choice(['convnet',
                                             'knn',
                                             'svm',
                                             'random-forests',
                                             'lda']), default='convnet')
@click.option('--amplitude-weighting', is_flag=True)
@click.option('--fold', type=int, required=True, help='Fold number of the crossval experiment')
@click.option('--crossval-type', type=click.Choice(['intra-session',
                                                    'universal-intra-session',
                                                    'inter-session',
                                                    'universal-inter-session',
                                                    'intra-subject',
                                                    'universal-intra-subject',
                                                    'inter-subject',
                                                    'one-fold-intra-subject',
                                                    'universal-one-fold-intra-subject']), required=True)
@packargs
def crossval(args):
    if args.root:
        if args.log:
            args.log = os.path.join(args.root, args.log)
        if args.snapshot:
            args.snapshot = os.path.join(args.root, args.snapshot)

    with Context(args.log, parallel=True):
        logger.info('Args:\n{}', pformat(args))
        for i in range(args.num_epoch):
            path = args.snapshot + '-%04d.params' % (i + 1)
            if os.path.exists(path):
                logger.info('Found snapshot {}, exit', path)
                return

        dataset = Dataset.from_name(args.dataset)
        get_crossval_data = getattr(dataset, 'get_%s_data' % args.crossval_type.replace('-', '_'))
        train, val = get_crossval_data(
            batch_size=args.batch_size,
            fold=args.fold,
            preprocess=args.preprocess,
            num_mini_batch=args.num_mini_batch,
            balance_gesture=args.balance_gesture,
            amplitude_weighting=args.amplitude_weighting
        )
        logger.info('Train samples: {}', train.num_sample)
        logger.info('Val samples: {}', val.num_sample)
        mod = Module.parse(
            args.module,
            adabn=args.adabn,
            num_adabn_epoch=args.num_adabn_epoch,
            for_training=True,
            num_eval_epoch=args.num_eval_epoch,
            snapshot_period=args.snapshot_period,
            symbol_kargs=dict(
                num_gesture=dataset.num_gesture,
                num_glove=dataset.num_glove,
                num_semg_row=dataset.num_semg_row,
                num_semg_col=dataset.num_semg_col,
                num_conv_layer=args.num_conv_layer,
                num_conv_filter=args.num_conv_filter,
                num_lc_layer=args.num_lc_layer,
                num_lc_hidden=args.num_lc_hidden,
                lc_kernel=args.lc_kernel,
                lc_stride=args.lc_stride,
                lc_pad=args.lc_pad,
                num_fc_layer=args.num_fc_layer,
                num_fc_hidden=args.num_fc_hidden,
                num_bottleneck=args.num_bottleneck,
                dropout=args.dropout,
                num_glove_layer=args.num_glove_layer,
                num_glove_hidden=args.num_glove_hidden,
                num_mini_batch=args.num_mini_batch,
                glove_loss_weight=args.glove_loss_weight
            ),
            context=[mx.gpu(i) for i in args.gpu]
        )
        mod.fit(
            train_data=train,
            eval_data=val,
            num_epoch=args.num_epoch,
            num_train=train.num_sample,
            batch_size=args.batch_size,
            lr_step=args.lr_step,
            lr_factor=args.lr_factor,
            lr=args.lr,
            wd=args.wd,
            snapshot=args.snapshot,
            params=args.params,
            ignore_params=args.ignore_params,
            fix_params=args.fix_params,
            decay_all=args.decay_all
        )


if __name__ == '__main__':
    cli(obj=Bunch())
