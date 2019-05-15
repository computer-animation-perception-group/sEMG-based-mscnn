from __future__ import division
import click
import mxnet as mx
from logbook import Logger
from pprint import pformat
import os
from .utils import packargs, Bunch
from .module import Module
from .module_multistream import Module as Module_multistream
from .data import Preprocess, Dataset
from . import Context, constant
from .genIndex import genIndex
import types

logger = Logger('semgfeature')



@click.group()
def cli():
    pass


@cli.group()
@click.option('--downsample', type=int, default=0)
@click.option('--num-semg-row', type=int, default=constant.NUM_SEMG_ROW, help='Rows of sEMG image')
@click.option('--num-semg-col', type=int, default=constant.NUM_SEMG_COL, help='Cols of sEMG image')
@click.option('--num-epoch', type=int, default=60, help='Maximum epoches')
@click.option('--num-tzeng-batch', type=int, default=constant.NUM_TZENG_BATCH,
              help='Batch number of each Tzeng update, 2 means interleaved domain and label update')
@click.option('--lr-step', type=int, multiple=True, default=[20, 40], help='Epoch numbers to decay learning rate')
@click.option('--lr-factor', type=float, multiple=True)
@click.option('--batch-size', type=int, default=1000,
              help='Batch size, should be 900 with --minibatch for s21 inter-subject experiment')
@click.option('--lr', type=float, default=0.1, help='Base learning rate')
@click.option('--wd', type=float, default=0.0001, help='Weight decay')
@click.option('--subject-wd', type=float, help='Weight decay multiplier of the subject branch')
@click.option('--gpu', type=int, multiple=True, default=[0])
@click.option('--gamma', type=float, default=constant.GAMMA, help='Gamma in RevGrad')
@click.option('--log', type=click.Path(), help='Path of the logging file')
@click.option('--snapshot', type=click.Path(), help='Snapshot prefix')
@click.option('--root', type=click.Path(), help='Root path of the experiment, auto create if not exists')
@click.option('--revgrad', is_flag=True, help='Use RevGrad')
@click.option('--num-revgrad-batch', type=int, default=2,
              help=('Batch number of each RevGrad update, 2 means interleaved domain and label update, '
                    'see "Adversarial Deep Averaging Networks for Cross-Lingual Sentiment Classification" for details'))
@click.option('--tzeng', is_flag=True, help='Use Tzeng_ICCV_2015')
@click.option('--confuse-conv', is_flag=True, help='Domain confusion (for both RevGrad and Tzeng) on conv2')
@click.option('--confuse-all', is_flag=True, help='Domain confusion (for both RevGrad and Tzeng) on all layers')
@click.option('--subject-loss-weight', type=float, default=1, help='Ganin et al. use 0.1 in their code')
@click.option('--subject-confusion-loss-weight', type=float, default=1,
              help='Tzeng confusion loss weight, larger than 1 seems better')
@click.option('--lambda-scale', type=float, default=constant.LAMBDA_SCALE,
              help='Global scale of lambda in RevGrad, 1 in their paper and 0.1 in their code')
@click.option('--params', type=click.Path(exists=True), help='Inital weights')
@click.option('--ignore-params', multiple=True, help='Ignore params in --params with regex')
@click.option('--random-shift-fill', type=click.Choice(['zero', 'margin']),
              default=constant.RANDOM_SHIFT_FILL, help='Random shift filling value')
@click.option('--random-shift-horizontal', type=int, default=0, help='Random shift input horizontally by x pixels')
@click.option('--random-shift-vertical', type=int, default=0, help='Random shift input vertically by x pixels')
@click.option('--random-scale', type=float, default=0,
              help='Random scale input data globally by 2^scale, and locally by 2^(scale/4)')
@click.option('--random-bad-channel', type=float, multiple=True, default=[],
              help='Random (with a probability of 0.5 for each image) assign a pixel as specified value, usually [-1, 0, 1]')
@click.option('--num-feature-block', type=int, default=constant.NUM_FEATURE_BLOCK, help='Number of FC layers in feature extraction part')
@click.option('--num-gesture-block', type=int, default=constant.NUM_GESTURE_BLOCK, help='Number of FC layers in gesture branch')
@click.option('--num-subject-block', type=int, default=constant.NUM_SUBJECT_BLOCK, help='Number of FC layers in subject branch')
@click.option('--adabn', is_flag=True, help='AdaBN for model adaptation, must be used with --minibatch')
@click.option('--num-adabn-epoch', type=int, default=constant.NUM_ADABN_EPOCH)
@click.option('--num-pixel', type=int, default=constant.NUM_PIXEL, help='Pixelwise reduction layers')
@click.option('--num-filter', type=int, default=constant.NUM_FILTER, help='Kernels of the conv layers')
@click.option('--num-hidden', type=int, default=constant.NUM_HIDDEN, help='Kernels of the FC layers')
@click.option('--num-bottleneck', type=int, default=constant.NUM_BOTTLENECK, help='Kernels of the bottleneck layer')
@click.option('--dropout', type=float, default=constant.DROPOUT, help='Dropout ratio')
@click.option('--window', type=int, default=1, help='Multi-frame as image channels')
@click.option('--lstm-window', type=int)
@click.option('--num-presnet', type=int, multiple=True, help='Deprecated')
@click.option('--presnet-branch', type=int, multiple=True, help='Deprecated')
@click.option('--drop-presnet', is_flag=True)
@click.option('--bng', is_flag=True, help='Deprecated')
@click.option('--minibatch', is_flag=True, help='Split data into minibatch by subject id')
@click.option('--drop-branch', is_flag=True, help='Dropout after each FC in branches')
@click.option('--pool', is_flag=True, help='Deprecated')
@click.option('--fft', is_flag=True, help='Deprecaded. Perform FFT and use spectrum amplitude as image channels. Cannot be used on non-uniform (segment length) dataset like NinaPro')
@click.option('--fft-append', is_flag=True, help='Append FFT feature to raw frames in channel axis')
@click.option('--dual-stream', is_flag=True, help='Use raw frames and FFT feature as dual-stream')
@click.option('--zscore/--no-zscore', default=True, help='Use z-score normalization on input')
@click.option('--zscore-bng', is_flag=True, help='Use global BatchNorm as z-score normalization, for window > 1 or FFT')
@click.option('--lstm', is_flag=True)
@click.option('--num-lstm-hidden', type=int, default=constant.NUM_LSTM_HIDDEN, help='Kernels of the hidden layers in LSTM')
@click.option('--num-lstm-layer', type=int, default=constant.NUM_LSTM_LAYER, help='Number of the hidden layers in LSTM')
@click.option('--dense-window/--no-dense-window', default=True, help='Dense sampling of windows during training')
@click.option('--lstm-last', type=int, default=0)
@click.option('--lstm-dropout', type=float, default=constant.LSTM_DROPOUT, help='LSTM dropout ratio')
@click.option('--lstm-shortcut', is_flag=True)
@click.option('--lstm-bn/--no-lstm-bn', default=True, help='BatchNorm in LSTM')
@click.option('--lstm-grad-scale/--no-lstm-grad-scale', default=True, help='Grad scale by the number of LSTM output')
@click.option('--faug', type=float, default=0)
@click.option('--faug-classwise', is_flag=True)
@click.option('--num-eval-epoch', type=int, default=1)
@click.option('--snapshot-period', type=int, default=1)
@click.option('--gpu-x', type=int, default=0)
@click.option('--drop-conv', is_flag=True)
@click.option('--drop-pixel', type=int, multiple=True, default=(-1,))
@click.option('--drop-presnet-branch', is_flag=True)
@click.option('--drop-presnet-proj', is_flag=True)
@click.option('--fix-params', multiple=True)
@click.option('--presnet-proj-type', type=click.Choice(['A', 'B']), default='A')
@click.option('--decay-all', is_flag=True)
@click.option('--presnet-promote', is_flag=True)
@click.option('--pixel-reduce-loss-weight', type=float, default=0)
@click.option('--fast-pixel-reduce/--no-fast-pixel-reduce', default=True)
@click.option('--pixel-reduce-bias', is_flag=True)
@click.option('--pixel-reduce-kernel', type=int, multiple=True, default=(1, 1))
@click.option('--pixel-reduce-stride', type=int, multiple=True, default=(1, 1))
@click.option('--pixel-reduce-pad', type=int, multiple=True, default=(0, 0))
@click.option('--pixel-reduce-norm', is_flag=True)
@click.option('--pixel-reduce-reg-out', is_flag=True)
@click.option('--num-pixel-reduce-filter', type=int, multiple=True, default=tuple(None for _ in range(constant.NUM_PIXEL)))
@click.option('--num-conv', type=int, default=2)
@click.option('--pixel-same-init', is_flag=True)
@click.option('--presnet-dense', is_flag=True)
@click.option('--conv-shortcut', is_flag=True)
@click.option('--preprocess', callback=lambda ctx, param, value: Preprocess.parse(value))
@click.option('--bandstop', is_flag=True)
@click.option('--fusion-type', type=click.Choice(['fuse_1','fuse_2',
                                                  'fuse_3','fuse_4', 
                                                  'fuse_5',
                                                  'softmax_fuse_1',
                                                  'softmax_fuse_2',
                                                  'softmaxwithfc_fuse_1',
                                                  'multistream_multistruct_fuse_1',
                                                  'multistream_multistruct_fuse_2',
                                                  'twostep_fuse_1']), default='fuse_1')
@click.option('--feature-name', type=click.Choice(['actimg','sigimg',
                                                   'rawimg','featuremap',
                                                   'featureimage','chdiffimage', 
                                                   'chdiffsigimage', 'chdiff_sigimg_multistream',
                                                   'ch_multistream',
                                                   'block_multistream',
                                                   'block_multistream_v2',
                                                   'ch_multistream_plus_rawimg',
                                                   'featureimg_rawimg_multistream',
                                                   'sigimg_rawimg_multistream', 'frame_multistream',
                                                   'simplestackedimg', 
                                                   'rawfeatureimg',
                                                   'rawsemg_feature_singlestream',
                                                   'rawsemg_feature_multisource',
                                                   'singleframe_multistream',
                                                   'chwise_feature_multistream',
                                                   'chwise_feature_rawimg_multistream']), required=True)
@click.option('--dataset', type=click.Choice(['s21', 'csl',
                                              'dba', 'dbb', 'dbc',
                                              'ninapro-db1-matlab-lowpass',
                                              'ninapro-db1-sigimg',
                                              'ninapro-db1-sigimg-fast',
                                              'ninapro-db1-softmax-as-input',
                                              'ninapro-db1-rawdata-semgfeature-multisource',
                                              'ninapro-db1-features',
                                              'ninapro-db1/caputo',
                                              'ninapro-db1',
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
@packargs
def exp(args):
    pass


@exp.command()
@click.option('--fold', type=int, required=True, help='Fold number of the crossval experiment')
@click.option('--crossval-type', type=click.Choice(['intra-session',
                                                    'universal-intra-session',
                                                    'inter-session',
                                                    'universal-inter-session',
                                                    'intra-subject',
                                                    'universal-intra-subject',
                                                    'inter-subject',
                                                    'one-fold-intra-subject',
                                                    'universal-one-fold-intra-subject',
                                                    'universal-one-fold-intra-subject-test',
                                                    'one-fold-inter-subject-test']), required=True)
@packargs
def crossval(args):
    if args.root:
        if args.log:
            args.log = os.path.join(args.root, args.log)
        if args.snapshot:
            args.snapshot = os.path.join(args.root, args.snapshot)

    if args.gpu_x:
        args.gpu = sum([list(args.gpu) for i in range(args.gpu_x)], [])

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
            adabn=args.adabn,
            minibatch=args.minibatch,
            balance_gesture=args.balance_gesture,
            amplitude_weighting=args.amplitude_weighting,
            random_shift_fill=args.random_shift_fill,
            random_shift_horizontal=args.random_shift_horizontal,
            random_shift_vertical=args.random_shift_vertical,
            feature_name = args.feature_name,
            window=args.window,
            num_semg_row = args.num_semg_row,
            num_semg_col = args.num_semg_col
        )
        logger.info('Train samples: {}', train.num_sample)
        logger.info('Val samples: {}', val.num_sample)
         
        if args.feature_name == 'sigimg' or args.feature_name == 'actimg':
             semg_rows = args.window             
             semg_cols = len(genIndex(args.num_semg_col*args.num_semg_row))-1        
        elif args.feature_name == 'rawimg':
             semg_rows = args.window 
             semg_cols = args.num_semg_col*args.num_semg_row
        elif args.feature_name == 'featuremap':
             semg_rows = args.num_semg_row
             semg_cols = args.num_semg_col
        elif args.feature_name == 'featureimage':  
             semg_rows = train.feature_dim          
             semg_cols =  args.num_semg_col*args.num_semg_row 
             
        elif args.feature_name == 'chdiffimage':
#             semg_rows = args.window 
#             semg_cols = train.shape_1
             semg_rows = 1
             semg_cols = 95  
        elif args.feature_name == 'chdiffsigimage':
             semg_rows = args.window             
             semg_cols = len(genIndex(args.num_semg_col*args.num_semg_row-1))-1
        elif args.feature_name == 'chdiff_sigimg_multistream':
             semg_rows = [args.window, args.window]
             semg_cols = train.shapes
        elif args.feature_name == 'featureimg_rawimg_multistream':
             semg_rows = [train.feature_dim, args.window]
             semg_cols = [args.num_semg_col*args.num_semg_row, args.num_semg_col*args.num_semg_row]
        elif args.feature_name == 'ch_multistream':
             semg_rows = []             
             for i in range(len(train.shapes)):
                   semg_rows.append(1)             
             semg_cols = train.shapes
        elif args.feature_name == 'block_multistream':
             semg_rows = []             
             for i in range(len(train.shapes)):
                   semg_rows.append(args.window)             
             semg_cols = train.shapes     
        elif args.feature_name == 'block_multistream_v2':
             semg_cols = []             
             for i in range(len(train.shapes)):
                   semg_cols.append(args.window)             
             semg_rows = train.shapes 
        elif args.feature_name == 'ch_multistream_plus_rawimg':
             semg_rows = train.shapes1
             semg_cols = train.shapes2
        elif args.feature_name == 'sigimg_rawimg_multistream':
             semg_rows = [args.window, args.window]
             semg_cols = train.shapes
        elif args.feature_name == 'frame_multistream': 
             semg_rows = []             
             for i in range(len(train.shapes)):
                   semg_rows.append(1)             
             semg_cols = train.shapes
        elif args.feature_name == 'simplestackedimg':
             semg_rows = args.window             
             semg_cols = 50 
        elif args.feature_name == 'rawfeatureimg':
             semg_rows = train.shape1             
             semg_cols = 10  
        elif args.feature_name == 'rawsemg_feature_singlestream':
             semg_rows = train.shape1             
             semg_cols = 10    
        elif args.feature_name == 'rawsemg_feature_multisource':
             semg_rows = train.shapes             
             semg_cols = [args.num_semg_col*args.num_semg_row, args.num_semg_col*args.num_semg_row] 
        elif args.feature_name == 'singleframe_multistream':
             semg_rows = [args.num_semg_row, args.num_semg_row]             
             semg_cols = [args.num_semg_col, args.num_semg_col] 
        elif args.feature_name == 'chwise_feature_multistream':
             semg_rows = []             
             for i in range(len(train.shapes)):
                   semg_rows.append(1)             
             semg_cols = train.shapes
        elif args.feature_name == 'chwise_feature_rawimg_multistream':
             semg_rows = []             
             for i in range(len(train.shapes)):
                   semg_rows.append(1)             
             semg_cols = train.shapes     
        else:
             semg_rows = 1
             semg_cols = 1             
       
        
        if type(train.num_channel) is types.ListType :
             print 'Use multi-stream CNN!'
             logger.info('Use multi-stream CNN')
             mod = Module_multistream.parse(
                args.module,
                revgrad=args.revgrad,
                num_revgrad_batch=args.num_revgrad_batch,
                tzeng=args.tzeng,
                num_tzeng_batch=args.num_tzeng_batch,
                num_gesture=train.num_gesture,
                num_subject=train.num_subject,
                subject_loss_weight=args.subject_loss_weight,
                lambda_scale=args.lambda_scale,
                adabn=args.adabn,
                num_adabn_epoch=args.num_adabn_epoch,
                random_scale=args.random_scale,
                multi_stream=True,
                num_stream=len(train.num_channel),
                lstm=args.lstm,
                num_lstm_hidden=args.num_lstm_hidden,
                num_lstm_layer=args.num_lstm_layer,
                for_training=True,
                faug=args.faug,
                faug_classwise=args.faug_classwise,
                num_eval_epoch=args.num_eval_epoch,
                snapshot_period=args.snapshot_period,
                pixel_same_init=args.pixel_same_init,
                symbol_kargs=dict(
                    num_semg_row=semg_rows,
                    num_semg_col=semg_cols,
                    num_filter=args.num_filter,
                    num_pixel=args.num_pixel,
                    num_feature_block=args.num_feature_block,
                    num_gesture_block=args.num_gesture_block,
                    num_subject_block=args.num_subject_block,
                    num_hidden=args.num_hidden,
                    num_bottleneck=args.num_bottleneck,
                    dropout=args.dropout,
                    num_channel=train.num_channel,
                    num_presnet=args.num_presnet,
                    presnet_branch=args.presnet_branch,
                    drop_presnet=args.drop_presnet,
                    bng=args.bng,
                    fusion_type = args.fusion_type,
                    subject_confusion_loss_weight=args.subject_confusion_loss_weight,
                    minibatch=args.minibatch,
                    confuse_conv=args.confuse_conv,
                    confuse_all=args.confuse_all,
                    subject_wd=args.subject_wd,
                    drop_branch=args.drop_branch,
                    pool=args.pool,
                    zscore=args.zscore,
                    zscore_bng=args.zscore_bng,
                    num_stream=len(train.num_channel),
                    lstm_last=args.lstm_last,
                    lstm_dropout=args.lstm_dropout,
                    lstm_shortcut=args.lstm_shortcut,
                    lstm_bn=args.lstm_bn,
                    lstm_window=args.lstm_window,
                    lstm_grad_scale=args.lstm_grad_scale,
                    drop_conv=args.drop_conv,
                    drop_presnet_branch=args.drop_presnet_branch,
                    drop_presnet_proj=args.drop_presnet_proj,
                    presnet_proj_type=args.presnet_proj_type,
                    presnet_promote=args.presnet_promote,
                    pixel_reduce_loss_weight=args.pixel_reduce_loss_weight,
                    pixel_reduce_bias=args.pixel_reduce_bias,
                    pixel_reduce_kernel=args.pixel_reduce_kernel,
                    pixel_reduce_stride=args.pixel_reduce_stride,
                    pixel_reduce_pad=args.pixel_reduce_pad,
                    pixel_reduce_norm=args.pixel_reduce_norm,
                    pixel_reduce_reg_out=args.pixel_reduce_reg_out,
                    num_pixel_reduce_filter=args.num_pixel_reduce_filter,
                    fast_pixel_reduce=args.fast_pixel_reduce,
                    drop_pixel=args.drop_pixel,
                    num_conv=args.num_conv,
                    presnet_dense=args.presnet_dense,
                    conv_shortcut=args.conv_shortcut
                ),
                context=[mx.gpu(i) for i in args.gpu]
            )
        else:
            print ("semg rows  =  %d, semg cols = %d" % (semg_rows, semg_cols))
            mod = Module.parse(
                args.module,
                revgrad=args.revgrad,
                num_revgrad_batch=args.num_revgrad_batch,
                tzeng=args.tzeng,
                num_tzeng_batch=args.num_tzeng_batch,
                num_gesture=train.num_gesture,
                num_subject=train.num_subject,
                subject_loss_weight=args.subject_loss_weight,
                lambda_scale=args.lambda_scale,
                adabn=args.adabn,
                num_adabn_epoch=args.num_adabn_epoch,
                random_scale=args.random_scale,
                dual_stream=args.dual_stream,
                lstm=args.lstm,
                num_lstm_hidden=args.num_lstm_hidden,
                num_lstm_layer=args.num_lstm_layer,
                for_training=True,
                faug=args.faug,
                faug_classwise=args.faug_classwise,
                num_eval_epoch=args.num_eval_epoch,
                snapshot_period=args.snapshot_period,
                pixel_same_init=args.pixel_same_init,
                symbol_kargs=dict(
                    num_semg_row=semg_rows,
                    num_semg_col=semg_cols,
                    num_filter=args.num_filter,
                    num_pixel=args.num_pixel,
                    num_feature_block=args.num_feature_block,
                    num_gesture_block=args.num_gesture_block,
                    num_subject_block=args.num_subject_block,
                    num_hidden=args.num_hidden,
                    num_bottleneck=args.num_bottleneck,
                    dropout=args.dropout,
                    num_channel=train.num_channel // (args.lstm_window or 1),
                    num_presnet=args.num_presnet,
                    presnet_branch=args.presnet_branch,
                    drop_presnet=args.drop_presnet,
                    bng=args.bng,
                    subject_confusion_loss_weight=args.subject_confusion_loss_weight,
                    minibatch=args.minibatch,
                    confuse_conv=args.confuse_conv,
                    confuse_all=args.confuse_all,
                    subject_wd=args.subject_wd,
                    drop_branch=args.drop_branch,
                    pool=args.pool,
                    zscore=args.zscore,
                    zscore_bng=args.zscore_bng,
                    num_stream=2 if args.dual_stream else 1,
                    lstm_last=args.lstm_last,
                    lstm_dropout=args.lstm_dropout,
                    lstm_shortcut=args.lstm_shortcut,
                    lstm_bn=args.lstm_bn,
                    lstm_window=args.lstm_window,
                    lstm_grad_scale=args.lstm_grad_scale,
                    drop_conv=args.drop_conv,
                    drop_presnet_branch=args.drop_presnet_branch,
                    drop_presnet_proj=args.drop_presnet_proj,
                    presnet_proj_type=args.presnet_proj_type,
                    presnet_promote=args.presnet_promote,
                    pixel_reduce_loss_weight=args.pixel_reduce_loss_weight,
                    pixel_reduce_bias=args.pixel_reduce_bias,
                    pixel_reduce_kernel=args.pixel_reduce_kernel,
                    pixel_reduce_stride=args.pixel_reduce_stride,
                    pixel_reduce_pad=args.pixel_reduce_pad,
                    pixel_reduce_norm=args.pixel_reduce_norm,
                    pixel_reduce_reg_out=args.pixel_reduce_reg_out,
                    num_pixel_reduce_filter=args.num_pixel_reduce_filter,
                    fast_pixel_reduce=args.fast_pixel_reduce,
                    drop_pixel=args.drop_pixel,
                    num_conv=args.num_conv,
                    presnet_dense=args.presnet_dense,
                    conv_shortcut=args.conv_shortcut
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
            gamma=args.gamma,
            snapshot=args.snapshot,
            params=args.params,
            ignore_params=args.ignore_params,
            fix_params=args.fix_params,
            decay_all=args.decay_all
        )


if __name__ == '__main__':
    cli(obj=Bunch())
