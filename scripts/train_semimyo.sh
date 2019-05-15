#!/usr/bin/env bash

# ver=20161127.1
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0 \
  # --num-glove-layer 2 --num-glove-hidden 128 \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0 \
    # --num-glove-layer 2 --num-glove-hidden 128 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v20161127.1/model-0028.params \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.2
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.01 \
  # --num-glove-layer 2 --num-glove-hidden 128 \
  # --gpu 1 \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.01 \
    # --num-glove-layer 2 --num-glove-hidden 128 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v20161127.2/model-0028.params \
    # --gpu 1 \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.3
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.1 \
  # --num-glove-layer 2 --num-glove-hidden 128 \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.1 \
    # --num-glove-layer 2 --num-glove-hidden 128 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v20161127.3/model-0028.params \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.4
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.01 \
  # --num-glove-layer 2 --num-glove-hidden 256 \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.01 \
    # --num-glove-layer 2 --num-glove-hidden 256 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v20161127.4/model-0028.params \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.5
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.01 \
  # --num-glove-layer 4 --num-glove-hidden 128 \
  # --gpu 1 \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.01 \
    # --num-glove-layer 4 --num-glove-hidden 128 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v20161127.5/model-0028.params \
    # --gpu 1 \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.6
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.1 \
  # --num-glove-layer 2 --num-glove-hidden 128 \
  # --gpu $@ \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.1 \
    # --num-glove-layer 2 --num-glove-hidden 128 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver/model-0028.params \
    # --gpu $@ \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.7
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.1 \
  # --num-glove-layer 4 --num-glove-hidden 128 \
  # --gpu $@ \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.1 \
    # --num-glove-layer 4 --num-glove-hidden 128 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver/model-0028.params \
    # --gpu $@ \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.8
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.1 \
  # --num-glove-layer 8 --num-glove-hidden 128 \
  # --gpu $@ \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.1 \
    # --num-glove-layer 8 --num-glove-hidden 128 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver/model-0028.params \
    # --gpu $@ \
    # --crossval-type one-fold-intra-subject --fold $i
# done

# ver=20161127.9
# scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
  # --log log --snapshot model \
  # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
  # --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  # --balance-gesture 1 \
  # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  # --glove-loss-weight 0.1 \
  # --num-glove-layer 2 --num-glove-hidden 64 \
  # --gpu $@ \
  # --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 26 | shuf); do
  # scripts/rundocker answeror/sigr:2016-11-27-cuda8 python -m sigr.train_semimyo crossval \
    # --log log --snapshot model \
    # --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
    # --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    # --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    # --balance-gesture 1 \
    # --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    # --glove-loss-weight 0.1 \
    # --num-glove-layer 2 --num-glove-hidden 64 \
    # --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver/model-0028.params \
    # --gpu $@ \
    # --crossval-type one-fold-intra-subject --fold $i
# done

ver=20161127.10
scripts/rundocker answeror/sigr:2016-11-27 python -m sigr.train_semimyo crossval \
  --log log --snapshot model \
  --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
  --root .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver \
  --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
  --balance-gesture 1 \
  --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
  --glove-loss-weight 0.1 \
  --num-glove-layer 2 --num-glove-hidden 256 \
  --gpu $@ \
  --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 26 | shuf); do
  scripts/rundocker answeror/sigr:2016-11-27 python -m sigr.train_semimyo crossval \
    --log log --snapshot model \
    --num-epoch 56 --lr-step 32 --lr-step 48 --snapshot-period 56 \
    --root .cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-$i-v$ver \
    --batch-size 1000 --dataset ninapro-db1-raw/semg-glove \
    --balance-gesture 1 \
    --preprocess '{ninapro-lowpass-parallel,identity,identity}' \
    --glove-loss-weight 0.1 \
    --num-glove-layer 2 --num-glove-hidden 256 \
    --params .cache/semimyo-ninapro-db1-raw-semg-glove-universal-one-fold-intra-subject-v$ver/model-0028.params \
    --gpu $@ \
    --crossval-type one-fold-intra-subject --fold $i
done
