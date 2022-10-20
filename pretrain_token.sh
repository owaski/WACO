#!/usr/bin/env bash

tag=pretrain_token
DATA_ROOT=$1
SAVE_DIR=$2
split_name=$3
from_tag=$4

TGT_LANG=de
pretrain_ckpt=wmt16_ende_xstnet_pretrain.pt
MODEL_DIR=$SAVE_DIR/$tag

mkdir -p ${MODEL_DIR}
cp $SAVE_DIR/$from_tag/checkpoint_best.pt $SAVE_DIR/$tag/checkpoint_last.pt

fairseq-train $DATA_ROOT \
    --distributed-world-size $num_gpus \
    --task speech_to_text_triplet_align_with_extra_mt \
    --train-subset $split_name --valid-subset dev_st_${TGT_LANG} \
    --config-yaml config_st_${TGT_LANG}.yaml \
    --langpairs en-${TGT_LANG} --lang-prefix-tok "<lang:${TGT_LANG}>" \
    --max-audio-positions 600000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-audio-tokens 1000000 --max-text-tokens 2000 --max-tokens 1000000  --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    \
    --arch xstnet_base --w2v2-model-path $SAVE_DIR/pretrained/wav2vec_small.pt \
    \
    --optimizer adam --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 25000  --weight-decay 0.0 \
    \
    --criterion multi_task_cross_entropy_with_contrastive_token_with_extra_MT \
    --label-smoothing 0.1 --ignore-prefix-size 1 --report-accuracy \
    --contrastive-weight 1.0 0.0 --contrastive-temperature 0.05 0.05 --contrastive-seqlen-type none --contrastive-level token \
    \
    --update-freq $(expr 16 / $num_gpus) --max-update 500000 \
    \
    --tensorboard-logdir tensorboard_logs/$tag --log-interval 100 \
    --save-interval-updates 1000 --save-interval 1 \
    --keep-last-epochs 1 --keep-interval-updates 1 --keep-best-checkpoints 1 \
    --save-dir ${MODEL_DIR} \
    --ddp-backend=no_c10d --fp16 \
    --reset-optimizer --reset-dataloader --all-gather-list-size 32768 \
    --best-checkpoint-metric contrastive_loss