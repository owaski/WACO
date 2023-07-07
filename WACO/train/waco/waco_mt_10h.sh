#!/usr/bin/env bash

tag=waco_mt_10h

MODEL_DIR=/mnt/data7/siqiouyang/runs/WACO/$tag

mkdir -p ${MODEL_DIR}

export num_gpus=1

python train.py /mnt/data/siqiouyang/datasets/must-c-v1.0 \
    --distributed-world-size $num_gpus \
    --task speech_to_text_triplet_align_with_extra_mt_nllb \
    --train-subset train_asr_mt_iwslt,train_asr_mt_cv --valid-subset dev_asr_mt_iwslt,dev_asr_mt_cv \
    --config-yaml config_st_mt_en.yaml \
    --langpairs mt-en --lang-prefix-tok "eng_Latn" \
    --max-audio-positions 600000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-audio-tokens 1000000 --max-text-tokens 2000 --max-tokens 1000000  --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    \
    --arch xstnet_nllb_base --w2v2-model-path /mnt/data/siqiouyang/runs/mST/pretrained/xlsr2_300m.pt \
    --nllb-dir /mnt/data/siqiouyang/runs/ConST/pretrained/nllb \
    \
    --optimizer adam --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 25000  --weight-decay 0.0 \
    \--criterion multi_task_cross_entropy_with_contrastive_token_with_extra_MT \
    --label-smoothing 0.1 --ignore-prefix-size 1 --report-accuracy \
    --contrastive-weight 1.0 0.0 --contrastive-temperature 0.20 0.20 --contrastive-seqlen-type none --contrastive-level token \
    \
    --update-freq $(expr 16 / $num_gpus) --max-update 500000 \
    \
    --tensorboard-logdir tensorboard_logs/$tag --log-interval 100 \
    --save-interval-updates 1000 --save-interval 10 --validate-interval 10 \
    --keep-last-epochs 1 --keep-interval-updates 1 --keep-best-checkpoints 1 \
    --save-dir ${MODEL_DIR} \
    --ddp-backend=no_c10d --fp16 \
    --all-gather-list-size 32768 \
    --best-checkpoint-metric contrastive_loss \
    --reset-optimizer --reset-dataloader