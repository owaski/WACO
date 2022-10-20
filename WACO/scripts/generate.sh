#!/usr/bin/env bash

split=tst-COMMON_st_de
name=ablation_pretrain_token_mfat_10h_noaudiopretrain_ft_1h

fairseq-generate $DATA_ROOT --gen-subset $split --task speech_to_text --prefix-size 1 \
--max-tokens 4000000 --max-source-positions 4000000 --beam 10 --lenpen 0.6 --scoring sacrebleu \
--config-yaml config_st_de.yaml  --path $MODEL_PATH \
--results-path $RESULT_PATH # \
# --mt-mode 