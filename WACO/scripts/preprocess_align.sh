conda activate st
python prepare_mfa.py --data-root $DATA_ROOT --lang de

conda activate mfa

mfa models download acoustic english_mfa
mfa models download dictionary english_mfa

mfa align $DATA_ROOT/en-de/data/train/align english_mfa english_mfa $DATA_ROOT/en-de/data/train/align/textgrids/

conda activate st
python finish_mfa.py --data-root $DATA_ROOT --lang de