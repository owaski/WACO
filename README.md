# WACO: Word-Aligned Contrastive Learning for Speech Translation

\[[Project Page](https://owaski.github.io/WACO/)\] \[[Arxiv](https://arxiv.org/abs/2212.09359)\]

This repository contains the code for the paper **WACO: Word-Aligned Contrastive Learning for Speech Translation**. The repo is still under development.

## Training & Inference Instruction

### Requirements

```bash
git clone https://github.com/owaski/WACO
cd WACO

conda create -n waco python=3.8
conda activate waco
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
pip install -e ./
```

### Data Preparation

#### MuST-C English-German Data Preparation

First, download the raw data from https://ict.fbk.eu/must-c/, unzip the file and save files to path ```${DATA_PATH}```.
```bash
tar -zxf ${DATA_PATH}/MUSTC_v1.0_en-de.tar.gz -C ${DATA_PATH}
```
Then run the following script to generate the yaml configuration file, tsv file, sub-word model and dictionary. 
```bash
python WACO/prepare_data/prep_mustc_data.py --data-root ${DATA_PATH} --lang de --vocab-type unigram --vocab-size 10000
```
We also provide the prepared configs, tsvs and dictionaries [here]() for convenience. 


#### IWSLT23 Maltese-English Data Preparation

Please follow the instructions in jupyter notebook `WACO/prepare_data/prepare_mt_en.ipynb` to create Mt-En data from IWSLT23 and CommonVoice.
We also provide the processed data and tsvs [here](). Note the dictionary is from NLLB since we use NLLB as pre-trained MT checkpoint for Mt-En translation.

### Forced Alignment

First you need to install latest [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) from source and then follow the instructions in `WACO/scripts/produce_alignment.ipynb` to align the speech and transcript.

### Training

#### WACO Learning

```bash
bash WACO/train/waco/waco_de_10h.sh # for must-c en-de 10h asr
bash WACO/train/waco/waco_mt_10h.sh # for iwslt mt-en 10h asr
```

We also provide the pre-trained WACO checkpoints [here](https://www.dropbox.com/scl/fo/fj9ylr8up8x4cxidexq1p/h?rlkey=mv1reocfdxyi66paayrie07fr&dl=0) for convenience.

#### Multitask Finetuning

```bash
bash WACO/train/finetune/waco_de_10h_ft_1h.sh # for must-c en-de 10h asr
bash WACO/train/finetune/waco_mt_10h_ft_1h.sh # for iwslt mt-en 10h asr
```

### Evaluation

```bash
python fairseq_cli/generate.py /mnt/data7/siqiouyang/datasets/must-c-v1.0/ --gen-subset tst-COMMON_st_de --task speech_to_text \
--prefix-size 1 --max-tokens 4000000 --max-source-positions 4000000 --beam 10 --lenpen 0.6 --scoring sacrebleu \
--config-yaml config_st_de.yaml  --path /mnt/data7/siqiouyang/runs/WACO/waco_de_10h_ft_1h/checkpoint_best.pt

python fairseq_cli/generate.py /mnt/data7/siqiouyang/datasets/must-c-v1.0/ --gen-subset test_st_mt_en --task speech_to_text_nllb \
--prefix-size 1 --max-tokens 4000000 --max-source-positions 4000000 --beam 10 --lenpen 0.3 --scoring sacrebleu \
--config-yaml config_st_mt_en.yaml --path /mnt/data7/siqiouyang/runs/WACO/waco_mt_10h_ft_1h/checkpoint_best.pt
```