import os
import random
import argparse
from string import punctuation

from tqdm import tqdm

import sentencepiece
import torchaudio
import textgrids

import numpy as np
import torch as th

from ConST.prepare_data.data_utils import load_df_from_tsv, save_df_to_tsv

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str)
parser.add_argument('--lang', type=str)
args = parser.parse_args()

root = args.root
lang = args.lang

spm = sentencepiece.SentencePieceProcessor()
spm.Load(os.path.join(root, 'spm_unigram10000_st.model'))
train_df = load_df_from_tsv(os.path.join(root, 'train_st.tsv'))

save_dir = os.path.join(root, 'en-{}'.format(lang), 'data', 'train', 'align')
os.makedirs(save_dir, exist_ok=True)

sentences = train_df['src_text'].tolist()

def covered(s, punctuation):
    for c in s:
        if c not in punctuation:
            return False
    return True

space = '▁'
tokenized_sentences = []
segmentss = []
punctuation = punctuation + '—’'
for sent in tqdm(train_df['src_text'].tolist()):
    tokens = spm.EncodeAsPieces(sent)
    segments = []
    last = -1
    for idx, token in enumerate(tokens):
        if token.startswith(space) or covered(token, punctuation):
            if last != -1 and last <= idx - 1:
                segments.append((last, idx - 1))
            last = idx + (token == space or covered(token, punctuation) or \
                (token.startswith(space) and len(token) > 1 and covered(token[1:], punctuation)))    
    
    if last < len(tokens):
        segments.append((last, len(tokens) - 1))

    tokenized_sentence = []
    for seg in segments:
        token = ''.join(tokens[seg[0] : seg[1] + 1]).replace(space, '')
        if token.replace(',', '').isnumeric():
            token = token.replace(',', '')
        tokenized_sentence.append(token)

    tokenized_sentences.append(tokenized_sentence)
    segmentss.append(segments)
    
filtered_grids = []
n_outlier = 0
for i, id in enumerate(tqdm(train_df['id'])):
    grid_path = os.path.join(save_dir, 'textgrids/{}.TextGrid'.format(id))
    if os.path.exists(grid_path):
        grid = textgrids.TextGrid(grid_path)
        filtered_grid = [tok for tok in grid['words'] if tok.text != '']

        if len(filtered_grid) != len(tokenized_sentences[i]):
            print(i, [w.text for w in filtered_grid], tokenized_sentences[i], sep='\n')
            n_outlier += 1
            continue

        interval = np.array([(word.xmin, word.xmax) for word in filtered_grid])
        audio_path = os.path.join(save_dir, '{}.wav'.format(id))
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        interval = interval / duration

        th.save([segmentss[i], interval], os.path.join(save_dir, '{}.pt'.format(id)))