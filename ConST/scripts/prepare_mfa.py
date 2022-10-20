import os
import random
import argparse
from string import punctuation
import wave

from tqdm import tqdm

import sentencepiece
import torchaudio
import textgrids

import numpy as np
import torch as th

from ConST.prepare_data.data_utils import load_df_from_tsv, save_df_to_tsv

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--lang', type=str)
args = parser.parse_args()

root = args.root
lang = args.lang

spm = sentencepiece.SentencePieceProcessor()
spm.Load(os.path.join(root, 'spm_unigram10000_st_{}.model'.format(lang)))
train_df = load_df_from_tsv(os.path.join(root, 'train_st_{}.tsv'.format(lang)))

save_dir = os.path.join(root, 'en-{}'.format(lang), 'data', 'train', 'align')
os.makedirs(save_dir, exist_ok=True)

last_audio_path = None

for idx in tqdm(range(len(train_df))):
    audio_path, offset, num_frames = os.path.join(root, train_df['audio'][idx]).split(':')
    offset, num_frames = int(offset), int(num_frames)
    if last_audio_path is None or audio_path != last_audio_path:
        waveform, frame_rate = torchaudio.load(os.path.join(root, audio_path))
        last_audio_path = audio_path
    torchaudio.save(os.path.join(save_dir, '{}.wav'.format(train_df['id'][idx])), waveform[:, num_frames : num_frames + offset], sample_rate=frame_rate)

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
    
for i, id in enumerate(tqdm(train_df['id'])):
    with open(os.path.join(save_dir, '{}.txt'.format(id)), 'w') as w:
        w.write(' '.join(tokenized_sentences[i]))