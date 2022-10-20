# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from data_utils import (
    save_df_to_tsv
)
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "speaker", "src_lang"]


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = LIBRISPEECH(out_root.as_posix(), url=split, download=True) 

    # Generate TSV manifest
    print("Generating manifest...")
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = LIBRISPEECH(out_root.as_posix(), url=split)
        for _, _, utt, spk_id, chapter_no, utt_no in tqdm(dataset, desc=split):
            sample_id = "{}-{}-{:04d}".format(spk_id, chapter_no, utt_no)
            audio_path = os.path.join('LibriSpeech/{}/{}/{}'.format(split, spk_id, chapter_no), sample_id + '.flac')

            info = torchaudio.info(os.path.join(out_root, audio_path))

            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_path)
            manifest["n_frames"].append(info.num_frames)
            manifest["src_text"].append(utt.lower())
            manifest["src_lang"].append('en')
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()