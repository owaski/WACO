import os
import subprocess
from tqdm import tqdm

SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

def get_transcripts(path):
    transcripts = {}
    with open(path) as f:
        for line in f:
            file, transcript = line.strip().split(' ',maxsplit=1)
            transcripts[file] = transcript
    return transcripts


def convert_flac(path, out_path):
    subprocess.call(['sox', path, out_path])


output_directory = None 
librispeech_root = None

for split in SPLITS:
    librispeech_directory = os.path.join(librispeech_root, split)
    for speaker in tqdm(os.listdir(librispeech_directory), desc=split):
        speaker_dir = os.path.join(librispeech_directory, speaker)
        out_speaker_dir = os.path.join(output_directory, speaker)
        os.makedirs(out_speaker_dir, exist_ok=True)
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            transcript_path = os.path.join(chapter_dir, '{}-{}.trans.txt'.format(speaker, chapter))
            transcripts = get_transcripts(transcript_path)
            # print(transcripts)

            for k, v in transcripts.items():
                ts_path = os.path.join(out_speaker_dir, k + '.lab')
                with open(ts_path, 'w', encoding='utf8') as f:
                    f.write(v)

            sound_files = os.listdir(chapter_dir)
            for sf in sound_files:
                if not sf.endswith('.flac'):
                    continue
                sf_path = os.path.join(chapter_dir, sf)
                sf_out_path = os.path.join(out_speaker_dir, sf.replace('.flac', '.wav'))
                convert_flac(sf_path, sf_out_path)
