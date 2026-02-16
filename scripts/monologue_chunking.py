################################################################
# Monologue Chunking Script
# -------------------------------------------------------------
# Since srts did not have precise timestamps for each sentence, we:
# 1. Use spacy to split text into sentencces
# 2. Prepare corpus folders for MFA (including validating them and updating a custom dictionary for OOVs)
# 3. Run MFA to get word-level alignments
# 4. Upgrade word-level alignments to sentence-level alignments based on spacy sentences
# 5. Chunk audio and video based on sentence-level alignments (with ffmpeg)
################################################################

from datetime import datetime
import logging
import re
import shutil
import spacy
import librosa
import argparse
import subprocess
import numpy as np
from pathlib import Path

from tqdm import tqdm

CONTRACTION_SUFFIXES = {"'ll","’ll","'re","’re","'ve","’ve","'d","’d","'m","’m","n't","n’t","'s","’s","'em","’em","'n","’n"}

################################################################
# Arg Parse
################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "folder_path",
    type=str,
    help="Path to the processed sensorium data folder",
)
parser.add_argument(
    '--custom_dict_path',
    type=str,
    default='./custom_oovs.dict',
)
parser.add_argument(
    '--force_redo',
    action='store_true',
    help="whether to remove existing folders in monologue folders and redo the chunking"
)
args = parser.parse_args()
args.folder_path = Path(args.folder_path)
assert args.folder_path.exists(), f"Folder path {args.folder_path} does not exist."

# Create custom dictionary path
args.custom_dict_path = Path(args.custom_dict_path)
if not args.custom_dict_path.exists():
    with open(args.custom_dict_path, "w") as f:
        pass

#
if args.force_redo:
    print(f"[NOTE] Running with --force_redo. Existing folders created using this script will be deleted")

################################################################
# logging
################################################################
logger = logging.getLogger("monologue_chunking")
start_datetime = datetime.now()
start_datetime_str = start_datetime.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f'monologue_chunking_{start_datetime_str}.log',
    level=logging.INFO,
)
logger.info(f"[START] Monologue chunking started at {start_datetime_str}")

################################################################
# Functions
################################################################
NUM_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
    "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety",
}

DECADE_MAP = {
    "10s": "tens", "20s": "twenties", "30s": "thirties", "40s": "forties",
    "50s": "fifties", "60s": "sixties", "70s": "seventies", "80s": "eighties", "90s": "nineties",
}

def normalize_token(tok: str) -> str:
    t = tok.strip().lower()

    t = re.compile(r"^[^\w]+|[^\w]+$").sub("", t)  # remove leading/trailing punctuation
    if not t:
        return ""
    
    if t in DECADE_MAP:
        return DECADE_MAP[t]

    if re.fullmatch(r"\d+", t):
        return NUM_MAP.get(t, "")  # unknown numbers -> drop (or keep t if you prefer)

    t = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", t)

    return t

def normalize_text(text: str) -> str:
    toks = re.split(r"\s+", text)
    cleaned = [normalize_token(x) for x in toks]
    cleaned = [x for x in cleaned if x]
    return " ".join(cleaned)

def simple_parse_textgrid(
    textgrid_lines : list[str]
):
    meta = {}
    word_intervals = []
    _state = 'start'
    for i, line in enumerate(textgrid_lines):
        if _state == 'start':
            if 'xmin =' in line:
                meta['xmin'] = float(line.split('=')[1].strip())
            if 'xmax =' in line:
                meta['xmax'] = float(line.split('=')[1].strip())
            if 'class' in line and line.strip().split('=')[-1].strip().strip('"') == 'IntervalTier':
                _state = 'interval_tier'
        elif _state == 'interval_tier':
            if 'name =' in line and line.strip().split('=')[-1].strip().strip('"') == 'words':
                _state = 'words_intervals'
            else:
                raise RuntimeError("Unexpected TextGrid format: Expected 'words' tier.")
        elif _state == 'words_intervals':
            if 'xmin =' in line or 'xmax =' in line:
                continue
            if 'intervals:' in line:
                split_line = line.strip().split(':')[1].strip()
                num_intervals = int(split_line.split(' = ')[-1].strip())
                break
    interval_idx = 0
    for j in range(i+1, len(textgrid_lines)-4, 4):
        if interval_idx >= num_intervals:
            break
        xmin_line = textgrid_lines[j+1].strip()
        xmax_line = textgrid_lines[j+2].strip()
        text_line = textgrid_lines[j+3].strip()
        xmin = float(xmin_line.split('=')[1].strip())
        xmax = float(xmax_line.split('=')[1].strip())
        text = text_line.split('=')[1].strip().strip('"')
        word_intervals.append({
            'start': xmin,
            'end': xmax,
            'text': text,
        })
        interval_idx += 1
        
    return word_intervals, meta

def sentence_to_words(
    sentence : str,
    nlp_spacy : spacy.language.Language,
):
    doc = (nlp_spacy(sentence))
    sentence_words = []
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.is_space or tok.is_punct:
            i += 1
            continue

        # merge previous token + contraction suffix
        if i + 1 < len(doc) and doc[i + 1].text in CONTRACTION_SUFFIXES:
            merged = tok.text + doc[i + 1].text
            sentence_words.append(merged.lower())
            i += 2
            continue
        
        # special cases for a.m. / p.m. and similar patterns where we want to separate the letters and ignore the dots
        if tok.text.lower() in ["a.m.", "p.m.", "a.m", "p.m"]:
            # breakpoint()
            for j, part in enumerate(tok.text.split('.')):
                if part:
                    sentence_words.append(part.lower())
            i += j
            continue
        sentence_words.append(tok.text.lower())
        i += 1
        
    # sentence_words = [token.text.lower() for token in doc if not token.is_space and not token.is_punct]
    return sentence_words

def simple_word_to_sentence_intervals(
    word_intervals: list[dict],
    meta: dict,
    sentences: list[str],
    nlp_spacy,
    verbosity: int = 1,
):
    def norm(w: str) -> str:
        return (w or "").strip().lower()

    sentence_intervals = []
    wi_idx = 0

    pbar = tqdm(
        enumerate(sentences),
        desc="Upgrading word-level to sentence-level intervals",
        disable=(verbosity < 1),
    )

    for sent_idx, sentence in pbar:
        sentence_words = [norm(w) for w in sentence_to_words(sentence, nlp_spacy)]
        sentence_words = [w for w in sentence_words if w != ""]  # safety

        if len(sentence_words) == 0:
            raise RuntimeError(f"Empty tokenized sentence at index {sent_idx}: '{sentence}'")

        # Greedy sequential match: consume words in order
        matched_indices = []
        cursor = wi_idx

        for w in sentence_words:
            found = False
            while cursor < len(word_intervals):
                wi_text = norm(word_intervals[cursor].get("text", ""))

                # skip silence/empty intervals
                if wi_text == "":
                    cursor += 1
                    continue

                if wi_text == w:
                    matched_indices.append(cursor)
                    cursor += 1
                    found = True
                    break

                cursor += 1

            if not found:
                # Debug info to help you inspect local region
                window = word_intervals[max(0, wi_idx-5):min(len(word_intervals), wi_idx+25)]
                window_str = " | ".join([norm(x.get("text","")) for x in window])
                print(f"Could not find match for word '{w}' in sentence {sent_idx}: '{sentence}'")
                print(f"Starting search at wi_idx={wi_idx}. Nearby word_intervals: {window_str}")
                breakpoint()

        start_wi_idx = matched_indices[0]
        end_wi_idx = matched_indices[-1]

        # Include trailing silences after the last matched word (your original behavior)
        while end_wi_idx + 1 < len(word_intervals) and norm(word_intervals[end_wi_idx + 1].get("text", "")) == "":
            end_wi_idx += 1

        sentence_intervals.append({
            "start": float(word_intervals[start_wi_idx]["start"]),
            "end": float(word_intervals[end_wi_idx]["end"]),
            "text": sentence,
        })

        # Advance global cursor to after what we consumed
        wi_idx = end_wi_idx + 1

    return sentence_intervals
        
def combine_short_sentences(
    sentences : list[str],
    nlp_spacy : spacy.language.Language,
    max_words : int = 3,
):
    if not sentences:
        return sentences
    combined_sentences = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = sentence_to_words(sentence, nlp_spacy)
        if len(sentence_words) <= max_words:
            # Combine with previous or next sentence (whichever is shorter)
            if combined_sentences and (i == len(sentences) - 1 or len(sentence_to_words(sentences[i + 1], nlp_spacy)) >= len(sentence_words)):
                combined_sentences[-1] += ' ' + sentence
            elif i < len(sentences) - 1:
                sentences[i + 1] = sentence + ' ' + sentences[i + 1]
            else:
                combined_sentences.append(sentence)
        else:
            combined_sentences.append(sentence)
        i += 1
    return combined_sentences
        
def mkdir_if_not_exists(path: Path, force_redo: bool = False):
    """
        Utility to create a dir, with option to force redo by deleting the existing dir
    """
    if force_redo:
        if path.exists():
            shutil.rmtree(path)
    try:
        path.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Directory {path} already exists. If you want to overwrite, please delete the directory or rerun with --force_redo flag.")
    
################################################################
# Models
################################################################
nlp_spacy = spacy.load("en_core_web_sm")

# Check if mfa acoustic and dictionary models are available
cmd = ['mfa', 'model', 'list', 'acoustic']
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if p.stderr.startswith('No models found'):
    raise RuntimeError("No MFA acoustic models found. Please ensure an acoustic and dictionary model are downloaded and installed.")
cmd = ['mfa', 'model', 'list', 'dictionary']
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if p.stderr.startswith('No models found'):
    raise RuntimeError("No MFA dictionary models found. Please ensure an acoustic and dictionary model are downloaded and installed.")
cmd = ['mfa', 'model', 'list', 'g2p']
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if p.stderr.startswith('No models found'):
    raise RuntimeError("No MFA g2p models found. Please ensure a g2p model is downloaded and installed for out of vocabulary words.")

################################################################
# 
# -------------------------------------------------------------
# processed/
# |  *monologue/
# |    |  *_Emotion.srt
# |    |  *_Neutral.srt
# |    |  *_Emotion.mp4
# |    |  *_Neutral.mp4
# |    |  *_Emotion.wav
# |    |  *_Neutral.wav
################################################################
folders = sorted(args.folder_path.glob("*"))
for folder in folders:
    if not folder.is_dir():
        continue
    if not folder.name.split("_")[1] == "monologue":
        continue
    
    #
    logger.info(f"[INFO] Processing folder: {folder}")
    
    # Get srt files ("*_Emotion.srt", "*_Neutral.srt")
    neutral_srt_files = list(folder.glob("*_Neutral.srt"))
    emotion_srt_files = list(folder.glob("*_Emotion.srt")) + list(folder.glob("*_Emotional.srt"))
    assert len(neutral_srt_files) == 1, f"Expected one Neutral srt file in {folder}, found {len(neutral_srt_files)}"
    assert len(emotion_srt_files) == 1, f"Expected one Emotion srt file in {folder}, found {len(emotion_srt_files)}"
    neutral_srt_file = neutral_srt_files[0]
    emotion_srt_file = emotion_srt_files[0]
    
    # Get mp4 files ("*_Emotion.mp4", "*_Neutral.mp4")
    neutral_mp4_files = list(folder.glob("*_Neutral.mp4"))
    emotion_mp4_files = list(folder.glob("*_Emotion.mp4")) + list(folder.glob("*_Emotional.mp4"))
    assert len(neutral_mp4_files) == 1, f"Expected one Neutral mp4 file in {folder}, found {len(neutral_mp4_files)}"
    assert len(emotion_mp4_files) == 1, f"Expected one Emotion mp4 file in {folder}, found {len(emotion_mp4_files)}"
    neutral_mp4_file = neutral_mp4_files[0]
    emotion_mp4_file = emotion_mp4_files[0]
    
    # Get wav files ("*_Emotion.wav", "*_Neutral.wav")
    neutral_wav_files = list(folder.glob("*_Neutral.wav"))
    emotion_wav_files = list(folder.glob("*_Emotion.wav")) + list(folder.glob("*_Emotional.wav"))
    assert len(neutral_wav_files) == 1, f"Expected one Neutral wav file in {folder}, found {len(neutral_wav_files)}"
    assert len(emotion_wav_files) == 1, f"Expected one Emotion wav file in {folder}, found {len(emotion_wav_files)}"
    neutral_wav_file = neutral_wav_files[0]
    emotion_wav_file = emotion_wav_files[0]
    
    # Load srt
    neutral_srt = []
    emotion_srt = []
    with open(neutral_srt_file, "r") as f:
        neutral_srt = f.readlines()
    with open(emotion_srt_file, "r") as f:
        emotion_srt = f.readlines()
    # breakpoint()
    
    # Check if srt files are either one long sentence or have proper timestamps
    num_timestamps = 0
    for line in neutral_srt:
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line.strip()):
            num_timestamps += 1
    if num_timestamps > 1:
        # Just get all text and ignore timestamps
        neutral_srt_list = []
        _state = "index"
        for line in neutral_srt:
            if _state == "index":
                _state = "timestamp"
                continue
            if _state == "timestamp":   
                if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line.strip()):
                    _state = "utterance"
                    continue
            if _state == "utterance":
                _temp_line = line.strip()
                if _temp_line == "":
                    _state = "index"
                    continue
                else:
                    neutral_srt_list.append(_temp_line)
        neutral_srt = ' '.join(neutral_srt_list).strip()
    else:
        neutral_srt = ' '.join(neutral_srt[2:]).strip()
        
    num_timestamps = 0
    for line in emotion_srt:
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line.strip()):
            num_timestamps += 1
    if num_timestamps > 1:
        # Just get all text and ignore timestamps
        emotion_srt_list = []
        _state = "index"
        for line in emotion_srt:
            if _state == "index":
                _state = "timestamp"
                continue
            if _state == "timestamp":   
                if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line.strip()):
                    _state = "utterance"
                    continue
            if _state == "utterance":
                _temp_line = line.strip()
                if _temp_line == "":
                    _state = "index"
                    continue
                else:
                    emotion_srt_list.append(_temp_line)
        emotion_srt = ' '.join(emotion_srt_list).strip()
    else:
        emotion_srt = ' '.join(emotion_srt[2:]).strip()
    
    # Process srt with spacy
    neutral_srt = re.sub(r'\s+', ' ', neutral_srt)
    emotion_srt = re.sub(r'\s+', ' ', emotion_srt)
    neutral_sentences = [s.text.strip() for s in nlp_spacy(neutral_srt).sents if s.text.strip()]
    emotion_sentences = [s.text.strip() for s in nlp_spacy(emotion_srt).sents if s.text.strip()]
    
    # Combine sentences that are too short (2 words or less) with the shorter of the neighboring sentences
    neutral_sentences = combine_short_sentences(neutral_sentences, nlp_spacy)
    emotion_sentences = combine_short_sentences(emotion_sentences, nlp_spacy)
    original_neutral_sentences = neutral_sentences.copy()
    original_emotion_sentences = emotion_sentences.copy()
    neutral_sentences = [normalize_text(s) for s in neutral_sentences]
    emotion_sentences = [normalize_text(s) for s in emotion_sentences]
    
    # Prep folders for word alignments (i.e. MFA), copy wav to there, and write text files
    neutral_corpus_folder = folder / "corpus_neutral"
    emotion_corpus_folder = folder / "corpus_emotion"
    mkdir_if_not_exists(neutral_corpus_folder, args.force_redo)
    mkdir_if_not_exists(emotion_corpus_folder, args.force_redo)
    
    shutil.copy(neutral_wav_file, neutral_corpus_folder / f"{folder.name}.wav")
    shutil.copy(emotion_wav_file, emotion_corpus_folder / f"{folder.name}.wav")
    with open(neutral_corpus_folder / f"{folder.name}.txt", "w") as f:
        for s in neutral_sentences:
            f.write(s + "\n")
    with open(emotion_corpus_folder / f"{folder.name}.txt", "w") as f:
        for s in emotion_sentences:
            f.write(s + "\n")
    
    # Load audio
    # neutral_audio, neutral_sr = librosa.load(neutral_wav_file, sr=None)
    # emotion_audio, emotion_sr = librosa.load(emotion_wav_file, sr=None)
    # if neutral_sr != emotion_sr:
    #     print(f"[WARNING] Sampling rates do not match in {folder}. Neutral: {neutral_sr}, Emotion: {emotion_sr}")
    # sr = neutral_sr
    
    # Validate the corpus folders
    neutral_alignment_folder = folder / "aligned_neutral"
    emotion_alignment_folder = folder / "aligned_emotion"
    mkdir_if_not_exists(neutral_alignment_folder, args.force_redo)
    mkdir_if_not_exists(emotion_alignment_folder, args.force_redo)
    cmd = ['mfa', 'validate', str(neutral_corpus_folder), str(args.custom_dict_path), 'english_us_arpa', '--single_speaker', '--clean', '--output_directory', str(neutral_corpus_folder)]
    subprocess.run(cmd)
    # breakpoint()
    cmd = ['mfa', 'validate', str(emotion_corpus_folder), str(args.custom_dict_path), 'english_us_arpa', '--single_speaker', '--clean', '--output_directory', str(emotion_corpus_folder)]
    subprocess.run(cmd)
    
    ################################################################
    # Check for OOVs and [4NOW] add them to custom dictionary if any
    ################################################################

    # Read oovs_found.txt
    neutral_oov_file = neutral_corpus_folder / "oovs_found_custom_oovs.txt"
    emotion_oov_file = emotion_corpus_folder / "oovs_found_custom_oovs.txt"
    neutral_oovs = []
    emotion_oovs = []
    if neutral_oov_file.exists():
        with open(neutral_oov_file, "r") as f:
            neutral_oovs = [line.strip() for line in f.readlines() if line.strip()]
        if neutral_oovs:
            cmd = ['mfa', 'g2p', str(neutral_oov_file), 'english_us_arpa', str(neutral_corpus_folder / "tmp_custom_dict.txt")]
            subprocess.run(cmd, check=True)
    if emotion_oov_file.exists():
        with open(emotion_oov_file, "r") as f:
            emotion_oovs = [line.strip() for line in f.readlines() if line.strip()]
        if emotion_oovs:
            cmd = ['mfa', 'g2p', str(emotion_oov_file), 'english_us_arpa', str(emotion_corpus_folder / "tmp_custom_dict.txt")]
            subprocess.run(cmd, check=True)
            
    # Append to custom dictionary (checking for duplicates)
    pronunciations = []
    with open(args.custom_dict_path, "r") as f:
        pronunciations = [line.strip() for line in f.readlines() if line.strip()]
    pronunciations = set(pronunciations)
    emotion_pronunciations = []
    neutral_pronunciations = []
    if (emotion_corpus_folder / "tmp_custom_dict.txt").exists():
        with open(emotion_corpus_folder / "tmp_custom_dict.txt", "r") as f:
            emotion_pronunciations = [line.strip() for line in f.readlines() if line.strip()]
    if (neutral_corpus_folder / "tmp_custom_dict.txt").exists():
        with open(neutral_corpus_folder / "tmp_custom_dict.txt", "r") as f:
            neutral_pronunciations = [line.strip() for line in f.readlines() if line.strip()]
    for p in emotion_pronunciations + neutral_pronunciations:
        pronunciations.add(p)
    if pronunciations:
        with open(args.custom_dict_path, "w") as f:
            for p in sorted(pronunciations):
                f.write(p + "\n")
                
    # Force Alignment with MFA
    cmd = [
        'mfa', 'align',
        str(neutral_corpus_folder),
        str(args.custom_dict_path),
        'english_us_arpa',
        str(neutral_alignment_folder),
        '--single_speaker',
        '--clean',
        '--beam', '100',
        '--retry_beam', '400',
    ]
    subprocess.run(cmd, check=True)
    cmd = [
        'mfa', 'align',
        str(emotion_corpus_folder),
        str(args.custom_dict_path),
        'english_us_arpa',
        str(emotion_alignment_folder),
        '--single_speaker',
        '--clean',
        '--beam', '100',
        '--retry_beam', '400',
    ]
    subprocess.run(cmd, check=True)
    
    # Load word-level alignments
    neutral_textgrid_file = neutral_alignment_folder / f"{folder.name}.TextGrid"
    emotion_textgrid_file = emotion_alignment_folder / f"{folder.name}.TextGrid"
    assert neutral_textgrid_file.exists(), f"Neutral TextGrid file not found: {neutral_textgrid_file}"
    assert emotion_textgrid_file.exists(), f"Emotion TextGrid file not found: {emotion_textgrid_file}"
    with open(neutral_textgrid_file, "r") as f:
        neutral_textgrid = f.readlines()
    with open(emotion_textgrid_file, "r") as f:
        emotion_textgrid = f.readlines()
        
    neutral_word_intervals, neutral_meta = simple_parse_textgrid(neutral_textgrid)
    emotion_word_intervals, emotion_meta = simple_parse_textgrid(emotion_textgrid)    
    
    # Upgrade word-level alignments to sentence-level alignments
    neutral_sentence_intervals = simple_word_to_sentence_intervals(
        neutral_word_intervals,
        neutral_meta,
        neutral_sentences,
        nlp_spacy
    )
    emotion_sentence_intervals = simple_word_to_sentence_intervals(
        emotion_word_intervals,
        emotion_meta,
        emotion_sentences,
        nlp_spacy
    )
    
    ################################################################
    # Chunk audio and video based on sentence-level alignments
    ################################################################
    neutral_monologue_sentences_folder = folder / "neutral_monologue_sentences"
    mkdir_if_not_exists(neutral_monologue_sentences_folder, args.force_redo)
    emotion_monologue_sentences_folder = folder / "emotion_monologue_sentences"
    mkdir_if_not_exists(emotion_monologue_sentences_folder, args.force_redo)
    
    neutral_monologue_sentences_csv = neutral_monologue_sentences_folder / "sentence_intervals.csv"
    emotion_monologue_sentences_csv = emotion_monologue_sentences_folder / "sentence_intervals.csv"
    with open(neutral_monologue_sentences_csv, "w") as f:
        f.write("sentence_idx,start_time,end_time,text\n")
    with open(emotion_monologue_sentences_csv, "w") as f:
        f.write("sentence_idx,start_time,end_time,text\n")
    for i, sentence_interval in enumerate(neutral_sentence_intervals):
        start_time = sentence_interval['start']
        end_time = sentence_interval['end']
        sentence_text = sentence_interval['text']
        
        # Update csv
        with open(neutral_monologue_sentences_csv, "a") as f:
            f.write(f"{i},{start_time:.3f},{end_time:.3f},\"{sentence_text}\"\n")
            
        # Chunk audio with ffmpeg
        neutral_output_wav = neutral_monologue_sentences_folder / f"sentence_{i:03d}.wav"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(neutral_wav_file),
            "-ss", f"{start_time:.3f}", "-to", f"{end_time:.3f}",
            "-vn",
            "-acodec", "pcm_s16le",
            str(neutral_output_wav),
        ]
        subprocess.run(cmd, check=True)
        
        # Chunk video with ffmpeg
        neutral_output_mp4 = neutral_monologue_sentences_folder / f"sentence_{i:03d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(neutral_mp4_file),
            "-ss", f"{start_time:.3f}", "-to", f"{end_time:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(neutral_output_mp4),
        ]
        subprocess.run(cmd, check=True)
        
    for i, sentence_interval in enumerate(emotion_sentence_intervals):
        start_time = sentence_interval['start']
        end_time = sentence_interval['end']
        sentence_text = sentence_interval['text']
        
        # Update csv
        with open(emotion_monologue_sentences_csv, "a") as f:
            f.write(f"{i},{start_time:.3f},{end_time:.3f},\"{sentence_text}\"\n")
            
        # Chunk audio with ffmpeg
        emotion_output_wav = emotion_monologue_sentences_folder / f"sentence_{i:03d}.wav"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(emotion_wav_file),
            "-ss", f"{start_time:.3f}", "-to", f"{end_time:.3f}",
            "-vn",
            "-acodec", "pcm_s16le",
            str(emotion_output_wav),
        ]
        subprocess.run(cmd, check=True)
        
        # Chunk video with ffmpeg
        emotion_output_mp4 = emotion_monologue_sentences_folder / f"sentence_{i:03d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(emotion_mp4_file),
            "-ss", f"{start_time:.3f}", "-to", f"{end_time:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(emotion_output_mp4),
        ]
        subprocess.run(cmd, check=True)
    
    # breakpoint()
    
finish_datetime = datetime.now()
finish_datetime_str = finish_datetime.strftime("%Y%m%d_%H%M%S")
logger.info(f"[END] Monologue chunking finished at {finish_datetime_str}")
total_elapsed = finish_datetime - start_datetime_str
logger.info(f"[END] Total elapsed time: {total_elapsed}")
logger.info(f"[FIN]")