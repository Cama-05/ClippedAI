"""Main script to process videos and create YouTube Shorts using AI-powered transcription and clip detection."""

import argparse
import os
import pickle
import re
import subprocess
import sys
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
try:
    from omegaconf.nodes import AnyNode
    torch.serialization.add_safe_globals([AnyNode])
except Exception:
    pass

import nltk
from clipsai import Transcriber, ClipFinder, resize, MediaEditor, AudioVideoFile
from clipsai.clip.clip import Clip
from dotenv import load_dotenv

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", message="Model was trained with pyannote.audio")
warnings.filterwarnings("ignore", message="Model was trained with torch")
warnings.filterwarnings("ignore", message="Lightning automatically upgraded")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated")

# Suppress unnecessary warnings via environment variables
os.environ['FFREPORT'] = 'file=ffmpeg.log:level=32'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# --- Directories ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
FONT_DIR = "fonts" 

# --- Hugging Face ---
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your_huggingface_token_here")

# --- Clip Duration ---
MIN_CLIP_DURATION = int(os.getenv("MIN_CLIP_DURATION", "45"))
MAX_CLIP_DURATION = int(os.getenv("MAX_CLIP_DURATION", "120"))

# --- Transcription ---
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "large-v1")

# --- Resizing ---
ASPECT_RATIO_WIDTH = int(os.getenv("ASPECT_RATIO_WIDTH", "9"))
ASPECT_RATIO_HEIGHT = int(os.getenv("ASPECT_RATIO_HEIGHT", "16"))

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

# --- Subtitles ---
SUBTITLE_FONT = "Montserrat Extra Bold"
SUBTITLE_FONT_SIZE = 80
SUBTITLE_ALIGNMENT = 8  # Top center
SUBTITLE_MARGIN_V = 120
SUBTITLE_STYLES = {
    "Default": {
        "Fontname": SUBTITLE_FONT,
        "Fontsize": SUBTITLE_FONT_SIZE,
        "PrimaryColour": "&H00FFFFFF",  # White
        "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H40000000",
        "BackColour": "&HFF000000",
        "Bold": -1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 100,
        "ScaleY": 100,
        "Spacing": 2,
        "Angle": 0,
        "BorderStyle": 1,
        "Outline": 15,
        "Shadow": 0,
        "Alignment": SUBTITLE_ALIGNMENT,
        "MarginL": 30,
        "MarginR": 30,
        "MarginV": SUBTITLE_MARGIN_V,
        "Encoding": 1,
    },
    "Yellow": {
        "Fontname": SUBTITLE_FONT,
        "Fontsize": SUBTITLE_FONT_SIZE,
        "PrimaryColour": "&H0000FFFF",  # Yellow
        "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H40000000",
        "BackColour": "&HFF000000",
        "Bold": -1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 100,
        "ScaleY": 100,
        "Spacing": 2,
        "Angle": 0,
        "BorderStyle": 1,
        "Outline": 15,
        "Shadow": 0,
        "Alignment": SUBTITLE_ALIGNMENT,
        "MarginL": 30,
        "MarginR": 30,
        "MarginV": SUBTITLE_MARGIN_V,
        "Encoding": 1,
    },
}

def get_transcription_file_path(input_path: str) -> str:
    """Generate the transcription file path based on input video path."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(INPUT_DIR, f"{base_name}_transcription.pkl")

def load_existing_transcription(transcription_path: str):
    """Load existing transcription if it exists."""
    if os.path.exists(transcription_path):
        print(f"Found existing transcription: {transcription_path}")
        try:
            with open(transcription_path, "rb") as f:
                transcription = pickle.load(f)
            print("Successfully loaded existing transcription!")
            return transcription
        except Exception as e:
            print(f"Error loading existing transcription: {e}")
            return None
    return None

def save_transcription(transcription, transcription_path: str):
    """Save transcription to file."""
    try:
        with open(transcription_path, "wb") as f:
            pickle.dump(transcription, f)
        print(f"Transcription saved to: {transcription_path}")
    except Exception as e:
        print(f"Error saving transcription: {e}")

def ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def safe_filename(s: str) -> str:
    """Remove characters not allowed in filenames, but keep spaces, punctuation, and emojis."""
    # Define a set of characters that are generally safe for filenames
    # This includes alphanumeric, spaces, and some common punctuation
    safe_chars = string.ascii_letters + string.digits + " -_."
    # Add common punctuation that might be part of a title but needs careful handling
    safe_chars += "!?,:;@#$%^&+=[]{}"
    # Add a range of common emojis. This list can be expanded if needed.
    # Using a broad range to cover most common emojis without making the string excessively long.
    emoji_chars = "".join(chr(i) for i in range(0x1F600, 0x1F64F)) + \
                  "".join(chr(i) for i in range(0x1F300, 0x1F5FF)) + \
                  "".join(chr(i) for i in range(0x1F900, 0x1F9FF)) + \
                  "".join(chr(i) for i in range(0x1FA70, 0x1FAFF))
    
    valid_chars = safe_chars + emoji_chars
    return ''.join(c for c in s if c in valid_chars)


def get_font_path(font_name: str) -> str:
    """Get the path to a font file in the font directory."""
    # Try different extensions for the font file
    for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
        font_path = os.path.join("fonts", f"{font_name}{ext}")
        if os.path.exists(font_path):
            return font_path
    # If not found with extension, return as is (for system fonts)
    return os.path.join("fonts", font_name)

def trim_video_ffmpeg(input_path: str, start_time: float, end_time: float, output_path: str) -> str:
    """
    Trim video using FFmpeg with full re-encode to fix A/V desync and VFR stuttering.
    YouTube downloads are Variable Frame Rate (VFR); stream-copy trims preserve the
    original timestamps causing progressive audio/video drift.  Re-encoding to a
    constant 30 fps resets all timestamps and eliminates both issues.
    """
    duration = end_time - start_time
    cmd = [
        'ffmpeg',
        '-ss', f'{start_time:.3f}',   # fast keyframe seek before input
        '-i', input_path,
        '-t', f'{duration:.3f}',
        '-c:v', 'libx264',
        '-preset', 'slow',            # quality over speed (user preference)
        '-crf', '18',                 # high quality (0=lossless, 23=default)
        '-r', '30',                   # force constant 30 fps → eliminates stuttering
        '-c:a', 'aac',
        '-b:a', '192k',
        '-avoid_negative_ts', 'make_zero',   # reset timestamps to 0
        '-movflags', '+faststart',    # progressive web playback
        '-y',
        output_path
    ]
    print(f'Trimming clip ({start_time:.2f}s – {end_time:.2f}s) with re-encode → {os.path.basename(output_path)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg trim failed:\n{result.stderr}')
    return output_path


def convert_to_vertical_ffmpeg(input_path: str, output_path: str) -> str:
    """
    Force a true 9:16 output (1080x1920) without pyannote.
    Uses a blurred background + centered foreground to preserve subject visibility.
    """
    filter_graph = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:10[bg];"
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-filter_complex', filter_graph,
        '-map', '[v]',
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',
        '-r', '30',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-y',
        output_path,
    ]
    print(f'Forcing 9:16 output with fallback conversion -> {os.path.basename(output_path)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg 9:16 conversion failed:\n{result.stderr}')
    return output_path


def get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Return (width, height) for a video file using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0:s=x',
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffprobe failed:\n{result.stderr}')
    width_str, height_str = result.stdout.strip().split('x')
    return int(width_str), int(height_str)

def transcribe_with_progress(audio_file_path, transcriber):
    """Transcribe with progress tracking"""
    print('Transcribing video...')
    
    # Get video duration for progress calculation
    try:
        probe_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_file_path]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        print(f"Video duration: {duration:.2f} seconds")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        duration = 0
        print(f"Could not determine video duration for progress tracking: {e}")
    
    # Custom progress callback
    def progress_callback(current_time):
        if duration > 0:
            progress = (current_time / duration) * 100
            print(f"Transcription progress: {progress:.1f}% ({current_time:.1f}s / {duration:.1f}s)") 
        else:
            print(f"Transcription progress: {current_time:.1f}s processed")
    
    # For now, we'll use a simple approach since clipsai doesn't expose progress directly
    # You can enhance this by modifying the clipsai library or using a different approach
    print("Starting transcription (progress updates may be limited)...")
    transcription = transcriber.transcribe(audio_file_path=audio_file_path, iso6391_lang_code='en')
    print("Transcription completed!")
    return transcription

def create_animated_subtitles(video_path, transcription, clip, output_path):
    """
    Create clean, bold subtitles matching the provided style: white bold for text, yellow bold for numbers/currency, no effects, TOP CENTER.
    """
    print('Creating styled subtitles...')
    
    # Get word info for the clip
    word_info = [w for w in transcription.get_word_info() if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time]
    if not word_info:
        print('No word-level transcript found for the clip. Skipping subtitles.')
        return video_path
    
    # Build cues: group words into phrases of max 25 chars
    cues = []
    current_cue = {
        'words': [],
        'start_time': None,
        'end_time': None
    }
    
    for w in word_info:
        word = w["word"]
        start_time = w["start_time"] - clip.start_time
        end_time = w["end_time"] - clip.start_time  # Fixed: should be clip.start_time, not clip.end_time
        
        should_start_new = False
        if current_cue['start_time'] is None:
            should_start_new = True
        elif len(' '.join(current_cue['words']) + ' ' + word) > 25:
            should_start_new = True
        elif start_time - current_cue['end_time'] > 0.5:
            should_start_new = True
        
        if should_start_new:
            if current_cue['words']:
                cues.append({
                    'start': current_cue['start_time'],
                    'end': current_cue['end_time'],
                    'text': ' '.join(current_cue['words'])
                })
            current_cue = {
                'words': [word],
                'start_time': start_time,
                'end_time': end_time
            }
        else:
            current_cue['words'].append(word)
            current_cue['end_time'] = end_time
    if current_cue['words']:
        cues.append({
            'start': current_cue['start_time'],
            'end': current_cue['end_time'],
            'text': ' '.join(current_cue['words'])
        })
    
    # Determine font used and print to console
    font_used = "Montserrat Extra Bold"
    print(f"Subtitles will use font: {font_used}")
    print("NOTE: Ensure 'Montserrat Extra Bold' font is installed or is available in the fonts directory.")

    # Write ASS subtitle file with clean, bold styling at the TOP CENTER
    ass_file = os.path.abspath(os.path.join(OUTPUT_DIR, 'temp_subtitles.ass')).replace('\\', '/')
    with open(ass_file, 'w', encoding='utf-8') as f:
        f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat Extra Bold,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Yellow,Montserrat Extra Bold,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback,Arial Rounded MT Bold,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: FallbackYellow,Arial Rounded MT Bold,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback2,Arial Black,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback2Yellow,Arial Black,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for cue in cues:
            start = ass_time(cue['start'])
            end = ass_time(cue['end'])
            words = cue['text'].split()
            line = ''
            for w in words:
                if any(char.isdigit() for char in w) or '$' in w or (',' in w and w.replace(',', '').isdigit()):
                    line += f'{{\\style Yellow}}{w} '
                else:
                    line += f'{w} '
            line = line.strip()
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line}\n")
    
    final_output = output_path.replace('.mp4', '_with_subtitles.mp4')
    # Escape drive-letter colon for FFmpeg filter parser on Windows (e.g., C\:/...)
    ass_filter_path = ass_file.replace(':', '\\:').replace("'", r"\'")

    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"ass=filename='{ass_filter_path}'",
        '-c:v', 'libx264',
        '-preset', 'slow',    # quality over speed
        '-crf', '16',         # slightly tighter than trim (compensates for double-encode loss)
        '-r', '30',           # enforce CFR in case resize produced VFR
        '-c:a', 'copy',
        '-movflags', '+faststart',
        '-y',
        final_output
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        os.remove(ass_file)
        print(f'Styled subtitles added successfully!')
        return final_output
    except subprocess.CalledProcessError as e:
        print(f'Error adding subtitles: {e}')
        print(f'FFmpeg stderr: {e.stderr.decode()}')
        print(f'FFmpeg stdout: {e.stdout.decode()}')
        return video_path

def get_viral_title(transcript_text, groq_api_key):
    import requests
    # Limit examples to avoid too long prompt
    examples = [
        "She was almost dead 😵", "He made $1,000,000 in 1 hour 💸", "This changed everything... 😲", 
        "They couldn't believe what happened! 😱", "He risked it all for this 😬"
    ]
    prompt = (
        "Given the following transcript, generate a catchy, viral YouTube Shorts title (max 7 words). "
        "ALWAYS include an emoji in the title. ONLY output the title, nothing else. Do NOT use hashtags. "
        "Do NOT explain, do NOT repeat the prompt, do NOT add quotes. The title should be in the style of these examples: "
        + ", ".join(examples) + ".\n\nTranscript:\n" + transcript_text
    )
    headers = {
        'Authorization': f'Bearer {groq_api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 30,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        # Just return the first line of the response as the title, and filter out any lines that look like explanations or quotes
        content = result['choices'][0]['message']['content']
        lines = [l.strip('"') for l in content.strip().split('\n') if l.strip() and not l.lower().startswith('here') and not l.lower().startswith('title:')]
        title = lines[0] if lines else "Untitled Clip"
        return title
    except requests.exceptions.HTTPError as e:
        print(f"Error with Groq API: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        return "Untitled Clip"
    except Exception as e:
        print(f"Unexpected error with Groq API: {e}")
        return "Untitled Clip"

def download_youtube_video(url: str, output_dir: str = "input") -> str:
    """
    Download a YouTube video at 1080p quality using yt-dlp.
    Returns the path to the downloaded .mp4 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    # Download best 1080p video merged with best audio into mp4
    # --restrict-filenames keeps the filename ASCII-only (avoids libmagic UnicodeDecodeError on Windows)
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--restrict-filenames",
        "--output", output_template,
        "--print", "after_move:filepath",
        url,
    ]
    print(f"Downloading YouTube video at 1080p: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # The last printed line is the final filepath
    filepath = result.stdout.strip().splitlines()[-1].strip()
    if not filepath or not os.path.exists(filepath):
        # Fallback: find newest mp4 in output_dir
        mp4s = sorted(Path(output_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise FileNotFoundError("yt-dlp completed but no mp4 found in input/")
        filepath = str(mp4s[0])

    print(f"Video downloaded: {filepath}")
    return filepath


def get_youtube_transcript(url: str):
    """
    Fetch the YouTube transcript for the given URL using youtube-transcript-api.
    Returns a list of segment dicts with keys: text, start, duration.
    Prefers manual transcripts over auto-generated ones.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Cannot extract video ID from URL: {url}")
    video_id = match.group(1)

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    all_transcripts = list(transcript_list)

    if not all_transcripts:
        raise RuntimeError("No transcripts available for this video.")

    manual = [t for t in all_transcripts if not t.is_generated]
    generated = [t for t in all_transcripts if t.is_generated]
    chosen = (manual or generated)[0]

    label = "manual" if not chosen.is_generated else "auto-generated"
    print(f"YouTube transcript: {chosen.language} ({chosen.language_code}) — {label}")

    raw = chosen.fetch()
    segments = []
    for entry in raw:
        # Support both dict-style (API <0.6) and object-style (API >=0.6)
        try:
            start = float(entry["start"])
            duration = float(entry.get("duration", 0.0))
            text = entry["text"]
        except (TypeError, KeyError):
            start = float(entry.start)
            duration = float(getattr(entry, "duration", 0.0))
            text = entry.text
        segments.append({"start": start, "duration": duration, "text": text.strip()})

    language_code = chosen.language_code
    return segments, language_code


def youtube_transcript_to_clipsai(segments: list, language_code: str):
    """
    Convert YouTube transcript segments (phrase-level) to a clipsai Transcription object.
    Timestamps are distributed across characters proportionally within each segment,
    giving word-level precision sufficient for ClipFinder.
    """
    from clipsai.transcribe.transcription import Transcription

    char_info = []

    for seg in segments:
        seg_start = seg["start"]
        seg_duration = seg["duration"] if seg["duration"] > 0 else 0.5
        seg_end = seg_start + seg_duration
        raw_text = seg["text"] + " "  # trailing space as word separator

        words = raw_text.split(" ")
        words = [w for w in words if w]  # drop empty strings

        if not words:
            continue

        # Distribute segment time proportionally by character count
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            continue

        cursor = seg_start
        for word in words:
            if not word:
                continue
            word_duration = seg_duration * (len(word) / total_chars)
            word_end = cursor + word_duration

            for i, ch in enumerate(word):
                char_start = cursor + word_duration * (i / len(word))
                char_end = cursor + word_duration * ((i + 1) / len(word))
                char_info.append({
                    "char": ch,
                    "start_time": round(char_start, 4),
                    "end_time": round(char_end, 4),
                    "speaker": None,
                })
            # Add a space character between words
            char_info.append({
                "char": " ",
                "start_time": round(word_end, 4),
                "end_time": round(word_end, 4),
                "speaker": None,
            })
            cursor = word_end

    if not char_info:
        raise RuntimeError("YouTube transcript produced no character data.")

    transcription_dict = {
        "source_software": "youtube-transcript-api",
        "time_created": datetime.now(),
        "language": language_code,
        "num_speakers": None,
        "char_info": char_info,
    }
    return Transcription(transcription_dict)


def calculate_engagement_score(clip, transcription):
    """
    Calculate a custom engagement score for a clip based on available data.
    Higher scores indicate more engaging content.
    """
    # Get words in the clip
    clip_words = [w for w in transcription.get_word_info() 
                  if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time]
    
    if not clip_words:
        return 0.0
    
    # Calculate various engagement factors
    duration = clip.end_time - clip.start_time
    word_count = len(clip_words)
    word_density = word_count / duration if duration > 0 else 0
    
    # Count numbers, currency, and exclamation marks (engagement indicators)
    engagement_words = 0
    for word_info in clip_words:
        word = word_info["word"]
        if any(char.isdigit() for char in word) or '$' in word or '!' in word:
            engagement_words += 1
    
    # Calculate engagement score (0-1 scale)
    # Factors: word density (45%), engagement words ratio (30%), duration balance (25%)
    word_density_score = min(word_density / 3.0, 1.0)  # Normalize to 0-1
    engagement_ratio = engagement_words / word_count if word_count > 0 else 0
    duration_score = min(duration / 75.0, 1.0)  # Prefer clips around 75 seconds
    
    engagement_score = (word_density_score * 0.45 + 
                       engagement_ratio * 0.30 + 
                       duration_score * 0.25)
    
    return engagement_score

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="ClippedAI – generate YouTube Shorts from a local video or a YouTube URL."
)
_parser.add_argument(
    "--url",
    metavar="YOUTUBE_URL",
    default=None,
    help="YouTube video URL to download (1080p) and process instead of files in input/",
)
_args = _parser.parse_args()

# ---------------------------------------------------------------------------
# Build video→transcription map
# ---------------------------------------------------------------------------

# Find all mp4 files in the input directory
input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]

# If a YouTube URL is provided, download the video and fetch the transcript
if _args.url:
    print("\n=== YouTube URL mode ===")
    downloaded_path = download_youtube_video(_args.url, INPUT_DIR)
    downloaded_filename = os.path.basename(downloaded_path)

    # Try to get the YouTube transcript; fall back to whisper transcription if unavailable
    yt_transcription = None
    try:
        yt_segments, yt_lang = get_youtube_transcript(_args.url)
        print(f"Converting YouTube transcript ({len(yt_segments)} segments) to clipsai format...")
        yt_transcription = youtube_transcript_to_clipsai(yt_segments, yt_lang)
        print("YouTube transcript converted successfully.")
    except Exception as yt_err:
        print(f"YouTube transcript unavailable ({yt_err}). Will use Whisper transcription instead.")

    # Prepend downloaded video to the file list (deduplicate if already there)
    if downloaded_filename not in input_files:
        input_files.insert(0, downloaded_filename)

    # Pre-populate transcription map for this video
    video_transcription_map = {downloaded_filename: None}

    # Prompt number of clips for the downloaded video only
    video_max_clips = {}
    clip_ranges = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]
    print(f"\nHow many clips do you want for '{downloaded_filename}'?")
    for i, (low, high) in enumerate(clip_ranges, 1):
        print(f"  {i}) {low}-{high}")
    try:
        user_choice = int(input("Your choice: ").strip().replace('\n', ''))
        if not (1 <= user_choice <= len(clip_ranges)):
            raise ValueError
    except Exception:
        print("Invalid input. Defaulting to 2 clips.")
        user_choice = 1
    video_max_clips[downloaded_filename] = clip_ranges[user_choice - 1][1]

else:
    yt_transcription = None

    if not input_files:
        raise FileNotFoundError('No mp4 file found in input directory.')

    # Find all transcription files in the input directory
    transcription_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_transcription.pkl')]

    # If more than one mp4, ask user to match transcription files (if any)
    video_transcription_map = {}
    if len(input_files) > 1:
        print("Multiple video files detected:")
        for idx, f in enumerate(input_files, 1):
            print(f"  {idx}) {f}")
        print("\nAvailable transcription files:")
        for idx, f in enumerate(transcription_files, 1):
            print(f"  {idx}) {f}")
        print("\nFor each video, enter the number of the matching transcription file, or 0 to transcribe from scratch.")
        for vid_idx, video_file in enumerate(input_files, 1):
            while True:
                try:
                    match = input(f"Match transcription for '{video_file}' (0 for none): ").strip().replace('\n', '')
                    match_idx = int(match)
                    if match_idx == 0:
                        video_transcription_map[video_file] = None
                        break
                    elif 1 <= match_idx <= len(transcription_files):
                        video_transcription_map[video_file] = transcription_files[match_idx-1]
                        break
                    else:
                        print("Invalid choice. Try again.")
                except Exception:
                    print("Invalid input. Try again.")
    else:
        # Only one video, try to auto-match
        video_file = input_files[0]
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        expected_trans = f"{base_name}_transcription.pkl"
        if expected_trans in transcription_files:
            video_transcription_map[video_file] = expected_trans
        else:
            video_transcription_map[video_file] = None

    # Prompt user for number of clips for each video BEFORE any processing
    video_max_clips = {}
    clip_ranges = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]
    for video_file in video_transcription_map:
        print(f"\nHow many clips do you want for '{video_file}'?")
        for i, (low, high) in enumerate(clip_ranges, 1):
            print(f"  {i}) {low}-{high}")
        try:
            user_choice = int(input("Your choice: ").strip().replace('\n', ''))
            if not (1 <= user_choice <= len(clip_ranges)):
                raise ValueError
        except Exception:
            print("Invalid input. Defaulting to 2 clips.")
            user_choice = 1
        max_clips = clip_ranges[user_choice-1][1]
        print(f"Will select up to {max_clips} clips (if available and engaging).\n")
        video_max_clips[video_file] = max_clips

# ---------------------------------------------------------------------------
# Process each video file
# ---------------------------------------------------------------------------
for video_idx, (video_file, transcription_file) in enumerate(video_transcription_map.items(), 1):
    print(f"\n=== Processing Video {video_idx}/{len(video_transcription_map)}: {video_file} ===")
    input_path = os.path.abspath(os.path.join(INPUT_DIR, video_file))
    transcription_path = os.path.join(INPUT_DIR, transcription_file) if transcription_file else get_transcription_file_path(input_path)
    max_clips = video_max_clips[video_file]

    # 1. Transcribe the video (or use YouTube transcript, or load existing)
    transcriber = Transcriber(model_size=os.getenv('TRANSCRIPTION_MODEL', 'large-v1'))

    # Priority: YouTube transcript (if URL mode) > cached pkl > fresh Whisper
    if yt_transcription is not None and video_file == list(video_transcription_map.keys())[0]:
        transcription = yt_transcription
        print("Using YouTube transcript (skipping Whisper transcription).")
    elif transcription_file:
        transcription = load_existing_transcription(transcription_path)
        if transcription is None:
            transcription = transcribe_with_progress(input_path, transcriber)
            save_transcription(transcription, transcription_path)
    else:
        transcription = load_existing_transcription(transcription_path)
        if transcription is None:
            transcription = transcribe_with_progress(input_path, transcriber)
            save_transcription(transcription, transcription_path)

    # 2. Find clips
    clipfinder = ClipFinder()
    clips = clipfinder.find_clips(transcription=transcription)
    if not clips:
        print('No clips found in the video.')
        continue

    # 3. Filter clips by duration and select the best ones
    valid_clips = [c for c in clips if MIN_CLIP_DURATION <= (c.end_time - c.start_time) <= MAX_CLIP_DURATION]
    selected_clips = []

    if valid_clips:
        # Calculate engagement scores for all valid clips
        clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in valid_clips]
        # Sort by engagement score (highest first)
        clip_scores.sort(key=lambda x: x[1], reverse=True)
        # Select up to max_clips, but only include clips with engagement >= 0.6 (for 3rd and beyond)
        for i, (clip, score) in enumerate(clip_scores):
            if i < 2 or score >= 0.6:
                if len(selected_clips) < max_clips:
                    selected_clips.append(clip)
            else:
                break
        print(f'Selected top {len(selected_clips)} clips:')
        for i, clip in enumerate(selected_clips):
            score = calculate_engagement_score(clip, transcription)
            print(f'  Clip {i+1}: {clip.start_time:.1f}s - {clip.end_time:.1f}s (duration: {clip.end_time - clip.start_time:.1f}s, engagement: {score:.3f})')
        print(f'Clip selection criteria: Top engaging clips within {MIN_CLIP_DURATION}-{MAX_CLIP_DURATION} second range')
    else:
        print(f'No clips found between {MIN_CLIP_DURATION} and {MAX_CLIP_DURATION} seconds.')
        # Find clips that are too short and try to extend them
        short_clips = [c for c in clips if c.end_time - c.start_time < MIN_CLIP_DURATION]
        if short_clips:
            print('Attempting to extend most engaging short clips to minimum duration...')
            short_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in short_clips]
            short_clip_scores.sort(key=lambda x: x[1], reverse=True)
            # Take top 2 short clips and extend them
            for i, (clip, score) in enumerate(short_clip_scores[:2]):
                if clip.end_time - clip.start_time < MIN_CLIP_DURATION:
                    extension_needed = MIN_CLIP_DURATION - (clip.end_time - clip.start_time)
                    max_extension = min(extension_needed, MAX_CLIP_DURATION - (clip.end_time - clip.start_time))
                    extended_clip = Clip(
                        start_time=clip.start_time,
                        end_time=clip.end_time + max_extension,
                        start_char=clip.start_char,
                        end_char=clip.end_char
                    )
                    selected_clips.append(extended_clip)
                    print(f'Extended clip {i+1}: {extended_clip.start_time:.1f}s - {extended_clip.end_time:.1f}s (duration: {extended_clip.end_time - extended_clip.start_time:.1f}s)')
        else:
            # All clips are too long, trim the most engaging ones
            print('All clips are too long. Trimming most engaging clips to maximum duration...')
            long_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in clips]
            long_clip_scores.sort(key=lambda x: x[1], reverse=True)
            # Take top 2 long clips and trim them
            for i, (clip, score) in enumerate(long_clip_scores[:2]):
                if clip.end_time - clip.start_time > MAX_CLIP_DURATION:
                    trimmed_clip = Clip(
                        start_time=clip.start_time,
                        end_time=clip.start_time + MAX_CLIP_DURATION,
                        start_char=clip.start_char,
                        end_char=clip.end_char
                    )
                    selected_clips.append(trimmed_clip)
                    print(f'Trimmed clip {i+1}: {trimmed_clip.start_time:.1f}s - {trimmed_clip.end_time:.1f}s (duration: {trimmed_clip.end_time - trimmed_clip.start_time:.1f}s)')

    # Process each selected clip
    for clip_index, clip in enumerate(selected_clips):
        print(f'\n--- Processing Clip {clip_index + 1}/{len(selected_clips)} ---')
        # 4. Trim the video to the selected clip
        media_editor = MediaEditor()
        trimmed_path = os.path.join(OUTPUT_DIR, f'trimmed_clip_{clip_index + 1}.mp4')
        print('Trimming video to selected clip...')
        trim_video_ffmpeg(input_path, clip.start_time, clip.end_time, trimmed_path)
        # 5. Try to resize to 9:16 aspect ratio
        output_path = os.path.join(OUTPUT_DIR, f'yt_short_{clip_index + 1}.mp4')
        try:
            print('Resizing video to 9:16 aspect ratio...')
            aspect_ratio_width = int(os.getenv('ASPECT_RATIO_WIDTH', '9'))
            aspect_ratio_height = int(os.getenv('ASPECT_RATIO_HEIGHT', '16'))
            crops = resize(
                video_file_path=trimmed_path,
                pyannote_auth_token=HUGGINGFACE_TOKEN,
                aspect_ratio=(aspect_ratio_width, aspect_ratio_height)
            )
            resized_video_file = media_editor.resize_video(
                original_video_file=AudioVideoFile(trimmed_path),
                resized_video_file_path=output_path,
                width=crops.crop_width,
                height=crops.crop_height,
                segments=crops.to_dict()["segments"],
            )
            print(f'YouTube Short (9:16) saved to {output_path}')
        except Exception as e:
            print(f'Resizing failed: {e}')
            print('Falling back to FFmpeg vertical conversion (guaranteed 9:16)...')
            output_path = convert_to_vertical_ffmpeg(trimmed_path, output_path)

        # Validate output aspect ratio and enforce 9:16 if needed.
        try:
            width, height = get_video_dimensions(output_path)
            ratio = width / height if height else 0
            print(f'Output dimensions: {width}x{height} (ratio: {ratio:.4f})')
            if abs(ratio - (9 / 16)) > 0.01:
                print('Aspect ratio is not 9:16. Re-encoding with 9:16 fallback...')
                fixed_output_path = output_path.replace('.mp4', '_9x16.mp4')
                output_path = convert_to_vertical_ffmpeg(output_path, fixed_output_path)
        except Exception as dim_err:
            print(f'Could not validate output dimensions: {dim_err}')

        # 6. Add styled subtitles
        final_output = create_animated_subtitles(output_path, transcription, clip, output_path)
        # 7. Generate viral title using Groq API
        clip_text = " ".join([w["word"] for w in transcription.get_word_info() if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time])
        groq_api_key = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
        title = get_viral_title(clip_text, groq_api_key)
        print(f"\nViral Title for Clip {clip_index + 1}: {title}")
        # 8. Save the final video with the viral title (keep spaces, punctuation, and emojis)
        import shutil
        viral_filename = safe_filename(title).strip() + ".mp4"
        viral_path = os.path.join(OUTPUT_DIR, viral_filename)
        shutil.copy(final_output, viral_path)
        print(f"Final video saved as: {viral_path}\n")

print(f"\nSuccessfully created YouTube Shorts for {len(video_transcription_map)} video(s)!")