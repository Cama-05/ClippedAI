"""Main script to process videos and create YouTube Shorts using AI-powered transcription and clip detection."""

import atexit
import argparse
import csv
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from urllib.parse import urlparse
import warnings
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keep noisy third-party libraries quiet before they are imported.
os.environ['PYANNOTE_SUPPRESS_TORCHCODEC_WARNING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# Must be set before importing torch, otherwise the warning is emitted too early.
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", message=r".*torchcodec is not installed correctly.*")

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
warnings.filterwarnings("ignore", message="Model was trained with pyannote.audio")
warnings.filterwarnings("ignore", message="Model was trained with torch")
warnings.filterwarnings("ignore", message="Lightning automatically upgraded")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated")

# Suppress unnecessary warnings via environment variables
os.environ['FFREPORT'] = 'file=ffmpeg.log:level=32'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load environment variables
load_dotenv()

# Download required NLTK data only if missing (avoids repeated startup overhead)
def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name, quiet=True)

ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')

# --- Directories ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
FONT_DIR = "fonts"
METADATA_CSV_PATH = os.getenv("METADATA_CSV_PATH", os.path.join(OUTPUT_DIR, "shorts_metadata.csv"))

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
ENABLE_PYANNOTE_RESIZE = os.getenv("ENABLE_PYANNOTE_RESIZE", "false").strip().lower() == "true"
ENABLE_GPU_VIDEO_EDITING = os.getenv("ENABLE_GPU_VIDEO_EDITING", "true").strip().lower() == "true"

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- LLM Provider (metadata generation) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Logo Overlay ---
LOGO_OPACITY = max(0.0, min(1.0, float(os.getenv("LOGO_OPACITY", "0.55"))))
LOGO_WIDTH_RATIO = max(0.05, min(0.9, float(os.getenv("LOGO_WIDTH_RATIO", "0.50"))))
LOGO_EDGE_MARGIN = max(0, int(os.getenv("LOGO_EDGE_MARGIN", os.getenv("LOGO_MARGIN_TOP", "70"))))

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
    """Sanitize filename for Windows while preserving readable Unicode (including emojis)."""
    # Normalize composed/decomposed Unicode sequences to stable code points.
    s = unicodedata.normalize("NFKC", s)

    forbidden = '<>:"/\\|?*'
    cleaned_chars = []
    for ch in s:
        code = ord(ch)
        if code < 32:
            continue
        if ch in forbidden:
            cleaned_chars.append('_')
            continue
        # Drop variation selectors (for example U+FE0F), often invisible in filenames.
        if 0xFE00 <= code <= 0xFE0F:
            continue
        cleaned_chars.append(ch)

    name = ''.join(cleaned_chars)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'_+', '_', name).strip(' .')

    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }

    if not name:
        return "clip"
    if name.upper() in reserved:
        name = f"{name}_file"

    return name[:150]


def get_font_path(font_name: str) -> str:
    """Get the path to a font file in the font directory."""
    # Try different extensions for the font file
    for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
        font_path = os.path.join("fonts", f"{font_name}{ext}")
        if os.path.exists(font_path):
            return font_path
    # If not found with extension, return as is (for system fonts)
    return os.path.join("fonts", font_name)


@lru_cache(maxsize=1)
def can_use_nvenc() -> bool:
    """Return True when FFmpeg NVENC is available and GPU editing is enabled."""
    if not ENABLE_GPU_VIDEO_EDITING:
        return False
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False


def get_video_encoder_args() -> List[str]:
    """Return FFmpeg video encoder args, preferring NVENC when available."""
    if can_use_nvenc():
        # p5 + vbr_hq + cq gives good quality/speed balance on RTX cards.
        return [
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-rc:v', 'vbr_hq',
            '-cq:v', '19',
            '-b:v', '0',
            '-profile:v', 'high',
            '-pix_fmt', 'yuv420p',
        ]

    # CPU fallback keeps previous quality-oriented defaults.
    return [
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '16',
        '-profile:v', 'high',
        '-pix_fmt', 'yuv420p',
    ]


def active_video_encoder_name() -> str:
    return 'h264_nvenc' if can_use_nvenc() else 'libx264'

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
        *get_video_encoder_args(),
        '-level:v', '4.1',            # supports 1080p@30fps
        '-colorspace', 'bt709',       # correct HD color metadata
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-r', '30',                   # force constant 30 fps → eliminates stuttering
        '-c:a', 'aac',
        '-b:a', '256k',               # higher audio quality
        '-ar', '48000',               # 48 kHz — YouTube standard sample rate
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
        *get_video_encoder_args(),
        '-level:v', '4.1',
        '-colorspace', 'bt709',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-r', '30',
        '-c:a', 'aac',
        '-b:a', '256k',
        '-ar', '48000',
        '-movflags', '+faststart',
        '-y',
        output_path,
    ]
    print(f'Forcing 9:16 output with fallback conversion -> {os.path.basename(output_path)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg 9:16 conversion failed:\n{result.stderr}')
    return output_path


def normalize_vertical_output_ffmpeg(input_path: str, output_path: str) -> str:
    """Normalize an already-vertical clip to an exact 1080x1920 canvas."""
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', 'scale=1080:1920:flags=lanczos,setsar=1,format=yuv420p',
        *get_video_encoder_args(),
        '-level:v', '4.1',
        '-colorspace', 'bt709',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-r', '30',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        '-y',
        output_path,
    ]
    print(f'Normalizing output to exact 1080x1920 -> {os.path.basename(output_path)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg normalization failed:\n{result.stderr}')
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


def get_video_fps(video_path: str) -> float:
    """Return FPS for the first video stream using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffprobe failed while reading fps:\n{result.stderr}')

    rate = result.stdout.strip()
    if not rate or rate == '0/0':
        return 0.0
    if '/' in rate:
        num_str, den_str = rate.split('/', 1)
        den = float(den_str)
        if den == 0:
            return 0.0
        return float(num_str) / den
    return float(rate)


def download_logo_from_url(logo_url: str, download_dir: str) -> str:
    """Download a remote logo asset to a local temporary path for FFmpeg."""
    import requests

    parsed = urlparse(logo_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported logo URL scheme: {parsed.scheme or 'missing'}")

    raw_name = os.path.basename(parsed.path) or "logo.png"
    suffix = os.path.splitext(raw_name)[1].lower()
    if suffix not in {".png", ".webp", ".jpg", ".jpeg"}:
        suffix = ".png"

    os.makedirs(download_dir, exist_ok=True)
    local_logo_path = os.path.join(download_dir, f"logo_asset{suffix}")

    response = requests.get(logo_url, timeout=60)
    response.raise_for_status()
    with open(local_logo_path, 'wb') as logo_file:
        logo_file.write(response.content)

    return local_logo_path


def normalize_logo_position(raw_position: str) -> str:
    """Normalize CLI logo position values, supporting English and Italian labels."""
    normalized = (raw_position or "").strip().lower()
    aliases = {
        "center": "center",
        "centro": "center",
        "top-center": "top-center",
        "center-top": "top-center",
        "centro-alto": "top-center",
        "bottom-center": "bottom-center",
        "center-bottom": "bottom-center",
        "centro-basso": "bottom-center",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Invalid --logo-position '{raw_position}'. "
            "Use one of: center, top-center, bottom-center "
            "(or: centro, centro-alto, centro-basso)."
        )
    return aliases[normalized]


def apply_logo_overlay_ffmpeg(input_path: str, output_path: str, logo_path: str, logo_position: str) -> str:
    """Overlay a semi-transparent logo at the requested screen position."""
    if logo_position == "top-center":
        overlay_y = str(LOGO_EDGE_MARGIN)
    elif logo_position == "bottom-center":
        overlay_y = f"main_h-overlay_h-{LOGO_EDGE_MARGIN}"
    else:
        overlay_y = "(main_h-overlay_h)/2"

    filter_graph = (
        f"[1:v][0:v]scale2ref=w=main_w*{LOGO_WIDTH_RATIO:.4f}:h=ow/mdar[logo][base];"
        f"[logo]format=rgba,colorchannelmixer=aa={LOGO_OPACITY:.3f}[logo_alpha];"
        f"[base][logo_alpha]overlay=(main_w-overlay_w)/2:{overlay_y}:format=auto[v]"
    )
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-i', logo_path,
        '-filter_complex', filter_graph,
        '-map', '[v]',
        '-map', '0:a?',
        *get_video_encoder_args(),
        '-level:v', '4.1',
        '-colorspace', 'bt709',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-r', '30',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        '-y',
        output_path,
    ]
    print(f"Applying logo overlay from: {logo_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg logo overlay failed:\n{result.stderr}')
    return output_path

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
        *get_video_encoder_args(),
        '-level:v', '4.1',
        '-colorspace', 'bt709',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-r', '30',
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

def get_viral_metadata(transcript_text: str) -> Tuple[str, str, str]:
    import requests

    prompt = (
        "You are generating metadata for short-form video platforms (YouTube Shorts, TikTok, Reels).\n"
        "Given the transcript, output EXACTLY 3 lines in this format:\n"
        "TITLE: <max 7 words, can include 1 emoji, no hashtags>\n"
        "DESCRIPTION: <1 short engaging sentence, max 140 chars,can include up to 2 hashtags>\n"
        "TAGS: <8-12 comma-separated tags, no # symbol>\n"
        "Do not output anything else.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    default_title = "Untitled Clip"
    default_description = "Short highlight clip generated by ClippedAI."
    default_tags = "shorts,viral,clip,content,highlights"

    provider = LLM_PROVIDER if LLM_PROVIDER in {"groq", "openai"} else "groq"
    if provider != LLM_PROVIDER:
        print(f"Invalid LLM_PROVIDER '{LLM_PROVIDER}'. Falling back to 'groq'.")

    if provider == "openai":
        api_key = OPENAI_API_KEY
        model = OPENAI_MODEL
        url = "https://api.openai.com/v1/chat/completions"
        provider_label = "OpenAI"
    else:
        api_key = GROQ_API_KEY
        model = GROQ_MODEL
        url = "https://api.groq.com/openai/v1/chat/completions"
        provider_label = "Groq"

    if not api_key or api_key.startswith("your_"):
        print(f"{provider_label} API key not configured. Using default metadata.")
        return default_title, default_description, default_tags

    try:
        print(f"Generating metadata via {provider_label} ({model})...")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            url,
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']

        title_match = re.search(r"^TITLE:\s*(.+)$", content, flags=re.IGNORECASE | re.MULTILINE)
        description_match = re.search(r"^DESCRIPTION:\s*(.+)$", content, flags=re.IGNORECASE | re.MULTILINE)
        tags_match = re.search(r"^TAGS:\s*(.+)$", content, flags=re.IGNORECASE | re.MULTILINE)

        title = title_match.group(1).strip().strip('"') if title_match else default_title
        description = description_match.group(1).strip().strip('"') if description_match else default_description
        tags = tags_match.group(1).strip().strip('"') if tags_match else default_tags

        if not title:
            title = default_title
        if not description:
            description = default_description
        if not tags:
            tags = default_tags

        # Normalize tags as comma-separated values without leading '#'.
        normalized_tags = []
        for raw_tag in tags.split(','):
            tag = raw_tag.strip().lstrip('#')
            if tag:
                normalized_tags.append(tag)
        tags = ", ".join(normalized_tags) if normalized_tags else default_tags

        return title[:120], description[:160], tags[:500]
    except requests.exceptions.HTTPError as e:
        print(f"Error with {provider_label} API: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        return default_title, default_description, default_tags
    except Exception as e:
        print(f"Unexpected error with {provider_label} API: {e}")
        return default_title, default_description, default_tags


def append_clip_metadata_csv(csv_path: str, row: Dict[str, str]) -> None:
    """Append clip metadata to CSV, creating header on first write."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "created_at",
        "source_video",
        "source_url",
        "clip_index",
        "clip_start_s",
        "clip_end_s",
        "clip_duration_s",
        "clip_file",
        "title",
        "description",
        "tags",
    ]

    needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

def download_youtube_video(url: str, output_dir: str = "input") -> str:
    """
    Download a YouTube video prioritizing 1440p (~2K) at 30fps using yt-dlp,
    with fallback to the best available format/resolution.
    Returns the path to the downloaded .mp4 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    # Prefer 1440p (~2K) at <=30fps; fallback progressively to any available quality.
    # --restrict-filenames keeps the filename ASCII-only (avoids libmagic UnicodeDecodeError on Windows)
    cmd = [
        "yt-dlp",
        "--format", (
            "bestvideo[height=1440][fps<=30][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height=1440][fps<=30]+bestaudio/"
            "bestvideo[height<=1440][fps<=30][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=1440][fps<=30]+bestaudio/"
            "bestvideo[height<=1440]+bestaudio/"
            "best[height<=1440]/"
            "best"
        ),
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--restrict-filenames",
        "--output", output_template,
        "--print", "before_dl:Selected source: %(width)sx%(height)s @ %(fps)s fps (format_id=%(format_id)s)",
        "--print", "after_move:filepath",
        url,
    ]
    print(f"Downloading YouTube video (target: 1440p ~2K @<=30fps, with fallback): {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    selected_source_line = next((line for line in stdout_lines if line.startswith("Selected source:")), None)
    if selected_source_line:
        print(selected_source_line)

    # The last printed line is the final filepath
    filepath = stdout_lines[-1] if stdout_lines else ""
    if not filepath or not os.path.exists(filepath):
        # Fallback: find newest mp4 in output_dir
        mp4s = sorted(Path(output_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise FileNotFoundError("yt-dlp completed but no mp4 found in input/")
        filepath = str(mp4s[0])

    try:
        width, height = get_video_dimensions(filepath)
        fps = get_video_fps(filepath)
        print(f"Downloaded source details: {width}x{height} @ {fps:.2f} fps")
    except Exception as e:
        print(f"Could not verify downloaded source details (resolution/fps): {e}")

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
_parser.add_argument(
    "--logo-url",
    metavar="LOGO_URL",
    default=None,
    help="Remote logo image URL (for example S3 or CDN). If omitted, no logo overlay is applied.",
)
_parser.add_argument(
    "--logo-position",
    metavar="LOGO_POSITION",
    default="center",
    help=(
        "Logo position: center, top-center, bottom-center "
        "(or: centro, centro-alto, centro-basso). Default: center."
    ),
)
_args = _parser.parse_args()
logo_position = normalize_logo_position(_args.logo_position)
print(
    f"Video encoder selected: {active_video_encoder_name()} "
    f"({'GPU' if can_use_nvenc() else 'CPU fallback'})"
)
if ENABLE_GPU_VIDEO_EDITING and not can_use_nvenc():
    print("GPU video editing requested but NVENC is unavailable. Falling back to libx264.")

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
resolved_logo_path = None
if _args.logo_url:
    logo_download_dir = tempfile.mkdtemp(prefix="clippedai_logo_")
    atexit.register(shutil.rmtree, logo_download_dir, ignore_errors=True)
    try:
        resolved_logo_path = download_logo_from_url(_args.logo_url, logo_download_dir)
        print(f"Logo overlay enabled from URL: {_args.logo_url}")
        print(f"Logo position: {logo_position}")
    except Exception as logo_download_err:
        raise RuntimeError(f"Could not download logo from --logo-url: {logo_download_err}") from logo_download_err
else:
    print("Logo overlay disabled: --logo-url not provided.")

for video_idx, (video_file, transcription_file) in enumerate(video_transcription_map.items(), 1):
    print(f"\n=== Processing Video {video_idx}/{len(video_transcription_map)}: {video_file} ===")
    input_path = os.path.abspath(os.path.join(INPUT_DIR, video_file))
    transcription_path = os.path.join(INPUT_DIR, transcription_file) if transcription_file else get_transcription_file_path(input_path)
    max_clips = video_max_clips[video_file]

    # 1. Transcribe the video (or use YouTube transcript, or load existing)

    # Priority: YouTube transcript (if URL mode) > cached pkl > fresh Whisper
    if yt_transcription is not None and video_file == list(video_transcription_map.keys())[0]:
        transcription = yt_transcription
        print("Using YouTube transcript (skipping Whisper transcription).")
    elif transcription_file:
        transcription = load_existing_transcription(transcription_path)
        if transcription is None:
            transcriber = Transcriber(model_size=os.getenv('TRANSCRIPTION_MODEL', 'large-v1'))
            transcription = transcribe_with_progress(input_path, transcriber)
            save_transcription(transcription, transcription_path)
    else:
        transcription = load_existing_transcription(transcription_path)
        if transcription is None:
            transcriber = Transcriber(model_size=os.getenv('TRANSCRIPTION_MODEL', 'large-v1'))
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
    pyannote_resize_available = ENABLE_PYANNOTE_RESIZE
    if not pyannote_resize_available:
        print('Pyannote resize disabled (ENABLE_PYANNOTE_RESIZE=false). Using FFmpeg 9:16 fallback for speed/stability.')

    for clip_index, clip in enumerate(selected_clips):
        print(f'\n--- Processing Clip {clip_index + 1}/{len(selected_clips)} ---')
        temp_dir = tempfile.mkdtemp(prefix="clippedai_")
        try:
            # 4. Trim the video to the selected clip
            media_editor = MediaEditor()
            trimmed_path = os.path.join(temp_dir, f'trimmed_clip_{clip_index + 1}.mp4')
            print('Trimming video to selected clip...')
            trim_video_ffmpeg(input_path, clip.start_time, clip.end_time, trimmed_path)

            # 5. Try to resize to 9:16 aspect ratio
            output_path = os.path.join(temp_dir, f'yt_short_{clip_index + 1}.mp4')
            if pyannote_resize_available:
                try:
                    print('Resizing video to 9:16 aspect ratio with pyannote...')
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
                    print(f'Resizing with pyannote failed: {e}')
                    print('Switching to FFmpeg 9:16 fallback for remaining clips in this video...')
                    pyannote_resize_available = False
                    output_path = convert_to_vertical_ffmpeg(trimmed_path, output_path)
            else:
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
                    width, height = get_video_dimensions(output_path)

                if width != 1080 or height != 1920:
                    print('Output is vertical but not exact 1080x1920. Normalizing...')
                    normalized_output_path = output_path.replace('.mp4', '_1080x1920.mp4')
                    output_path = normalize_vertical_output_ffmpeg(output_path, normalized_output_path)
            except Exception as dim_err:
                print(f'Could not validate output dimensions: {dim_err}')

            # 6. Optionally add the logo overlay on the processed clip.
            final_output = output_path
            if resolved_logo_path:
                logo_output_path = output_path.replace('.mp4', '_with_logo.mp4')
                try:
                    final_output = apply_logo_overlay_ffmpeg(
                        output_path,
                        logo_output_path,
                        resolved_logo_path,
                        logo_position,
                    )
                except Exception as logo_err:
                    print(f'Could not apply logo overlay: {logo_err}')
                    final_output = output_path

            # 7. Generate social metadata (title, description, tags).
            clip_text = " ".join([
                w["word"] for w in transcription.get_word_info()
                if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time
            ])
            title, description, tags = get_viral_metadata(clip_text)
            print(f"\nMetadata for Clip {clip_index + 1}:")
            print(f"  Title: {title}")
            print(f"  Description: {description}")
            print(f"  Tags: {tags}")

            # 8. Save final clip using the generated title as filename (with collision-safe suffix).
            safe_title = safe_filename(title).strip() or f"clip_{clip_index + 1}"
            viral_filename = f"{safe_title}.mp4"
            viral_path = os.path.join(OUTPUT_DIR, viral_filename)
            suffix = 2
            while os.path.exists(viral_path):
                viral_filename = f"{safe_title}_{suffix}.mp4"
                viral_path = os.path.join(OUTPUT_DIR, viral_filename)
                suffix += 1
            shutil.move(final_output, viral_path)
            print(f"Final video saved as: {viral_path}")

            # 9. Append metadata row to CSV.
            clip_duration = clip.end_time - clip.start_time
            metadata_row = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_video": video_file,
                "source_url": _args.url or "",
                "clip_index": str(clip_index + 1),
                "clip_start_s": f"{clip.start_time:.3f}",
                "clip_end_s": f"{clip.end_time:.3f}",
                "clip_duration_s": f"{clip_duration:.3f}",
                "clip_file": viral_filename,
                "title": title,
                "description": description,
                "tags": tags,
            }
            append_clip_metadata_csv(METADATA_CSV_PATH, metadata_row)
            print(f"Metadata appended to CSV: {METADATA_CSV_PATH}\n")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

print(f"\nSuccessfully created YouTube Shorts for {len(video_transcription_map)} video(s)!")