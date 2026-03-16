# ClippedAI - AI-Powered YouTube Shorts Generator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**Open-source alternative to OpusClip** - Transform long-form videos into engaging YouTube Shorts automatically using AI-powered transcription, clip detection, and viral title generation. Built on the powerful [clipsai](https://github.com/ClipsAI/clipsai) library.

## Features

- **Smart Clip Detection**: AI identifies the most engaging moments in your videos
- **Auto-Resize**: Automatically crops videos to 9:16 aspect ratio for YouTube Shorts
- **Transparent Logo Overlay**: Adds your logo at the top of the exported clip with adjustable opacity
- **Animated Subtitles**: Clean, bold subtitles with smart styling (white text, yellow for numbers/currency)
- **Viral Title Generation**: AI generates catchy, titles optimized for engagement
- **Transcription Caching**: Save time by reusing existing transcriptions
- **Multiple Video Support**: Process multiple videos in one session
- **Engagement Scoring**: Intelligent clip selection based on content engagement metrics

## Why Choose ClippedAI Over OpusClip?

| Feature | ClippedAI | OpusClip |
| ------- | --------- | -------- |
| **Cost** | 100% Free | $39/month |
| **Privacy** | Local processing | Cloud-based |
| **Customization** | Fully customisable | Limited options |
| **API Keys** | Free (HuggingFace + Groq) | Paid subscriptions |
| **Offline Use** | Works offline (with no auto titles) | Requires internet |
| **Source Code** | Open source | Proprietary |
| **Model Control** | Choose your own models | Fixed models |
| **Transcription Caching** | Save time & money | No caching |

**Perfect for:** Content creators, developers, and anyone who wants professional video editing capabilities without the monthly subscription costs!

## Quick Start

### Prerequisites

- **Python 3.8+** (Tested on 3.11)
- **FFmpeg** installed and available in PATH
- **8GB+ RAM** (16GB+ recommended for large models)
- **GPU** (optional but recommended for faster processing)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Shaarav4795/ClippedAI.git
   cd ClippedAI
   ```

2. **Create and activate virtual environment**

   ```bash
   # On macOS/Linux
   python3 -m venv env
   source env/bin/activate
   
   # On Windows
   python -m venv env
   env\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**

   ```bash
   # macOS (using Homebrew)
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows (using Chocolatey)
   choco install ffmpeg
   
   # Or download from https://ffmpeg.org/download.html
   ```

5. **Create environment file**

   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file with your API keys:
   nano .env
   ```

### API Keys Setup

#### HuggingFace Token (Required) - **100% FREE**

1. **Sign up for HuggingFace**
   - Go to [HuggingFace](https://huggingface.co/join) and create a free account

2. **Request access to Pyannote models**
   - Visit [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - Click "Access repository" and accept the terms
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Access repository" and accept the terms
   - Visit [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
   - Click "Access repository" and accept the terms

3. **Create your API token**
   - Go to [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Give it a name (e.g., "ClippedAI")
   - Select "Read" role (minimum required)
   - Click "Generate token"
   - **Copy the token immediately** (you won't see it again)

4. **Add the token to your environment file**
   - Edit the `.env` file and replace `your_huggingface_token_here` with your actual token
   - Example: `HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**Note**: The first time you run the script, it will download the Pyannote models (~2GB). This may take several minutes depending on your internet connection.

#### Groq API Key (Optional - free provider for viral titles)

1. Sign up at [Groq](https://console.groq.com/) (free tier available)
2. Get your API key from the dashboard
3. Add your API key to the `.env` file where `GROQ_API_KEY=your_groq_api_key_here`

#### OpenAI API Key (Optional - paid provider for viral titles)

1. Sign up at [OpenAI](https://platform.openai.com/)
2. Create an API key from [API Keys](https://platform.openai.com/api-keys)
3. Add your API key to the `.env` file where `OPENAI_API_KEY=your_openai_api_key_here`

#### Select Metadata Provider in `.env`

Use `LLM_PROVIDER` to choose which model provider generates title/description/tags:

```env
LLM_PROVIDER=groq      # free option
# LLM_PROVIDER=openai  # paid option
```

If the selected provider key is missing, the script safely falls back to default metadata.

## Choosing the Right Transcription Model

The script uses Whisper models via `clipsai`. Choose based on your hardware:

### Model Size Comparison

| Model | Size | Speed | Accuracy | RAM Usage | Best For |
|-------|------|-------|----------|-----------|----------|
| `tiny` | 39MB | Very Fast | Low | 1GB | Quick testing, basic accuracy |
| `base` | 74MB | Fast | Medium | 1GB | Good balance, most users |
| `small` | 244MB | Moderate | High | 2GB | Better accuracy, recommended |
| `medium` | 769MB | Slow | Very High | 4GB | High accuracy, good hardware |
| `large-v1` | 1550MB | Very Slow | Excellent | 8GB | Best accuracy, powerful hardware |
| `large-v2` | 1550MB | Very Slow | Excellent | 8GB | Latest model, best results |

### Hardware Recommendations

**For CPU-only systems:**

- 4GB RAM: Use `tiny` or `base`
- 8GB RAM: Use `small` or `medium`
- 16GB+ RAM: Use `large-v1` or `large-v2`

**For GPU systems:**

- Any GPU with 4GB+ VRAM: Use `large-v2` (best results)
- GPU with 2GB VRAM: Use `medium` or `large-v1`

### Changing the Model

The transcription model can be configured via the `TRANSCRIPTION_MODEL` environment variable in your `.env` file:

```
TRANSCRIPTION_MODEL=large-v1  # Options: tiny, base, small, medium, large-v1, large-v2
```

## Project Structure

```
ClippedAI/
├── main.py                 # Main application script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── input/                 # Place your videos here
│   ├── video1.mp4
│   ├── video2.mp4
│   └── *_transcription.pkl # Cached transcriptions (auto-generated)
├── output/                # Generated YouTube Shorts
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── ...
└── env/                   # Virtual environment (created during setup)
```

## Customization

All key settings can now be configured through the `.env` file or within `main.py` for subtitle styling.

## Usage

1. **Add your videos** to the `input/` folder

   ```bash
   cp /path/to/your/video.mp4 input/
   ```

2. **Optional: provide a remote logo URL**

   If you want a logo overlay, pass it at runtime with `--logo-url`.
   This is intended for hosted assets such as S3, Cloudflare R2, or a CDN.
   You can also set the position with `--logo-position`.

3. **Run the script**

   ```bash
   python main.py --logo-url "https://your-bucket.s3.amazonaws.com/brand/logo.png" --logo-position center
   ```

   If `--logo-url` is omitted, the logo overlay is skipped.

4. **Follow the prompts** to:
   - Match videos with existing transcriptions (if any)
   - Choose how many clips to generate per video
   - Let AI process and create your YouTube Shorts

5. **Find your results** in the `output/` folder

## Customization

### Font Configuration

The script uses Montserrat Extra Bold for subtitles (from Google Fonts). To change fonts:

1. **Place your preferred font file** in the `fonts/` directory
2. **Edit the font name** in `main.py` line 158:

   ```python
   SUBTITLE_FONT = "Your-Font-Name"
   ```

3. **Update the ASS style definitions** in the `create_animated_subtitles` function to reference the new font

### Environment Variables Configuration

All key settings can now be configured through the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | your_huggingface_token_here | HuggingFace API token for speaker diarization |
| `LLM_PROVIDER` | groq | Metadata provider (`groq` or `openai`) |
| `GROQ_API_KEY` | your_groq_api_key_here | Groq API key (used when `LLM_PROVIDER=groq`) |
| `GROQ_MODEL` | llama-3.1-8b-instant | Groq model for metadata generation |
| `OPENAI_API_KEY` | your_openai_api_key_here | OpenAI API key (used when `LLM_PROVIDER=openai`) |
| `OPENAI_MODEL` | gpt-4o-mini | OpenAI model for metadata generation |
| `MIN_CLIP_DURATION` | 45 | Minimum duration in seconds for YouTube Shorts |
| `MAX_CLIP_DURATION` | 120 | Maximum duration in seconds for YouTube Shorts |
| `TRANSCRIPTION_MODEL` | medium | Whisper model to use (tiny, base, small, medium, large-v1, large-v2) |
| `ASPECT_RATIO_WIDTH` | 9 | Width for aspect ratio (used with height for video resizing) |
| `ASPECT_RATIO_HEIGHT` | 16 | Height for aspect ratio (used with width for video resizing) |
| `ENABLE_GPU_VIDEO_EDITING` | true | Try to use FFmpeg `h264_nvenc` for GPU video encoding; auto-fallback to CPU `libx264` if unavailable |
| `LOGO_OPACITY` | 0.55 | Logo transparency from `0.0` to `1.0` |
| `LOGO_WIDTH_RATIO` | 0.50 | Logo width relative to the video width (0.50 = 540px on 1080x1920 output) |
| `LOGO_EDGE_MARGIN` | 70 | Edge distance in pixels used for `top-center` and `bottom-center` logo positions |

### GPU Video Editing (RTX / NVENC)

ClippedAI can use your NVIDIA GPU for FFmpeg video encoding steps (trim/resize/normalize/logo overlay) through `h264_nvenc`.

- Set `ENABLE_GPU_VIDEO_EDITING=true` in `.env`
- Ensure your FFmpeg build includes NVENC support
- If NVENC is not available, the script falls back automatically to CPU (`libx264`)

At startup, the script prints the selected encoder so you can confirm whether GPU is active.

### Logo Overlay

If `--logo-url` is provided, ClippedAI downloads that image for the current run and overlays it at the center of every exported clip.

If `--logo-url` is not provided, no logo overlay is applied.

Supported `--logo-position` values:

- `center` (or `centro`)
- `top-center` (or `centro-alto`)
- `bottom-center` (or `centro-basso`)

- Best format: transparent PNG
- Default position: centered on the screen
- Default opacity: `0.55`
- Default size: `50%` of the video width (recommended logo asset: `540x540` PNG)

Example `.env` configuration:

```env
LOGO_OPACITY=0.45
LOGO_WIDTH_RATIO=0.50
```

Example run command:

```bash
python main.py --url "https://www.youtube.com/watch?v=..." --logo-url "https://your-cdn.example.com/brands/acme/logo.png" --logo-position top-center
```

### Engagement Scoring

The AI uses multiple factors to select the best clips:

- Word density (45% weight)
- Engagement words ratio (30% weight)
- Duration balance (25% weight)

## Troubleshooting

### Common Issues

**"No module named 'clipsai'"**

```bash
pip install clipsai
```

**"FFmpeg not found"**

- Ensure FFmpeg is installed and in your system PATH
- Restart your terminal after installation

**"CUDA out of memory"**

- Use a smaller transcription model
- Close other GPU-intensive applications
- Reduce batch size if applicable

**"Font not found"**

- Install the required font system-wide
- Or change to a system font in the code

**"API key errors"**

- Verify your API keys are correct
- Check your internet connection
- Ensure you have sufficient API credits

**"HuggingFace access denied"**

- Make sure you've requested access to all three Pyannote repositories
- Wait a few minutes after requesting access before running the script
- Verify your HuggingFace token has "read" permissions

### Performance Tips

1. **Use SSD storage** for faster video processing
2. **Close unnecessary applications** to free up RAM
3. **Use GPU acceleration** if available
4. **Process videos in smaller batches** for large files
5. **Cache transcriptions** to avoid re-processing if testing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license - see the LICENSE file for details.

## Acknowledgments

- [clipsai](https://github.com/ClipsAI/clipsai) - Core video processing library
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [Groq](https://groq.com/) - AI title generation

## Support

- **Bug Reports**: [GitHub Issues](https://github.com/Shaarav4795/ClippedAI/issues)
- **Discord**: .shaarav4795.

---

**Star this repository** if you find it helpful!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Shaarav4795/ClippedAI&type=date&legend=top-left)](https://www.star-history.com/#Shaarav4795/ClippedAI&type=date&legend=top-left)
