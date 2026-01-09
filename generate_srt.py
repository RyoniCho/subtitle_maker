import stable_whisper
import argparse
import os

# Set PyTorch fallback for MPS to handle float64 operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import subprocess
import torch
import sys

def extract_audio(video_path, audio_output_path):
    """
    Extracts audio from video using ffmpeg.
    Converts to 16kHz mono wav which is optimal for Whisper.
    """
    command = [
        'ffmpeg',
        '-y',               # Overwrite output files
        '-i', video_path,   # Input file
        '-vn',              # No video
        '-acodec', 'pcm_s16le', # PCM 16-bit
        '-ar', '16000',     # 16kHz sampling rate
        '-ac', '1',         # Mono
        '-loglevel', 'error', # Suppress logs
        audio_output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False

def generate_srt(video_path, model_name='large-v2', language='ja', output_dir=None):
    if not os.path.exists(video_path):
        print(f"Error: File not found - {video_path}")
        return

    print(f"Processing: {video_path}")
    
    # Define output filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    srt_output_path = os.path.join(output_dir, f"{base_name}.srt")
    audio_temp_path = os.path.join(output_dir, f"{base_name}_temp.wav")
    
    # 1. Extract Audio
    print("Step 1: Extracting audio...")
    if not extract_audio(video_path, audio_temp_path):
        return

    # 2. Load Model
    print(f"Step 2: Loading Whisper model '{model_name}'...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        
    print(f"Using device: {device}")
    
    try:
        model = stable_whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        if os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)
        return

    # 3. Transcribe
    print(f"Step 3: Transcribing audio (Language: {language})...")
    # stable-ts solves hallucinations via dynamic quantization of timestamps and other heuristics
    # regroupling=True helps with splitting logic
    try:
        # Use fp16=False to avoid MPS float64/float16 issues (force float32)
        result = model.transcribe(
            audio_temp_path, 
            language=language, 
            regroup=True,
            fp16=False
        )  
        
        # 4. Save SRT
        print(f"Step 4: Saving SRT to {srt_output_path}...")
        result.to_srt_vtt(srt_output_path, segment_level=False, word_level=False)
        print("Done!")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        # Cleanup
        if os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SRT from Video using Stable-Whisper")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--model", "-m", default="large-v3", help="Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)")
    parser.add_argument("--lang", "-l", default="ja", help="Language code (ja, en, ko, etc.)")
    parser.add_argument("--output", "-o", default=None, help="Output directory (optional)")
    
    args = parser.parse_args()
    
    generate_srt(args.input, args.model, args.lang, args.output)
