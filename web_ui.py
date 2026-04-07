import streamlit as st
import os
import sys

# ==========================================
# Windows Unicode Path Fix (Nagisa & Torch)
# ==========================================
# This fix addresses the issue where Japanese tagger 'nagisa' and PyTorch 
# cannot read files from a Windows user folder with Korean/Non-ASCII characters.

# 1. Nagisa model path monkey-patch
try:
    import nagisa
    from nagisa.tagger import Tagger
    
    # Preserve original __init__
    original_nagisa_init = Tagger.__init__
    
    def patched_nagisa_init(self, *args, **kwargs):
        # Redirect to a safe, non-ASCII path
        safe_model_dir = r"C:\nagisa_data"
        if os.path.exists(safe_model_dir):
            if 'vocabs' not in kwargs:
                kwargs['vocabs'] = os.path.join(safe_model_dir, "nagisa_v001.vocabs")
            if 'params' not in kwargs:
                kwargs['params'] = os.path.join(safe_model_dir, "nagisa_v001.params")
            if 'hp' not in kwargs:
                kwargs['hp'] = os.path.join(safe_model_dir, "nagisa_v001.hp")
        return original_nagisa_init(self, *args, **kwargs)
    
    # Apply monkey-patch
    Tagger.__init__ = patched_nagisa_init
except Exception:
    pass

# 2. Torch & Environment Fixes
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Fix for Windows Unicode Path Error (User name with Korean/Non-ASCII characters)
# Must be set BEFORE importing torch or stable_whisper
if os.name == 'nt':
    # C:\Users\Public is usually safe (ASCII) and writable
    public_dir = os.environ.get('PUBLIC', os.environ.get('SystemDrive', 'C:') + '\\Users\\Public')
    if os.path.exists(public_dir):
        safe_cache_dir = os.path.join(public_dir, 'torch_cache')
        os.environ['TORCH_HOME'] = safe_cache_dir
        os.makedirs(safe_cache_dir, exist_ok=True)
        print(f"Windows detected: Override TORCH_HOME to {safe_cache_dir} to avoid encoding errors.")


# Set PyTorch fallback for MPS to handle float64 operations. 
# MUST be set before importing torch or any library that imports torch.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Fix for SSL certificate verify failed error (common on Windows/macOS)
import ssl
import certifi
import numpy as np
os.environ['SSL_CERT_FILE'] = certifi.where()
# ssl._create_default_https_context must be a callable that returns a context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from google import genai
import re
import time
import json
import threading
from datetime import datetime
import subprocess
import torch
import shutil
import sys
import tqdm

class TqdmToStreamlit(tqdm.tqdm):
    """
    Redirects tqdm progress to Streamlit progress bar.
    """
    st_progress_bar = None
    st_status_text = None

    @classmethod
    def set_streamlit_elements(cls, progress_bar, status_text):
        cls.st_progress_bar = progress_bar
        cls.st_status_text = status_text

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Don't display standard stderr output
        self.file = sys.stdout 
        
    def update(self, n=1):
        super().update(n)
        if self.total and self.st_progress_bar:
            # Calculate percentage
            percentage = self.n / self.total
            # Clamp between 0.0 and 1.0 to avoid errors
            percentage = max(0.0, min(1.0, percentage))
            self.st_progress_bar.progress(percentage)
            
            if self.st_status_text:
                # Show percentage and elapsed time in status text if desired
                # format_dict = self.format_dict
                # elapsed_str = tqdm.format_interval(format_dict['elapsed'])
                self.st_status_text.text(f"Transcribing... {int(percentage*100)}%")

    def close(self):
        super().close()
        if self.st_progress_bar:
            self.st_progress_bar.empty()
        if self.st_status_text:
            self.st_status_text.empty()

def open_file_dialog():
    """Opens a file dialog to select a video file using a subprocess to avoid threading issues."""
    script = """
import tkinter as tk
from tkinter import filedialog
import os

try:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.ts"), ("All Files", "*.*")]
    )
    if file_path:
        print(file_path)
    root.destroy()
except:
    pass
"""
    try:
        # Run tkinter in a separate process to avoid main thread crashes in Streamlit
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception as e:
        print(f"File dialog error: {e}")
        return None

import tqdm
import platform

# Check if stable_whisper is available
try:
    import stable_whisper
    STABLE_WHISPER_AVAILABLE = True
except ImportError:
    STABLE_WHISPER_AVAILABLE = False

# Check if mlx-qwen3-asr is available (Mac Silicon)
try:
    import mlx_qwen3_asr
    MLX_QWEN3_AVAILABLE = True
except ImportError:
    MLX_QWEN3_AVAILABLE = False

# Check if qwen-asr is available (CUDA/PyTorch)
try:
    from qwen_asr import Qwen3ASRModel
    CUDA_QWEN3_AVAILABLE = True
except ImportError:
    CUDA_QWEN3_AVAILABLE = False

# ==========================================
# Core Logic: Gemini Translation
# ==========================================

# Rate limiting settings
MAX_REQUESTS_PER_MINUTE = 10
request_timestamps = []
request_lock = threading.Lock()

def wait_for_rate_limit():
    """Rate limit 체크 및 대기"""
    with request_lock:
        now = datetime.now()
        while request_timestamps and (now - request_timestamps[0]).total_seconds() >= 60:
            request_timestamps.pop(0)
        
        if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (now - request_timestamps[0]).total_seconds()
            if wait_time > 0:
                time.sleep(wait_time)
        
        request_timestamps.append(now)

def SRT_to_numbered_blocks(content_str):
    """Parses SRT content string into blocks."""
    lines = content_str.split('\n')
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            num = int(line)
            j = i + 2
            text_lines = []
            while j < len(lines) and lines[j].strip() != "":
                text_lines.append(lines[j].rstrip('\n'))
                j += 1
            block = f"{num}. " + "\n".join(text_lines)
            blocks.append(block)
            i = j
        else:
            i += 1
    return blocks

def SliceStringListForGPTRequest(originStrList):
    MAX_CHUNK_SIZE = 20 * 1024  
    resultList = []
    tempText = ''
    current_size = 0
    for block in originStrList:
        block_with_newline = block + '\n'
        block_size = len(block_with_newline.encode('utf-8'))
        if current_size > 0 and (current_size + block_size > MAX_CHUNK_SIZE):
            resultList.append(tempText)
            tempText = block_with_newline
            current_size = block_size
        else:
            tempText += block_with_newline
            current_size += block_size
    if tempText:
        resultList.append(tempText)
    return resultList

def ParseGeminiResultToDict(resultText):
    translatedDict = dict()
    for line in resultText.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\.\s*(.*)$", line)
        if match:
            num = int(match.group(1))
            txt = match.group(2)
            translatedDict[num] = txt + "\n"
    return translatedDict

def ApplyTranslationToSRT(original_content, translated_text):
    """Merges translated text into original SRT."""
    translatedDict = ParseGeminiResultToDict(translated_text)
    lines = original_content.split('\n')
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.isdigit():
            # 1. Append Number
            num = int(line)
            new_lines.append(lines[i] + '\n')
            
            # 2. Append Timestamp
            if i + 1 < len(lines):
                i += 1
                new_lines.append(lines[i] + '\n')
            
            # 3. Handle Text
            i += 1
            if num in translatedDict:
                # Consume original text lines until empty line
                while i < len(lines) and lines[i].strip() != "":
                    i += 1
                
                # Add translated text
                t_lines = translatedDict[num].strip().split('\n')
                for tl in t_lines:
                    new_lines.append(tl + '\n')
            else:
                # No translation, keep original text
                 while i < len(lines) and lines[i].strip() != "":
                    new_lines.append(lines[i] + '\n')
                    i += 1
            
            # 4. Handle Block Separator (Empty Line)
            # The loops above stop when lines[i] is empty (or OOB).
            # We need to append this empty line to separate blocks.
            if i < len(lines):
                 new_lines.append(lines[i] + '\n')

        else:
            # Not a block start (could be metadata or extra newlines), just keep it
            new_lines.append(lines[i] + '\n')
        
        i += 1
        
    return "".join(new_lines)

# ==========================================
# Core Logic: Audio/Whisper
# ==========================================

def extract_audio_array(video_path):
    """Extracts audio to numpy array using ffmpeg (16kHz mono) via stdout pipe."""
    command = [
        'ffmpeg', '-y', '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        '-f', 's16le', '-'  # Output to stdout
    ]
    try:
        # Increase buffer limit for large files if needed, but run usually handles it
        result = subprocess.run(command, capture_output=True, check=True)
        # Convert bytes to numpy array
        audio_data = np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        return audio_data, None
    except subprocess.CalledProcessError as e:
        return None, f"FFmpeg Error Output:\n{e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)}"
    except Exception as e:
        return None, f"FFmpeg Execution Error: {e}"

def extract_audio(video_path, audio_output_path):
    """Legacy file-based extraction (kept for fallback compatibility if needed)."""
    # ... (code unchanged or minimal keep)
    # Replaced by extract_audio_array in main logic
    pass 

def transcribe_with_qwen3(audio_data, model_size="1.7B", language="ja"):
    """
    Transcribes audio using Qwen3-ASR.
    Supports MLX (Mac Silicon) and PyTorch (CUDA/CPU).
    """
    import tempfile
    
    # 1. Determine model path
    hf_model = f"Qwen/Qwen3-ASR-{model_size}"
    
    # 2. Transcribe based on available backend
    if MLX_QWEN3_AVAILABLE and platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon Mac: Use MLX
        st.info(f"Using MLX backend for Qwen3-ASR ({model_size})")
        
        # Create temp file because mlx-qwen3-asr CLI/API often expects file paths
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import scipy.io.wavfile as wav
            # Ensure audio_data is 16kHz mono as per extract_audio_array
            wav.write(tmp.name, 16000, (audio_data * 32767).astype(np.int16))
            tmp_path = tmp.name
        
        try:
            # mlx_qwen3_asr.transcribe returns a dict with 'segments'
            result = mlx_qwen3_asr.transcribe(tmp_path, model=model_size)
            if isinstance(result, str): # Fallback if only text returned
                result = {"text": result, "segments": [{"start": 0, "end": len(audio_data)/16000, "text": result}]}
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
        return result
    
    elif CUDA_QWEN3_AVAILABLE:
        # CUDA or CPU: Use standard qwen-asr (PyTorch)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using PyTorch backend ({device}) for Qwen3-ASR ({model_size})")
        
        # In a real app, we might want to cache this model instance
        model = Qwen3ASRModel.from_pretrained(hf_model, device=device)
        
        # Transcribe (some versions take path, some take array)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import scipy.io.wavfile as wav
            wav.write(tmp.name, 16000, (audio_data * 32767).astype(np.int16))
            tmp_path = tmp.name
        try:
            result = model.transcribe(tmp_path)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
        return result
    else:
        st.error("Qwen3-ASR backend not available. Please install mlx-qwen3-asr or qwen-asr.")
        return None

def segments_to_srt(segments):
    """Converts a list of subtitle segments to SRT string."""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start = float(segment.get("start", 0))
        end = float(segment.get("end", 0))
        text = str(segment.get("text", "")).strip()
        
        if not text:
            continue
            
        # SRT Format: HH:MM:SS,mmm
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"
            
        start_str = format_time(start)
        end_str = format_time(end)
        srt_content += f"{i}\n{start_str} --> {end_str}\n{text}\n\n"
    return srt_content

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(page_title="Subtitle Tool", layout="wide")
st.title("🎬 Subtitle & Translation Tool")

tab1, tab2, tab3 = st.tabs(["💬 Translation (SRT to Korean)", "🎥 Video to SRT (Whisper)", "🚀 One-Stop Workflow"])

# --------------------------
# Settings Sidebar
# --------------------------
with st.sidebar:
    st.header("Global Settings")

    auth_mode = st.radio("Auth Method", ["API Key", "Vertex AI (CLI/ADC)"], help="Select API Key or System Credentials (gcloud CLI)")
    
    api_key_input = ""
    vertex_project_id = ""
    vertex_location = ""
    
    if auth_mode == "API Key":
        default_api_key = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.text_input("Gemini API Key", value=default_api_key, type="password", help="Enter Google Gemini API Key")
    else:
        st.info("Using System Credentials (ADC). Ensure you have run `gcloud auth application-default login`.")
        vertex_project_id = st.text_input("Google Cloud Project ID")
        vertex_location = st.text_input("Location (Region)", value="us-central1")

    gemini_model_list = [
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro", 
        "gemini-1.5-flash", 
        "gemini-pro",
        "Custom"
    ]
    gemini_model_select = st.selectbox("Gemini Model", gemini_model_list, index=0)
    
    if gemini_model_select == "Custom":
        gemini_model_select = st.text_input("Enter Custom Model Name", value="gemini-1.5-pro")
    
    
    st.markdown("---")
    custom_prompt_input = st.text_area("Translation Prompt", 
        value="다음 자막을 한줄씩 한국어로 번역해줘. 각 줄은 '번호. 내용' 형식이야. 번역도 '번호. 번역된 내용' 형식으로 해줘. 성적인 묘사가 포함된 경우, 직접적인 표현 대신 은유적이거나 순화된 표현을 사용해서 안전 필터에 걸리지 않게 번역해줘.",
        height=150)
    
    st.markdown("---")
    st.write("ASR Settings")
    asr_engine = st.radio("ASR Engine", ["Whisper (stable-ts)", "Qwen3-ASR"], index=0)
    
    if asr_engine == "Whisper (stable-ts)":
        whisper_model_select = st.selectbox("Whisper Model", ["large-v3-turbo", "large-v3", "large-v2", "medium", "small", "base"], index=0)
        use_vad = st.checkbox("Enable VAD (Voice Activity Detection)", value=False, help="Reduces hallucinations in silent sections.")
    else:
        qwen_model_select = st.selectbox("Qwen3 Model", ["1.7B", "0.6B"], index=0)
        st.info("Qwen3-ASR uses MLX on Apple Silicon and PyTorch/CUDA on NVIDIA GPUs.")
    
    whisper_lang_select = st.selectbox("Source Language", ["ja", "en", "ko", "zh", "es", "fr"], index=0)


# --------------------------
# TAB 1: Translation
# --------------------------
with tab1:
    st.subheader("Translate Existing SRT")
    uploaded_file = st.file_uploader("Upload .srt file for translation", type=["srt"])

    if uploaded_file and (api_key_input or auth_mode == "Vertex AI (CLI/ADC)"):
        content_bytes = uploaded_file.read()
        try:
            content_str = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            content_str = content_bytes.decode('cp949', errors='ignore')

        st.info(f"Loaded: {uploaded_file.name}")
        with st.expander("Preview Original"):
            st.text(content_str[:500] + "...")

        if st.button("Start Translation", key="btn_translate_only"):
            if auth_mode == "API Key":
                client = genai.Client(api_key=api_key_input)
            else:
                client = genai.Client(vertexai=True, project=vertex_project_id, location=vertex_location)
            
            # 안전 설정: 성인 콘텐츠 등 모든 차단 필터 해제
            safety_config = {
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            }

            st.write("Parsing SRT...")
            status_text = st.empty()
            
            originStrList = SRT_to_numbered_blocks(content_str)
            slicedStrList = SliceStringListForGPTRequest(originStrList)
            
            st.write(f"Total parts: {len(slicedStrList)}")
            progress_bar = st.progress(0)
            
            full_result_text = ""
            
            for idx, chunk in enumerate(slicedStrList):
                wait_for_rate_limit()
                status_text.text(f"Translating part {idx + 1}/{len(slicedStrList)}...")
                prompt = f"{custom_prompt_input}\n\n{chunk}"
                
                try:
                    response = client.models.generate_content(
                        model=gemini_model_select,
                        contents=prompt,
                        config=safety_config
                    )
                    if response.text:
                        full_result_text += response.text + "\n"
                    else:
                        raise ValueError("Empty response")
                except Exception as e:
                    st.error(f"Error {idx+1}: {e}. Using original text.")
                    full_result_text += chunk + "\n"
                    time.sleep(2)
                
                progress_bar.progress((idx + 1) / len(slicedStrList))
                
            status_text.text("Merging...")
            final_srt = ApplyTranslationToSRT(content_str, full_result_text)
            
            st.success("Translation Complete!")
            st.download_button("Download SRT", final_srt, f"translated_{uploaded_file.name}", "text/plain")

    elif not (api_key_input or auth_mode == "Vertex AI (CLI/ADC)"):
        st.info("Enter Gemini API Key or configure Vertex AI in sidebar to translate.")

# --------------------------
# TAB 2: Video to SRT
# --------------------------
with tab2:
    st.subheader("Extract Subtitles from Video (Whisper Only)")
    
    if not STABLE_WHISPER_AVAILABLE:
        st.error("Please install stable-ts: `pip install stable-ts`")
    else:
        # Input Method
        input_method = st.radio("Input Method", ["Upload File", "Local File Path"], key="tab2_input")
        
        target_video_path = None
        
        if input_method == "Upload File":
            vid_file = st.file_uploader("Upload Video", type=["mp4", "mkv", "avi", "mov", "ts"], key="tab2_upload")
            if vid_file:
                # Save to temp
                with st.spinner("Saving uploaded file to temporary storage..."):
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    target_video_path = os.path.join(temp_dir, vid_file.name)
                    with open(target_video_path, "wb") as f:
                        f.write(vid_file.getbuffer())
                    st.success(f"File stored at: {target_video_path}")
        else:
            if "tab2_file_path" not in st.session_state:
                st.session_state.tab2_file_path = ""

            col_path, col_btn = st.columns([5, 1])
            with col_btn:
                if st.button("📁", key="tab2_browse"):
                    selected_file = open_file_dialog()
                    if selected_file:
                        st.session_state.tab2_file_path = selected_file
                        st.rerun()

            with col_path:
                local_path = st.text_input("Enter Absolute File Path", key="tab2_file_path")
            
            if local_path and os.path.exists(local_path):
                target_video_path = local_path
                st.success("File found!")
            elif local_path:
                st.error("File not found.")

        if st.button("Generate SRT", key="btn_whisper_only") and target_video_path:
            with st.status("Processing...", expanded=True) as status:
                try:
                    # 1. Extract Audio
                    status.write("Extracting Audio with FFmpeg...")
                    if target_video_path:
                        # 2. Extract Audio to Memory
                        status.write("Extracting Audio to Memory (Bypass file issues)...")
                        audio_array, error_msg = extract_audio_array(target_video_path)
                        
                        if audio_array is not None:
                            # 2. Load Model
                            status.write(f"Loading Whisper Model ({whisper_model_select})...")
                            device = "cpu"
                            if torch.cuda.is_available(): device = "cuda"
                            
                            model = stable_whisper.load_model(whisper_model_select, device=device)
                            
                            # 3. Transcribe
                            status.write("Transcribing...")
                            
                            srt_content = ""
                            if asr_engine == "Whisper (stable-ts)":
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Redirect tqdm to streamlit
                                TqdmToStreamlit.set_streamlit_elements(progress_bar, status_text)
                                
                                # Save original tqdm
                                original_tqdm = tqdm.tqdm
                                # Monkey patch
                                tqdm.tqdm = TqdmToStreamlit
                                
                                try:
                                    model = stable_whisper.load_model(whisper_model_select, device=device)
                                    # Pass numpy array directly
                                    result = model.transcribe(
                                        audio_array, 
                                        language=whisper_lang_select, 
                                        regroup=True,
                                        fp16=False,
                                        vad=use_vad,
                                        verbose=None
                                    )
                                    # 4. Save
                                    status.write("Saving SRT...")
                                    temp_srt_path = "temp_output.srt"
                                    result.to_srt_vtt(temp_srt_path, segment_level=True, word_level=False)
                                    with open(temp_srt_path, "r", encoding="utf-8") as f:
                                        srt_content = f.read()
                                    if os.path.exists(temp_srt_path): os.remove(temp_srt_path)
                                finally:
                                    # Restore original tqdm
                                    tqdm.tqdm = original_tqdm
                                    progress_bar.empty()
                                    status_text.empty()
                            else:
                                # Qwen3 ASR
                                result = transcribe_with_qwen3(audio_array, model_size=qwen_model_select, language=whisper_lang_select)
                                if result and "segments" in result:
                                    status.write("Saving SRT...")
                                    srt_content = segments_to_srt(result["segments"])
                                else:
                                    st.error("Qwen3 transcription failed.")
                                    srt_content = None

                            if srt_content:
                                base_name = os.path.basename(target_video_path)
                                srt_name = os.path.splitext(base_name)[0] + ".srt"
                                
                                st.balloons()
                                status.update(label="Complete!", state="complete", expanded=False)
                                
                                st.subheader("Result")
                                st.download_button("Download Generated SRT", srt_content, srt_name, "text/plain")
                                st.text_area("SRT Content", srt_content, height=300)
                            else:
                                status.update(label="Transcription failed.", state="error")
                        
                    else:
                        status.update(label=f"Failed at Audio Extraction: {error_msg}", state="error")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    status.update(label="Error Occurred", state="error")

# --------------------------
# TAB 3: One-Stop Workflow
# --------------------------
with tab3:
    st.subheader("🚀 One-Stop: Video -> SRT -> Translation")
    st.info("This workflow extracts subtitles from a video using Whisper and then translates them using Gemini - all in one go.")
    
    if not STABLE_WHISPER_AVAILABLE:
        st.error("Please install stable-ts: `pip install stable-ts`")
    elif not (api_key_input or auth_mode == "Vertex AI (CLI/ADC)"):
        st.warning("Please enter your Gemini API Key or configure Vertex AI in the sidebar.")
    else:
        # Input Method
        input_method_os = st.radio("Input Method", ["Upload File", "Local File Path"], key="tab3_input")
        
        target_video_path_os = None
        
        if input_method_os == "Upload File":
            vid_file_os = st.file_uploader("Upload Video", type=["mp4", "mkv", "avi", "mov", "ts"], key="tab3_upload")
            if vid_file_os:
                with st.spinner("Saving uploaded file to temporary storage..."):
                    temp_dir = "temp_uploads_os"
                    os.makedirs(temp_dir, exist_ok=True)
                    target_video_path_os = os.path.join(temp_dir, vid_file_os.name)
                    with open(target_video_path_os, "wb") as f:
                        f.write(vid_file_os.getbuffer())
                    st.success(f"File stored at: {target_video_path_os}")
        else:
            if "tab3_file_path" not in st.session_state:
                st.session_state.tab3_file_path = ""

            col_path_os, col_btn_os = st.columns([5, 1])
            with col_btn_os:
                if st.button("📁", key="tab3_browse"):
                    selected_file_os = open_file_dialog()
                    if selected_file_os:
                        st.session_state.tab3_file_path = selected_file_os
                        st.rerun()

            with col_path_os:
                local_path_os = st.text_input("Enter Absolute File Path", key="tab3_file_path")

            if local_path_os and os.path.exists(local_path_os):
                target_video_path_os = local_path_os
                st.success("File found!")
            elif local_path_os:
                st.error("File not found.")

        if st.button("Start Full Workflow", key="btn_onestop") and target_video_path_os:
             with st.status("Running One-Stop Workflow...", expanded=True) as status:
                try:
                    # --- STEP 1: Audio Extraction ---
                    # --- STEP 1: Audio Extraction ---
                    status.write("Step 1: Extracting Audio to Memory...")
                    audio_array_os, error_msg_os = extract_audio_array(target_video_path_os)
                    
                    if audio_array_os is None:
                        status.update(label=f"Failed at Audio Extraction: {error_msg_os}", state="error")
                        st.stop()
                    
                    # --- STEP 2: ASR Transcription ---
                    status.write(f"Step 2: Transcribing Audio ({asr_engine})...")
                    
                    original_srt_content = ""
                    if asr_engine == "Whisper (stable-ts)":
                        progress_bar_os = st.progress(0)
                        status_text_os = st.empty()
                        
                        device = "cpu"
                        if torch.cuda.is_available(): device = "cuda"
                        
                        # Redirect tqdm to streamlit
                        TqdmToStreamlit.set_streamlit_elements(progress_bar_os, status_text_os)
                        original_tqdm = tqdm.tqdm
                        tqdm.tqdm = TqdmToStreamlit
                        
                        try:
                            model_whisper = stable_whisper.load_model(whisper_model_select, device=device)
                            # Pass numpy array directly
                            result_whisper = model_whisper.transcribe(
                                audio_array_os, 
                                language=whisper_lang_select, 
                                regroup=True,
                                fp16=False,
                                vad=use_vad
                            )
                            # Save intermediate SRT to string
                            temp_srt_path_os = "temp_output_os.srt"
                            result_whisper.to_srt_vtt(temp_srt_path_os, segment_level=True, word_level=False)
                            
                            with open(temp_srt_path_os, "r", encoding="utf-8") as f:
                                original_srt_content = f.read()
                            if os.path.exists(temp_srt_path_os): os.remove(temp_srt_path_os)
                        finally:
                             tqdm.tqdm = original_tqdm
                             progress_bar_os.empty()
                             status_text_os.empty()
                    else:
                        # Qwen3 ASR
                        result_qwen = transcribe_with_qwen3(audio_array_os, model_size=qwen_model_select, language=whisper_lang_select)
                        if result_qwen and "segments" in result_qwen:
                            original_srt_content = segments_to_srt(result_qwen["segments"])
                        else:
                            status.update(label="Qwen3 Transcription failed.", state="error")
                            st.stop()
                        
                    status.write("Transcription Complete. Preparing translation...")
                    
                    # --- STEP 3: Gemini Translation ---
                    status.write(f"Step 3: Translating Subtitles (Gemini {gemini_model_select})...")
                    
                    if auth_mode == "API Key":
                        client = genai.Client(api_key=api_key_input)
                    else:
                        client = genai.Client(vertexai=True, project=vertex_project_id, location=vertex_location)

                    # 안전 설정: 성인 콘텐츠 등 모든 차단 필터 해제
                    safety_config = {
                        "safety_settings": [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ]
                    }

                    originStrList = SRT_to_numbered_blocks(original_srt_content)
                    slicedStrList = SliceStringListForGPTRequest(originStrList)
                    
                    full_result_text = ""
                    
                    # Create a progress bar inside the status container? No, st.progress works outside
                    # We can't nest st.progress cleanly inside status, so we'll just log
                    
                    total_chunks = len(slicedStrList)
                    for idx, chunk in enumerate(slicedStrList):
                        wait_for_rate_limit()
                        # Update status text
                        status.write(f"Translating chunk {idx + 1}/{total_chunks}...")
                        
                        prompt = f"{custom_prompt_input}\n\n{chunk}"
                        try:
                            response = client.models.generate_content(
                                model=gemini_model_select,
                                contents=prompt,
                                config=safety_config
                            )
                            if response.text:
                                full_result_text += response.text + "\n"
                            else:
                                raise ValueError("Empty response")
                        except Exception as e:
                            status.write(f"Error on chunk {idx+1}: {e}. Using original text.")
                            full_result_text += chunk + "\n"
                            time.sleep(2)
                    
                    status.write("Translation logic finished. Merging...")
                    
                    # --- STEP 4: Merge ---
                    final_translated_srt = ApplyTranslationToSRT(original_srt_content, full_result_text)
                    
                    # Cleanup
                    if os.path.exists(temp_srt_path_os): os.remove(temp_srt_path_os)
                    
                    # --- SAVE TO SESSION STATE ---
                    st.session_state['os_original_srt'] = original_srt_content
                    st.session_state['os_translated_srt'] = final_translated_srt
                    st.session_state['os_video_name'] = os.path.basename(target_video_path_os)
                    st.session_state['os_lang_code'] = whisper_lang_select

                    status.update(label="Workflow Complete!", state="complete", expanded=False)
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Critical Error: {e}")
                    status.update(label="Workflow Failed", state="error")
        
        # --- Output (Persistent) ---
        if 'os_original_srt' in st.session_state:
            st.divider()
            st.subheader("Workflow Results")
            
            col_dl1, col_dl2 = st.columns(2)
            
            base_name = st.session_state['os_video_name']
            lang_code = st.session_state.get('os_lang_code', 'src')
            original_srt_name = os.path.splitext(base_name)[0] + f".{lang_code}.srt"
            translated_srt_name = os.path.splitext(base_name)[0] + ".ko.srt"
            
            with col_dl1:
                st.download_button("Download Original SRT (Whisper)", st.session_state['os_original_srt'], original_srt_name, "text/plain")
                st.text_area("Original Content", st.session_state['os_original_srt'], height=200)
                
            with col_dl2:
                st.download_button("Download Translated SRT (Gemini)", st.session_state['os_translated_srt'], translated_srt_name, "text/plain")
                st.text_area("Translated Content", st.session_state['os_translated_srt'], height=200)
            
            if st.button("Clear Results", key="btn_clear_os"):
                del st.session_state['os_original_srt']
                del st.session_state['os_translated_srt']
                del st.session_state['os_video_name']
                st.rerun()
