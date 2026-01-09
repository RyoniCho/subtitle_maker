import mlx_whisper 
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

#https://github.com/awilliamson/whisper-mlx




parser=argparse.ArgumentParser(description="Subscription Maker")


parser.add_argument("--f",required=True)
parser.add_argument("--only_translate",default="false")
parser.add_argument("--lan",default="ja")
parser.add_argument("--model",default="mlx-community/whisper-large-v3-mlx")

args=parser.parse_args()

file_name=args.f
if args.only_translate == "true":
    print("Only Translate..")
    #translation_deepL.translationFile(file_name)
else:
    def optimize_audio_for_silicon(input_file):
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        # 오디오 스트림만 추출하고 필요한 설정 변경
        # -c:a copy: 오디오 코덱을 그대로 유지
        # -af 'aformat=sample_rates=16000,aresample=16000,aformat=channel_layouts=mono': 
        # 오디오 필터를 사용하여 샘플레이트와 채널 변경
        # -vn: 비디오 스트림 제거
        command = [
            'ffmpeg',
            # Apple Silicon 최적화를 위한 threads 설정
            '-threads', 'auto',     # 자동으로 최적의 스레드 수 설정
            '-hwaccel', 'auto',     # 사용 가능한 최적의 하드웨어 가속 자동 선택
            '-i', input_file,
            '-vn',                  # 비디오 스트림 제거
            '-ar', '16000',         # 샘플레이트를 16kHz로 설정
            '-ac', '1',            # 모노 채널
            '-c:a', 'pcm_s16le',   # WAV용 16-bit PCM 코덱
            '-f', 'wav',           # WAV 포맷
            '-v', 'info',          # 정보 로그 출력
            '-stats',              # 진행 상태 표시
            temp_path
        ]
        
        try:
            # stdout=None, stderr=None으로 설정하여 터미널에 직접 출력
            print("Starting audio extraction and optimization...")
            subprocess.run(command, check=True, stdout=None, stderr=None)
            print("Audio optimization completed")
            return temp_path
        except subprocess.CalledProcessError as e:
            print(f"Error during audio optimization: {e}")
            return input_file

    print("Whisper-MLX: Start to transcribe...")
    
    # 오디오 최적화
    print("Optimizing audio for Silicon...")
    optimized_audio = optimize_audio_for_silicon(file_name)
    

    
    # 트랜스크립션 실행
    print("Transcribing with Whisper-MLX...")
    result = mlx_whisper.transcribe(
        optimized_audio,  # 오디오 파일 경로
        path_or_hf_repo=f"{args.model}",  # 모델 경로
        language=args.lan,
        verbose=True,  # 로그 출력 옵션
    )

    print(f"optimezed_audio: {optimized_audio}")
    print(f"file_name: {file_name}")
    
    # 임시 파일 삭제
    
    os.unlink(optimized_audio)
   
    

    print("transcribe end")

    fileNameWithoutExtension = os.path.splitext(os.path.basename(file_name))[0]
    resultSrt = f"{fileNameWithoutExtension}.srt"

    # 디버그: 결과 구조 출력
    print("Transcription result structure:", result.keys() if isinstance(result, dict) else type(result))
    if isinstance(result, dict) and 'segments' in result:
        print("First segment structure:", result['segments'][0] if result['segments'] else "No segments")
    
    # SRT 포맷으로 변환
    with open(resultSrt, 'w', encoding='utf-8') as f:
        try:
            if isinstance(result, dict) and 'segments' in result:
                for i, segment in enumerate(result["segments"], 1):
                    try:
                        start = float(segment["start"])
                        end = float(segment["end"])
                        text = str(segment["text"]).strip()
                        
                        if not text:  # 빈 텍스트는 건너뛰기
                            continue
                        
                        # SRT 시간 포맷으로 변환 (HH:MM:SS,mmm)
                        start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                        end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
                        
                        # SRT 항목 작성
                        f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"Warning: Error processing segment {i}: {e}")
                        print(f"Segment data: {segment}")
                        continue
            else:
                print("Error: Unexpected transcription result format")
                print("Result structure:", result.keys() if isinstance(result, dict) else type(result))
                if isinstance(result, dict):
                    print("Available keys:", result.keys())
                    if 'text' in result:
                        print("Contains 'text' field. This might be a single transcription without segments.")
                raise ValueError("Unexpected transcription result format")
        except Exception as e:
            print(f"Error while processing transcription: {e}")
            raise

    print("Whisper: Export SRT Finished..")

    #print("Start Translate using DeepL..")

    #translation_deepL.translationFile(resultSrt)

    print("Finished")


