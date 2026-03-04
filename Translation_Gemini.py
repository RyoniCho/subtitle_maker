import google.generativeai as genai
from decimal import Decimal
import traceback
import sys
import os
import json
import random
import re
import shutil
import argparse
import time
from datetime import datetime
import threading
from queue import Queue
import logging
from dotenv import load_dotenv

# ALTS 관련 로그 숨기기
logging.getLogger('google.auth._default').setLevel(logging.ERROR)
os.environ['GRPC_TRACE'] = 'none'
os.environ['GRPC_VERBOSITY'] = 'none'

# Gemini API 키 설정
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# Rate limiting 설정
MAX_REQUESTS_PER_MINUTE = 10  # Gemini API의 분당 최대 요청 수
request_timestamps = []
request_lock = threading.Lock()

def wait_for_rate_limit():
    """Rate limit 체크 및 대기"""
    with request_lock:
        now = datetime.now()
        # 1분이 지난 타임스탬프 제거
        while request_timestamps and (now - request_timestamps[0]).total_seconds() >= 60:
            request_timestamps.pop(0)
        
        # 현재 분당 요청 수가 제한을 초과하면 대기
        if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (now - request_timestamps[0]).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting for {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # 현재 요청 시간 기록
        request_timestamps.append(now)

def Run_Gemini_Translation(inputStrs, start_chunk=0):
    endChunkIndex = GetEndChunkIndexFromResumeInfo()
    resultText = ''
    requestCount = 0
    currentChunkIndex = 0
    geminiRequestFailed = False
    try:
        model = genai.GenerativeModel('gemini-3.1-pro-preview')
        
        # 안전 설정: 성인 콘텐츠 등 모든 차단 필터 해제
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        for chunkIndex, chunk in enumerate(inputStrs):
            # start_chunk 인덱스부터 시작
            if chunkIndex < max(endChunkIndex + 1, start_chunk):
                isResumeProcess = True
                continue

            # Rate limit 체크
            wait_for_rate_limit()

            # chunk는 이미 번호+내용 형식으로 되어 있음
            prompt = f"다음 자막을 한줄씩 한국어로 번역해줘. 각 줄은 '번호. 내용' 형식이야. 번역도 '번호. 번역된 내용' 형식으로 해줘. 성적인 묘사가 포함된 경우, 직접적인 표현 대신 은유적이거나 순화된 표현을 사용해서 안전 필터에 걸리지 않게 번역해줘.\n\n{chunk}"

            try:
                response = model.generate_content(prompt, safety_settings=safety_settings)
                
                if response.text:
                    print(f"Gemini response for chunk {chunkIndex}:\n {response.text}") 
                    resultText += response.text + "\n"
                    requestCount += 1
                    currentChunkIndex = chunkIndex
                    print(f"Successfully translated chunk {chunkIndex + 1}/{len(inputStrs)}")
                else:
                    raise ValueError("Empty response (likely blocked)")
            except Exception as e:
                print(f"Error translating chunk {chunkIndex}: {e}")
                print(f"Safety block triggered. Using original text for chunk {chunkIndex}.")
                resultText += chunk + "\n"
                requestCount += 1
                currentChunkIndex = chunkIndex
                time.sleep(2)

    except Exception as e:
        print(f"Gemini request failed: {e}")
        geminiRequestFailed = True

    if requestCount > 0:
        if geminiRequestFailed:
            SaveResumeInfo(currentChunkIndex)
        else:
            RemoveResumeInfo()

        resultText = resultText.encode('utf-8').decode('utf-8')
    else:
        RemoveResumeInfo()
        print('Gemini Response is null. exit')
        sys.exit(-1)

    return resultText

def SaveGPTResponsePlainText(subFileName, resultText):
    saveFileName = f'{subFileName}.txt'
    fileMode = 'w'

    if isResumeProcess and os.path.exists(saveFileName):
        fileMode = 'a'

    with open(saveFileName, fileMode, encoding='utf-8') as f:
        f.write(resultText)

def ParseGeminiResultToDict(resultText):
    translatedDict = dict()
    for line in resultText.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 번호. 번역된 내용 형식
        match = re.match(r"^(\d+)\.\s*(.*)$", line)
        if match:
            num = int(match.group(1))
            txt = match.group(2)
            translatedDict[num] = txt
    return translatedDict

def SRT_to_numbered_blocks(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            num = int(lines[i].strip())
            # 시간 줄 건너뜀
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

def SaveSubscriptionFile(subFileName, subFileExtension, resultText):
    translatedDict = ParseGeminiResultToDict(resultText)
    originSubFile = f"{subFileName}.{subFileExtension}"
    writeSubFile = f"{subFileName}_translated.{subFileExtension}"
    with open(originSubFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            num = int(line)
            if num in translatedDict:
                # 시간 줄 건너뜀
                j = i + 2
                if j < len(lines):
                    # 번역문이 여러 줄일 경우 SRT에 맞게 분리
                    translated_lines = translatedDict[num].split('\n')
                    k = 0
                    # 기존 텍스트 줄을 번역문으로 교체, 남는 줄은 비움
                    while j + k < len(lines) and lines[j + k].strip() != "" and k < len(translated_lines):
                        lines[j + k] = translated_lines[k] + '\n'
                        k += 1
                    # 번역문이 더 길면 추가
                    while k < len(translated_lines):
                        lines.insert(j + k, translated_lines[k] + '\n')
                        k += 1
                    # 기존 텍스트 줄이 더 길면 남는 줄 비움
                    m = j + k
                    while m < len(lines) and lines[m].strip() != "":
                        lines[m] = "\n"
                        m += 1
        i += 1
    with open(writeSubFile, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def FilterString(text):
    if "(" not in text or ")" not in text:
        return text

    start = text.find("(") + 1
    end = text.find(")", start)
    inner_string = text[start:end]
    return inner_string



def FilterString(text):
    if "(" not in text or ")" not in text:
        return text

    start = text.find("(") + 1
    end = text.find(")", start)
    inner_string = text[start:end]
    return inner_string

def ReadOriginSubscription(fileName, fileExtension):
    spacing = 4
    currentIndex = 2
    contentsStr = ''
    

    contents = contentsStr.split('\n')
    del contents[-1]

    print("Total Sentences", len(contents))
    return contents

def ReadInvalidTranslationList():
    with open(f"InvalidTranslation_{SUBSCRIPT_FILE_NAME}.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 1:
            return None

        for line in lines:
            lineNum = line.split(":")[0].strip()
            originTextTable[lineNum] = line
        return lines

def SliceStringListForGPTRequest(originStrList):
    MAX_CHUNK_SIZE = 20 * 1024  # 50KB
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
    print(f"Total chunks: {len(resultList)}")
    for i, chunk in enumerate(resultList):
        print(f"Chunk {i+1} size: {len(chunk.encode('utf-8'))} bytes")
    with open('sliced_text.txt', 'w', encoding='utf-8') as f:
        f.writelines(resultList)
    return resultList

def ValidateTranslatedText(texts):
    sentenceList = texts.split('\n')
    output = ''

    for s in sentenceList:
        output += FixSentenceNoCariageReturn(s) + "\n"

    output = output.replace("\n(", "(")
    return output

def FixSentenceNoCariageReturn(sentence):
    outputStr = sentence
    colonCount = 0

    for char in sentence:
        if char == ':':
            colonCount += 1

    if colonCount <= 1:
        return outputStr

    for idx, char in enumerate(sentence):
        if idx > 2:
            if char.isdigit():
                if not sentence[idx-1].isdigit() and sentence[idx-1] != '\n':
                    print(idx, char)
                    outputStr = f"{outputStr[:idx]}\n{char}{outputStr[idx + 1:]}"

    return outputStr

def RemoveSubscriptionNumber(sentence):
    res = filter(lambda c: not c.isdigit(), sentence)
    res = "".join(res).replace(":", "")
    return res

def SaveResumeInfo(endChunkIndex):
    if not os.path.exists(RESUME_INFO_FILE_NAME):
        with open(RESUME_INFO_FILE_NAME, 'w', encoding='utf-8') as f:
            resumeInfoJson = dict()
            resumeInfoJson[SUBSCRIPT_FILE_NAME] = dict()
            resumeInfoJson[SUBSCRIPT_FILE_NAME]["EndChunkIndex"] = endChunkIndex
            f.write(json.dumps(resumeInfoJson))
    else:
        resumeInfo = LoadResumeInfo()
        if resumeInfo is not None:
            if SUBSCRIPT_FILE_NAME in resumeInfo:
                resumeInfo[SUBSCRIPT_FILE_NAME]["EndChunkIndex"] = endChunkIndex
            else:
                resumeInfo[SUBSCRIPT_FILE_NAME] = dict()
                resumeInfo[SUBSCRIPT_FILE_NAME]["EndChunkIndex"] = endChunkIndex

            with open(RESUME_INFO_FILE_NAME, 'w', encoding='utf-8') as f:
                json.dump(resumeInfo, f)

def GetEndChunkIndexFromResumeInfo():
    endChunkIndex = -1
    resumeInfo = LoadResumeInfo()
    if resumeInfo is not None:
        if SUBSCRIPT_FILE_NAME in resumeInfo:
            endChunkIndex = resumeInfo[SUBSCRIPT_FILE_NAME]["EndChunkIndex"]
    return endChunkIndex

def LoadResumeInfo():
    if os.path.exists(RESUME_INFO_FILE_NAME):
        with open(RESUME_INFO_FILE_NAME, 'r', encoding='utf-8') as f:
            jsonStr = f.read()
            jsonObj = json.loads(jsonStr)
            return jsonObj
    return None

def RemoveResumeInfo():
    resumeInfo = LoadResumeInfo()
    if resumeInfo is not None:
        if SUBSCRIPT_FILE_NAME in resumeInfo:
            del resumeInfo[SUBSCRIPT_FILE_NAME]
            with open(RESUME_INFO_FILE_NAME, 'w', encoding='utf-8') as f:
                json.dump(resumeInfo, f)

def CheckNeedToRecoverValdation():
    if os.path.exists(f"InvalidTranslation_{SUBSCRIPT_FILE_NAME}.txt"):
        userSelect = input("====== Need To Recover invalid Translation=====\nRecover?(y/n)")
        if userSelect != "y":
            return False
        return True
    return False

SUBSCRIPT_FILE_NAME = ""
SUBSCRIPT_FILE_EXTENSION = "srt"
RESUME_INFO_FILE_NAME = "./resume_info.json"
isResumeProcess = False
useGemini = True
isRecoverInvalid = False
validateTable = dict()
originTextTable = dict()

def Run_Translation():
    global isRecoverInvalid
    isRecoverInvalid = CheckNeedToRecoverValdation()

    if args.from_txt:
        saveFileName = f'{SUBSCRIPT_FILE_NAME}.txt'
        with open(saveFileName, 'r', encoding='utf-8') as f:
            translatedFullStr = f.read()
        SaveSubscriptionFile(SUBSCRIPT_FILE_NAME, SUBSCRIPT_FILE_EXTENSION, translatedFullStr)
        return

    if isRecoverInvalid:
        originStrList = ReadInvalidTranslationList()
        if originStrList is None:
            print("Read invalid list is null")
            return
    else:
        # SRT를 번호별 블록(여러 줄 포함)으로 파싱
        originStrList = SRT_to_numbered_blocks(f"{SUBSCRIPT_FILE_NAME}.{SUBSCRIPT_FILE_EXTENSION}")

    slicedStrList = SliceStringListForGPTRequest(originStrList)
    if useGemini:
        translatedFullStr = Run_Gemini_Translation(slicedStrList, start_chunk=args.start_chunk)
        SaveGPTResponsePlainText(SUBSCRIPT_FILE_NAME, translatedFullStr)
    else:
        saveFileName = f'{SUBSCRIPT_FILE_NAME}.txt'
        with open(saveFileName, 'r', encoding='utf-8') as f:
            translatedFullStr = f.read()

    SaveSubscriptionFile(SUBSCRIPT_FILE_NAME, SUBSCRIPT_FILE_EXTENSION, translatedFullStr)

parser = argparse.ArgumentParser(description="Translation-Gemini")
parser.add_argument("--f", required=True)
parser.add_argument("--start_chunk", type=int, default=0, help="시작할 청크 인덱스 (0부터 시작)")
parser.add_argument("--from_txt", action='store_true', help="기존 txt 파일로만 SRT 변환 (Gemini 요청 생략)")
args = parser.parse_args()

if __name__ == "__main__":
    try:
        movieFile = args.f
        name, extension = os.path.splitext(movieFile)
        SUBSCRIPT_FILE_NAME = name

        if not os.path.exists(f"{SUBSCRIPT_FILE_NAME}.{SUBSCRIPT_FILE_EXTENSION}"):
            shutil.copy(f"./data_proc/{SUBSCRIPT_FILE_NAME}.{SUBSCRIPT_FILE_EXTENSION}", 
                       f"./{SUBSCRIPT_FILE_NAME}.{SUBSCRIPT_FILE_EXTENSION}")

        Run_Translation()

    except Exception as e:
        print(e)
        print(traceback.format_exc())
