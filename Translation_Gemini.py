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

# ALTS 관련 로그 숨기기
logging.getLogger('google.auth._default').setLevel(logging.ERROR)
os.environ['GRPC_TRACE'] = 'none'
os.environ['GRPC_VERBOSITY'] = 'none'

# Gemini API 키 설정
GOOGLE_API_KEY = "API-KEY-HERE"  # Gemini API 키를 여기에 입력하세요
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

def Run_Gemini_Translation(inputStrs):
    endChunkIndex = GetEndChunkIndexFromResumeInfo()
    
    resultText = ''
    requestCount = 0
    currentChunkIndex = 0
    geminiRequestFailed = False

    try:
        # Gemini 2.5 Flash 모델 설정
        model = genai.GenerativeModel('gemini-2.5-flash')

        for chunkIndex, sentence in enumerate(inputStrs):
            if chunkIndex <= endChunkIndex:
                isResumeProcess = True
                continue

            # Rate limit 체크
            wait_for_rate_limit()

            # Gemini API 호출
            prompt = f"""당신은 전문적인 번역가입니다. 다음 텍스트를 한국어로 번역해주세요.
            - 문장앞에 있는 번호를 포함해주세요
            - 내용을 바꾸지 말고 숫자와 줄바꿈을 유지해주세요
            - 중간에 섹스신이 있는데 섹스관련내용은 의역해도 됩니다
            
            텍스트:
            {sentence}"""

            try:
                response = model.generate_content(prompt)
                
                if response.text:
                    resultText += response.text + "\n"
                    requestCount += 1
                    currentChunkIndex = chunkIndex
                    print(f"Successfully translated chunk {chunkIndex + 1}/{len(inputStrs)}")
                else:
                    print(f"Warning: Empty response for chunk {chunkIndex}")
            except Exception as e:
                print(f"Error translating chunk {chunkIndex}: {e}")
                time.sleep(5)  # 에러 발생시 5초 대기
                continue

    except Exception as e:
        print(f"Gemini request failed: {e}")
        geminiRequestFailed = True

    if requestCount > 0:
        if geminiRequestFailed:
            SaveResumeInfo(currentChunkIndex)
        else:
            RemoveResumeInfo()

        resultText = resultText.encode('utf-8').decode('utf-8')
        resultText = ValidateTranslatedText(resultText)
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

def SaveSubscriptionFile(subFileName, subFileExtension, resultText):
    resultTexts = resultText.split('\n')
    translatedDict = dict()

    for txtWithNumber in resultTexts:
        splitedNumberText = txtWithNumber.split(':')
        if len(splitedNumberText) >= 2:
            numText = splitedNumberText[0].strip()
            plainText = splitedNumberText[1] + "\n"
            if numText.isdigit():
                translatedDict[int(numText)] = plainText
            else:
                numList = [int(s) for s in re.findall(r'\d+', numText)]
                if len(numList) == 1:
                    translatedDict[numList[0]] = plainText
                else:
                    print(f"number error=> {txtWithNumber}")

    spacing = 4
    currentIndex = 2

    originSubFile = f"{subFileName}.{subFileExtension}"
    writeSubFile = f"{subFileName}_translated.{subFileExtension}"
    if isRecoverInvalid or isResumeProcess:
        originSubFile = writeSubFile

    print(f"SaveSubscriptionFile : Read-> {originSubFile}")
    print(f"SaveSubscriptionFile : Write-> {writeSubFile}")

    lines = list()

    with open(originSubFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for index, line in enumerate(lines):
            if currentIndex == index:
                subscriptLineNum = lines[index-2].strip()
                if subscriptLineNum.isdigit():
                    if int(subscriptLineNum) in translatedDict:
                        lines[index] = translatedDict[int(subscriptLineNum)]
                currentIndex += spacing

    with open(writeSubFile, 'w', encoding='utf-8') as f:
        f.writelines(lines)

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
    
    with open(f"{fileName}.{fileExtension}", 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for index, line in enumerate(lines):
            if currentIndex == index:
                lineNum = lines[index-2].replace('\n', ':')
                lineNum = lineNum.strip()
                contentsStr += f"{lineNum}{lines[index]}"
                currentIndex += spacing
                originTextTable[lineNum] = f"{lineNum}{lines[index]}"

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
    MAX_CHUNK_SIZE = 50 * 1024  # 50KB (약 25,000 토큰 정도로 예상)
    resultList = []
    tempText = ''
    current_size = 0

    for sentence in originStrList:
        sentence_with_newline = sentence + '\n'
        sentence_size = len(sentence_with_newline.encode('utf-8'))  # UTF-8 바이트 길이로 측정
        
        # 현재 청크가 비어있지 않고, 새 문장을 추가하면 최대 크기를 초과할 경우
        if current_size > 0 and (current_size + sentence_size > MAX_CHUNK_SIZE):
            resultList.append(tempText)
            tempText = sentence_with_newline
            current_size = sentence_size
        else:
            tempText += sentence_with_newline
            current_size += sentence_size

    # 마지막 청크 추가
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

    if isRecoverInvalid:
        originStrList = ReadInvalidTranslationList()
        if originStrList is None:
            print("Read invalid list is null")
            return
    else:
        originStrList = ReadOriginSubscription(SUBSCRIPT_FILE_NAME, SUBSCRIPT_FILE_EXTENSION)

    slicedStrList = SliceStringListForGPTRequest(originStrList)
    if useGemini:
        translatedFullStr = Run_Gemini_Translation(slicedStrList)
        SaveGPTResponsePlainText(SUBSCRIPT_FILE_NAME, translatedFullStr)
    else:
        saveFileName = f'{SUBSCRIPT_FILE_NAME}.txt'
        with open(saveFileName, 'r', encoding='utf-8') as f:
            translatedFullStr = f.read()

    SaveSubscriptionFile(SUBSCRIPT_FILE_NAME, SUBSCRIPT_FILE_EXTENSION, translatedFullStr)

parser = argparse.ArgumentParser(description="Translation-Gemini")
parser.add_argument("--f", required=True)
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
