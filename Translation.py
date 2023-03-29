import openai
from decimal import Decimal
import traceback
import sys
import os
import json





def Run_GPT_Translation(inputStrs):

    endChunkIndex=GetEndChunkIndexFromResumeInfo()

    openai.api_key="REPLACE_THIS_YOUR_OPEN_API_KEY"

    resultText=''
    totalToken=0
    requestCount=0
    currentChunkIndex=0
    gptRequestFailed=False

    try:

        for chunkIndex,sentence in enumerate(inputStrs):
            
            if chunkIndex <=endChunkIndex:
                isResumeProcess=True
                continue

            response= openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                {
                    "role":"system",
                    #"content":"Please line by line translate the following text into \"Korean\"."
                    "content":"한줄씩 한국어로 번역해줘."
                },
                {
                    "role":"assistant",
                    #"content":"Never translate to English. origin text include the number at the begining of line."
                    "content":"원문을 같이적어줘. 원문도 같이 적어줄때 문장앞에 있는 번호는 포함시켜줘. 번역텍스트에는 번호는 빼줘."
                },
                {
                    "role":"user",
                    #"content":f"write both origin text and translated text. show me like \"original text(translated text)\" \n\n{sentence}"
                    "content":f"형식을 반드시 지켜줘. 형식은 \'원문(번역텍스트)\' 로 써줘. \n\n{sentence}"
                }
                ]
            )
            currentToken=response['usage']['total_tokens']
            print("Tokens:",currentToken)
            totalToken+=int(currentToken)


            resultText+=response['choices'][0]['message']['content']
            requestCount+=1
            currentChunkIndex=chunkIndex

    except:
        print("GPT request failed")
        gptRequestFailed=True
    
    if requestCount>0:
        if gptRequestFailed ==True:
            SaveResumeInfo(currentChunkIndex)
        else:
            RemoveResumeInfo()

        resultText=resultText.encode('utf-8')
        resultText=resultText.decode('utf-8')
        resultText=ValidateTranslatedText(resultText)
    else:
        RemoveResumeInfo()
        print('GPT Response is null. exit')
        sys.exit(-1)

       

    print("Total Tokens:",totalToken)
    print(f"Cost:{float(totalToken/1000)*0.002}$")

    
    return resultText

def SaveGPTResponsePlainText(subFileName,resultText):
    saveFileName=f'{subFileName}.txt'
    fileMode='w'
    
    
    if isResumeProcess == True and os.path.exists(saveFileName)==True:
        fileMode='a'

    with open(saveFileName,fileMode,encoding='utf-8') as f:
        f.write(resultText)
    
    
def SaveSubscriptionFile(subFileName,subFileExtension,resultText):

    resultTexts=resultText.split('\n')

    translatedDict=dict()

    for txtWithNumber in resultTexts:
        splitedNumberText=txtWithNumber.split(':')
        if len(splitedNumberText)>=2:
            numText=splitedNumberText[0].strip()
            plainText=FilterString(splitedNumberText[1])+'\n'
            if numText.isdigit():
                translatedDict[int(numText)]=plainText
            else:
                print(f"number error=> {txtWithNumber}")


    spacing=4
    currentIndex=2
    # inputIndex=0

    originSubFile=f"{subFileName}.{subFileExtension}"
    writeSubFile=f"{subFileName}_translated.{subFileExtension}"
    if isRecoverInvalid ==True or isResumeProcess==True:
        originSubFile=writeSubFile
    
    print(f"SaveSubscriptionFile : Read-> {originSubFile}")
    print(f"SaveSubscriptionFile : Write-> {writeSubFile}")
    
    lines=list()

    with open(originSubFile,'r') as f:
        lines=f.readlines()
    

        for index,line in enumerate(lines):
            if currentIndex==index:
                
                subscriptLineNum=lines[index-2].strip()
                if subscriptLineNum.isdigit():
                    if int(subscriptLineNum) in translatedDict:
                        lines[index]=translatedDict[int(subscriptLineNum)]

                currentIndex+=spacing
                # inputIndex+=1
                # if inputIndex >=len(resultTexts):
                #     break
        
    with open(writeSubFile,'w',encoding='utf-8') as f:
        f.writelines(lines)

def FilterString(text):
    if "(" in text ==False or ")" in text == False: 
        return text
    
    start = text.find("(") + 1
    end = text.find(")", start)
    inner_string = text[start:end]
    return inner_string


    
def ReadOriginSubscription(fileName,fileExtension):
    
    spacing=4
    currentIndex=2

    contentsStr=''
    with open(f"{fileName}.{fileExtension}",'r') as f:
        lines=f.readlines()
    

        for index,line in enumerate(lines):
            if currentIndex==index:
                lineNum=lines[index-2].replace('\n',':')
                lineNum=lineNum.strip()
                contentsStr+=f"{lineNum}{lines[index]}" 
                currentIndex+=spacing

                originTextTable[lines[index-2].replace('\n','')]=f"{lineNum}{lines[index]}" 
            



    contents=contentsStr.split('\n')
    del contents[-1]
    print("Total Sentences",len(contents))

    return contents
def ReadInvalidTranslationList():
    
    with open(f"InvalidTranslation_{SUBSCRIPT_FILE_NAME}.txt",'r') as f:
        lines=f.readlines()
        if len(lines)<1:
            return None

        for line in lines:
            lineNum=line.split(":")[0].strip()
            originTextTable[lineNum]=line
        return lines
    
    
    

def SliceStringListForGPTRequest(originStrList):
    #1sentence -> 30token
    #gpt3.5-turbo : limit 4096 token per 1 request
    #price: gpt3.5-turbo 0.0020$ per 1000token

    #41880 token 
    #130 sentence ->1 request
    #in case sentence is too long, 70 sentence to 1 chunk

    count=0
    resultList=list()
    tempText=''

    for sentence in originStrList:
        if count >= 70:
            resultList.append(tempText)
            tempText=''
            count=0
        
        tempText+=sentence+'\n'
        count+=1 

    if count<=70 and count>0:
        resultList.append(tempText)

    with open('sliced_text.txt','w',encoding='utf-8') as f:
        f.writelines(resultList)

    return resultList
        

def ValidateTranslatedText(texts):
    sentenceList=texts.split('\n')
    output=''

    for s in sentenceList:
        output+=FixSentenceNoCariageReturn(s)+"\n"

    output=output.replace("\n(","(")
    return output
        


def FixSentenceNoCariageReturn(sentence):
    
    outputStr=sentence

    colonCount=0

    for char in sentence:
        if char == ':':
            colonCount+=1

    if colonCount<=1:
        return outputStr 

    for idx,char in enumerate(sentence):
        if idx >2:
            if char.isdigit():
                if sentence[idx-1].isdigit() is False and sentence[idx-1] != '\n':
                    print(idx,char)
                    outputStr = f"{outputStr[:idx]}\n{char}{outputStr[idx + 1:]}"  
                
    return outputStr
   

def RemoveSubscriptionNumber(sentence):
    res=filter(lambda c : not c.isdigit() ,sentence)
    res="".join(res).replace(":","")
    return res

def SaveResumeInfo(endChunkIndex):
    if os.path.exists(RESUME_INFO_FILE_NAME) == False:
        with open(RESUME_INFO_FILE_NAME,'w') as f:
            resumeInfoJson=dict()
            resumeInfoJson[SUBSCRIPT_FILE_NAME]=dict()
            resumeInfoJson[SUBSCRIPT_FILE_NAME]["EndChunkIndex"]=endChunkIndex
            f.write(json.dumps(resumeInfoJson))
    else:
        resumeInfo=LoadResumeInfo()
        if resumeInfo is not None:
            if SUBSCRIPT_FILE_NAME in resumeInfo:
                resumeInfo[SUBSCRIPT_FILE_NAME]["EndChunkIndex"]=endChunkIndex
            else:
                resumeInfo[SUBSCRIPT_FILE_NAME]=dict()
                resumeInfo[SUBSCRIPT_FILE_NAME]["EndChunkIndex"]=endChunkIndex
       
            with open(RESUME_INFO_FILE_NAME,'w') as f:
                json.dump(resumeInfo,f)

def GetEndChunkIndexFromResumeInfo():
    endChunkIndex=-1
    resumeInfo=LoadResumeInfo()
    if resumeInfo is not None:
        if SUBSCRIPT_FILE_NAME in resumeInfo:
            endChunkIndex=resumeInfo[SUBSCRIPT_FILE_NAME]["EndChunkIndex"]

    return endChunkIndex

def LoadResumeInfo():
    if os.path.exists(RESUME_INFO_FILE_NAME) == True:
        with open(RESUME_INFO_FILE_NAME,'r') as f:
            jsonStr=f.read()
            jsonObj=json.loads(jsonStr)
            return jsonObj
    
    return None        

def RemoveResumeInfo():

    resumeInfo=LoadResumeInfo()
    if resumeInfo is not None:
        if SUBSCRIPT_FILE_NAME in resumeInfo:
            del resumeInfo[SUBSCRIPT_FILE_NAME]

            with open(RESUME_INFO_FILE_NAME,'w') as f:
                json.dump(resumeInfo,f)

def ValidateGPT_Translation(resultText):
    listResult=resultText.split('\n')
    for res in listResult:
        splitRes=res.split("(")
        if len(splitRes)>1:
            originText=splitRes[0].replace(" ","")
            originText=originText.replace("\n","")

            if originText in validateTable:
                validateTable[originText]=True
    
    invalidList=list()
    for k,v in validateTable.items():
        if v == False:
            number=k.split(":")[0]
            print(f"Validate Failed: {k}")
            invalidList.append(originTextTable[number])
    
    if len(invalidList)>0:
        with open(f"InvalidTranslation_{SUBSCRIPT_FILE_NAME}.txt",'w',encoding='utf-8') as f:
            f.writelines(invalidList)

    


    


def SetTableForValidation(originTexts):
    for text in originTexts:

        textWithoutSpacing=text.replace(" ","")
        textWithoutSpacing=textWithoutSpacing.replace("\n","")

        #print(textWithoutSpacing)
        validateTable[textWithoutSpacing]=False
   
    

       
def CheckNeedToRecoverValdation():
    if os.path.exists(f"InvalidTranslation_{SUBSCRIPT_FILE_NAME}.txt") ==True:
        userSelect=input("====== Need To Recover invalid Translation=====\nRecover?(y/n)")
        if userSelect != "y":
            return False
        

        return True
    return False


SUBSCRIPT_FILE_NAME="REPLACE_SUBSCRIPT_FILE_NAME"
SUBSCRIPT_FILE_EXTENSION="srt"
RESUME_INFO_FILE_NAME="./resume_info.json"
isResumeProcess=False
useGPT=True
isRecoverInvalid=False
validateTable=dict()
originTextTable=dict()


def Run_Translation():

    global isRecoverInvalid
    isRecoverInvalid=CheckNeedToRecoverValdation()

    if isRecoverInvalid:
        originStrList=ReadInvalidTranslationList()
        if originStrList == None:
            print("Read invalid list is null")
            return
    else:
        originStrList=ReadOriginSubscription(SUBSCRIPT_FILE_NAME,SUBSCRIPT_FILE_EXTENSION)
    
    SetTableForValidation(originStrList)
    slicedStrList=SliceStringListForGPTRequest(originStrList)
    if useGPT ==True:

        translatedFullStr = Run_GPT_Translation(slicedStrList)
        ValidateGPT_Translation(translatedFullStr)
        SaveGPTResponsePlainText(SUBSCRIPT_FILE_NAME,translatedFullStr)
    else:
        saveFileName=f'{SUBSCRIPT_FILE_NAME}.txt'
        with open(saveFileName,'r',encoding='utf-8') as f:
            translatedFullStr=f.read()
        
    SaveSubscriptionFile(SUBSCRIPT_FILE_NAME,SUBSCRIPT_FILE_EXTENSION,translatedFullStr)



if __name__ == "__main__":
    try:
       Run_Translation()
 
       pass
        

        
    except Exception as e:
        print(e)
        print(traceback.format_exc())


