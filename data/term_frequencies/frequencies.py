from datasets import load_dataset
import requests
import json
import os
from collections import OrderedDict

def getTermFrequency(term):
    url = 'https://api.infini-gram.io/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'index': 'v4_piletrain_llama',
        'query_type': 'count',
        'query': term
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['token_ids'][0]

def getTermDict(terms):
    termDict = {}
    for term in terms:
        termDict[term] = getTermFrequency(term)
    return termDict

def getTermsFromFileString(fileString):
    return fileString.split('\n')

def getFileString(fileName, folderPath):
    for file in os.listdir(folderPath):
        if file == fileName:
            filePath = os.path.join(folderPath, fileName)
            with open(filePath, 'r') as file:
                return file.read()

def addFrequenciesToJson(fileName, wordList, chunkSize=25):
    progressFile = f"progress-{fileName}.txt"

    try: 
        if os.path.exists(f'{fileName}-frequencies.json'):
            with open(f'{fileName}-frequencies.json') as f:
                data = json.load(f)
        else:
            data = {}
        if fileName not in data:
            data[fileName] = {}

        if os.path.exists(progressFile):
            with open(progressFile, 'r') as f:
                last_index = int(f.read())
        else:
            last_index = 0
        counter = 0
        for i in range(last_index, len(wordList), chunkSize):
            chunk = wordList[i:i + chunkSize]
            termDict = getTermDict(chunk)
            data[fileName].update(termDict)
            with open(f'{fileName}-frequencies.json', 'w') as f:
                json.dump(data, f, indent=4)
            with open(progressFile, 'w') as f:
                f.write(str(i + chunkSize))
            print(f'words, {i}-{i+chunkSize} added to {fileName}-frequencies.json')
            counter = i + chunkSize
        if counter < len(wordList):
            chunk = wordList[counter:len(wordList)]
            termDict = getTermDict(chunk)
            data[fileName].update(termDict)
            with open(f'{fileName}-frequencies.json', 'w') as f:
                json.dump(data, f, indent=4)
            with open(progressFile, 'w') as f:
                f.write(str(i + chunkSize))
            print(f'words, {i}-{len(wordList)} added to {fileName}-frequencies.json')
    
    except Exception as e:
        print(f"An error occurred: {e}")

def addFileFrequenciesToJson(fileName, folderPath):
    wordList = getTermsFromFileString(getFileString(fileName, folderPath))
    uniqueList = list(OrderedDict.fromkeys(wordList))
    addFrequenciesToJson(fileName, uniqueList, 500)

addFileFrequenciesToJson('wordnet.txt', 'textfiles')
    