import os,sys,glob
import nltk.corpus
from sklearn.feature_extraction.text import TfidfTransformer
import math
transformer = TfidfTransformer(smooth_idf=True)
stop = set(nltk.corpus.stopwords.words('english'))

nameOfFile = ["001.txt","002.txt","003.txt","004.txt","005.txt","006.txt","007.txt","008.txt","009.txt","010.txt"]
mainContent = []

for i in range(10):
    filename = 'bbc/'+nameOfFile[i]
    tempArr = []
    with open(filename) as f:
        tempContent = f.readlines()
    for k in range(len(tempContent)):
        if(tempContent[k]=='\n'):
            pass
        else:
            tempArr = [l for l in tempContent[k].split('\n')[0].lower().split(' ') if l not in stop]
    mainContent.append(tempArr)

dict = {}
counts = []
words = []
for i in range(len(mainContent)):
    for j in range(len(mainContent[i])):
        tempListOfFreq = []
        for k in range(len(mainContent)):
            isExist = False
            for l in range(len(mainContent[k])):
                if(mainContent[i][j]==mainContent[k][l]):
                    isExist = True
            if(isExist==True):
                tempListOfFreq.append(1)
            else:
                tempListOfFreq.append(0)
        dict[mainContent[i][j]] = tempListOfFreq
        counts.append(tempListOfFreq)
        words.append(mainContent[i][j])

# print words
# print counts
tfidf = transformer.fit_transform(counts)
tfidf = tfidf.toarray()
# print tfidf
# newFile = open("index.txt","w")
# for k in dict:
#     newFile.write(k)
#     newFile.write("  : ")
#     for i in range(10):
#         newFile.write(str(dict[k][i]))
#         newFile.write(',')
#     newFile.write('\n')

sentence = raw_input("Enter your sentence: ")
sentence = sentence.split()

ownVector = []
for i in range(len(words)):
    ownVector.append(0)
for i in range(len(sentence)):
    for k in range(len(words)):
        if(sentence[i]==words[k]):
            ownVector[k]=1

results = []
for i in range(10):
    results.append([])
    for j in range(3):
        results[i].append(0)

for i in range(len(counts)):
    for j in range(len(counts[0])):
        results[j][0] += ownVector[i] * counts[i][j]
        results[j][1] += ownVector[i]**2
        results[j][2] += counts[i][j]**2

for i in range(len(results)):
    print str(i+1) + ' document: '+ format(results[i][0]*1.0/(math.sqrt(results[i][1])*math.sqrt(results[i][2])),'.5f')
