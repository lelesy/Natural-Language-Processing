# !/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import json
import pymorphy2
import nltk
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

morph = pymorphy2.MorphAnalyzer()

input = raw_input("Введите запрос: ")
new_input = input.split(" ")
main_list = []
for i in range(len(new_input)):
    temp = new_input[i].decode("utf-8")
    main_list.append(morph.parse(temp)[0][2])

print "Подождите, идет обработка процесса..."
main_response = {}
for word in main_list:
    response = []
    id_response = []
    with io.open('train.json','r',encoding="utf-8") as data_file:
        news = json.load(data_file)
        news_collection = []
        for i in range(0,10):
            found = False
            content = news[i]["text"]
            tokens = [j.lower() for j in nltk.word_tokenize(content)]
            for token in tokens:
                parseds = morph.parse(token)
                compared_word = parseds[0][2]
                if(word==compared_word):
                    found = True
            if(found==True):
                response.append(1)
            else:
                response.append(0)
            id_response.append(news[i]["id"])
    main_response[word] = response
new_arr = []
for item in main_response:
    new_arr.append(main_response[item])
stat_arr = []
for i in range(len(new_arr[0])):
    count = 0
    for j in range(len(new_arr)):
        count+=new_arr[j][i]
    stat_arr.append(count)
new_stat = {}

for j in range(len(stat_arr)):
    new_stat[id_response[j]] = stat_arr[j]

a = sorted(new_stat.iteritems(), key=lambda key_value: key_value[1])
print "Обработанные слова:"
for item in main_response:
    print item,
print ""
print " Статистика "
for item in reversed(a):
    print "В "+str(item[0])+" статье "+str(item[1])+" слов(а) найдено"
