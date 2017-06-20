import nltk
import operator
from collections import Counter

stopWord = set(nltk.corpus.stopwords.words('english'))

file1 = "business.txt"
file2 = "entertainment.txt"
file3 = "politics.txt"

with open(file1) as f:
	textOne = f.read()

with open(file2) as f:
	textTwo = f.read()

with open(file3) as f:
	textThree = f.read()

businessDB = textOne.split("newp")
entertainmentDB = textTwo.split("newp")
politicsDB = textThree.split("newp")

categories = [len(businessDB),len(entertainmentDB),len(politicsDB)]

cleanTextOne  = [i for i in textOne.lower().split() if i not in stopWord]
cleanTextTwo  = [i for i in textTwo.lower().split() if i not in stopWord]
cleanTextThree  = [i for i in textThree.lower().split() if i not in stopWord]


countOfAllWord=len(cleanTextOne)+len(cleanTextTwo)+len(cleanTextThree)

freq_1 = Counter(cleanTextOne)
freq_2 = Counter(cleanTextTwo)
freq_3 = Counter(cleanTextThree)

for i in freq_1.keys():
	freq_1[i]=float((freq_1[i]+1))/categories[0]
for i in freq_2.keys():
	freq_2[i]=float((freq_2[i]+1))/categories[1]
for i in freq_3.keys():
	freq_3[i]=float((freq_3[i]+1))/categories[2]

a=raw_input("Enter your sentence: ")

sentence = a.split()
prob_1=1
prob_2=1
prob_3=1

for w in sentence:
	if freq_1[w]!=0:
		prob_1*=freq_1[w]
	if freq_2[w]!=0:
		prob_2*=freq_2[w]
	if freq_3[w]!=0:
		prob_3*=freq_3[w]

resList = {"Business":prob_1*1.0/countOfAllWord,"Entertainment":prob_2*1.0/countOfAllWord,"Politics":prob_3*1.0/countOfAllWord}

sorted_x = sorted(resList.items(), key=operator.itemgetter(1))
print sorted_x[-1]
