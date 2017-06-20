spamDb = [[' Buy bicycles for free'],['Bicycles and motorbikes for free'],['Motorbikes rides easy and free']]
normDb = [['Let"s go ride bicycles'],[' Last week I bought motorbikes and they are cool'],['Some messages about bicycles and motorbikes, that are free, are spam messages']]
sentence = ['Cool bicycles and motorbikes']

probOfSpam = len(spamDb)*1.0/(len(spamDb)+len(normDb))
probOfNorm = len(normDb)*1.0/(len(spamDb)+len(normDb))


setOfWord = set()
setOfNormWord = set()
for i in range(len(spamDb)):
    temp = spamDb[i][0].split(' ')
    for j in range(len(temp)):
        setOfWord.add(temp[j].lower())

for i in range(len(normDb)):
    temp = normDb[i][0].split(' ')
    for j in range(len(temp)):
        setOfNormWord.add(temp[j].lower())

countOfAllSpamWord = len(setOfWord)
countOfAllNormWord = len(setOfNormWord)

countOfAllWord = countOfAllSpamWord+ countOfAllNormWord

s = sentence[0].split(' ')
probOfMessInSpam = []
for i in range(len(s)):
    mainFreq = 0
    for j in range(len(spamDb)):
        spamItem = spamDb[j][0].split(' ')
        eachfreq = 0
        for k in range(len(spamItem)):
            if(s[i].lower() == spamItem[k]):
                eachfreq+=1
        if(eachfreq>=1):
            mainFreq+=1
    temp = (mainFreq+1)*1.0/(len(spamDb)+countOfAllWord)
    probOfMessInSpam.append(temp)


resProb = 1
for k in range(len(probOfMessInSpam)):
    resProb = resProb * probOfMessInSpam[k]
print resProb*probOfSpam




spamDb = [['Let"s go ride bicycles'],[' Last week I bought motorbikes and they are cool'],['Some messages about bicycles and motorbikes, that are free, are spam messages']]
sentence = ['Cool bicycles and motorbikes']

probOfSpam = len(spamDb)*1.0/(len(spamDb)+len(normDb))


s = sentence[0].split(' ')
probOfMessInSpam = []
for i in range(len(s)):
    mainFreq = 0
    for j in range(len(spamDb)):
        spamItem = spamDb[j][0].split(' ')
        eachfreq = 0
        for k in range(len(spamItem)):
            if(s[i].lower() == spamItem[k]):
                eachfreq+=1
        if(eachfreq>=1):
            mainFreq+=1
        print mainFreq
    temp = (mainFreq+1)*1.0/(len(spamDb)+countOfAllWord)
    probOfMessInSpam.append(temp)


resProb = 1
for k in range(len(probOfMessInSpam)):
    resProb = resProb * probOfMessInSpam[k]
print resProb*probOfSpam
