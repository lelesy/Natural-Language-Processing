with open("a.txt") as f:
    content = f.readlines()
arr = []
for i in range(len(content)):
    temp = content[i].split(" ")
    temp[-1] = temp[-1].split("\n")[0]
    arr.append(temp)

dict = {}
eachDict = {}
freqOfWord = {}
freqOfWordW = {}
for i in range(len(arr)):
    for j in range(1,len(arr[i])):
        if(j==1):
            newTemp = 'S'+arr[i][j-1][-1]
            dict[newTemp]=0
            eachDict['S']=0
        temp = arr[i][j-1][-1]+arr[i][j][-1]
        freqOfWord[arr[i][j]] = 0
        freqOfWordW[arr[i][j][:-2]] = 0
        eachDict[arr[i][j-1][-1]] = 0
        dict[temp] = 0


for i in range(len(arr)):
    for j in range(1,len(arr[i])):
        if(j==1):
            newTemp = 'S'+arr[i][j-1][-1]
            dict[newTemp]+=1
            eachDict['S']+=1
        temp = arr[i][j-1][-1]+arr[i][j][-1]
        freqOfWord[arr[i][j]]+=1
        freqOfWordW[arr[i][j][:-2]]+=1
        dict[temp]+=1
        eachDict[arr[i][j-1][-1]]+=1

V = len(freqOfWordW)

dictOfSign = {}
dictOfWord = {}

for i in dict:
    for j in eachDict:
        if(i[0]==j):
            temp = i+' '+j
            dictOfSign[temp] = (dict[i]+1)*1.0/(eachDict[j]+V)

for i in freqOfWord:
    for j in freqOfWordW:
        if(i[:-2]==j):
            temp = i+' '+j
            dictOfWord[temp] = (freqOfWord[i]+1)*1.0/(freqOfWordW[j]+V)

for i in dictOfSign:
    print i,dictOfSign[i]
print '================================================'
for i in dictOfWord:
    print i,dictOfWord[i]


sentence = "the cat in store went to house"
# sentence = "the saw went to the cat in house"
sentence = sentence.split(" ")
del eachDict['S']
for i in range(0, len(sentence)):
    if(i==0):
        xy = []
        print sentence[i]

        for j in eachDict:
            temp = "S"+j+' '+'S'
            tempVal = sentence[i]+'_'+j+' '+sentence[i]
            findOutSign = False
            findOutWord = False
            x = 0
            y = 0
            for k in dictOfSign:
                if(temp==k):
                    findOutSign=True
            for k in dictOfWord:
                if(tempVal==k):
                    findOutWord=True
            if(findOutSign!=False):
                x = dictOfSign[temp]
            else:
                x = 1.0/(eachDict[j]+V)

            if(findOutWord!=False):
                y = dictOfWord[tempVal]
            else:
                y = 1.0/(freqOfWordW[sentence[i]]+V)
            xy.append(x*y)
    else:
        newWWW = []
        for w in range(len(xy)):
            newWW = []
            for j in eachDict:
                newW = []
                for k in eachDict:
                    b = 0
                    temp = j+k+' '+j
                    findOutBigram = False
                    for l in dictOfSign:
                        if(temp==l):
                            findOutBigram=True
                    if(findOutBigram!=False):
                        b = dictOfSign[temp]
                    else:
                        b = (1.0)/(eachDict[j]+V)
                    c = 0
                    tempVal = sentence[i]+'_'+j+' '+sentence[i]
                    findOutNewBigram = False
                    for s in dictOfWord:
                        if(tempVal==s):
                            findOutNewBigram = True
                    if(findOutNewBigram!=False):
                        c = dictOfWord[tempVal]
                    else:
                        c = 1.0/(freqOfWordW[sentence[i]]+V)
                    newW.append(xy[w]*b*c)#5
                newWW.append(max(newW))#5
            newWWW.append(max(newWW))
        xy = newWWW
print max(xy)
