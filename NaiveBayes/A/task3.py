import math
def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]




myWord = "in dairy life"

fname = "file.txt"
with open(fname) as f:
    content = f.readlines()

f2name = "file2.txt"
with open(f2name) as f:
    contentTwo = f.readlines()



def findProb(mainString):

    firstWord = mainString.split(" ")[0]
    errorWord = mainString.split(" ")[1]

    print errorWord

    sumOfAllWordInFirstFile = 0
    sumOfWordInFile = len(content)


    for i in range(len(content)):
        temp2 = content[i].split("\t")[1].split("\n")[0]
        sumOfAllWordInFirstFile+=int(temp2)



    sumOfAllWordInSecondFile = 0
    sumOfSentencesInFile = len(contentTwo)*2

    arr = []
    for i in range(len(contentTwo)):
        tempArr = []
        temp1 = contentTwo[i].split(" ")[0].lower()
        temp2 = contentTwo[i].split(" ")[1].split("\t")[0].lower()
        temp3 = contentTwo[i].split(" ")[1].split("\t")[1].split("\n")[0]

        sumOfAllWordInSecondFile+=int(temp3)
        if(temp1 == firstWord):
            tempArr.append(temp1)
            tempArr.append(temp2)
            tempArr.append(temp3)
            arr.append(tempArr)

    returnedArr = []
    added = False
    for i in range(len(content)):
        temp1 = content[i].split("\t")[0]
        temp2 = int(content[i].split("\t")[1].split("\n")[0])
        if(edit_distance(errorWord,temp1)<=1):
            for k in range(len(arr)):
                if(arr[k][1]==temp1):
                    someTemp = []
                    added = True

                    probOfWord = (temp2+1)*1.0/(sumOfAllWordInFirstFile+sumOfWordInFile)
                    probOfWord = math.log10(probOfWord)

                    print probOfWord

                    probOfStr = (int(arr[k][2])+1)*1.0/(sumOfAllWordInSecondFile + sumOfSentencesInFile)
                    probOfStr = math.log10(probOfStr)
                    mainProb = probOfWord + probOfStr

                    someTemp.append(arr[k][1])
                    someTemp.append(mainProb)
                    returnedArr.append(someTemp)

    if(added == False):
        someTemp = []
        probOfWord = (temp2+1)*1.0/(sumOfAllWordInFirstFile+sumOfWordInFile)
        probOfWord = math.log10(probOfWord)

        probOfStr = 1.0/(sumOfAllWordInSecondFile + sumOfSentencesInFile)
        probOfStr = math.log10(probOfStr)
        mainProb = probOfWord + probOfStr

        someTemp.append(arr[k][1])
        someTemp.append(mainProb)
        returnedArr.append(someTemp)

    return returnedArr


arr = findProb(myWord)
print arr

lastProb = 0
mainRes = []
for i in range(len(arr)):
    tempArr = findProb(arr[i][0]+" life")
    if(len(tempArr)!=0):
        print arr[i][1]+tempArr[0][1]
        lastProb += arr[i][1]+tempArr[0][1]

print "Last Prob: "+ str(lastProb)
