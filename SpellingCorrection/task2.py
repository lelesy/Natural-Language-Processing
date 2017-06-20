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


a = raw_input("Enter the word: ")
fname = "file.txt"
with open(fname) as f:
    content = f.readlines()

countOfWord = len(content)

# for i in content:
#     temp = i.split("\t")
#     print a + " : " + temp[0] + " = " + str(edit_distance(a,temp[0])


sumOfWord = 0
for i in content:
    temp = i.split("\t")[1].split("\n")[0]
    sumOfWord+=int(temp)

for i in content:
    temp = i.split("\t")
    if(edit_distance(a,temp[0])<=3):
        print a + " : " + temp[0] + " ====EDIT DISTANCE====" + str(edit_distance(a,temp[0])) + " ====WORD PROB.==== " + str(int(temp[1])*1.0/sumOfWord)

import math

fname = "file2.txt"
with open(fname) as f:
    content = f.readlines()

def func(a):
    freqOfWord = 0
    sumOfWord = 0
    res = 0
    for i in content:
        temp = i.split('\t')
        sumOfWord += int(temp[1].split('\n')[0])
        searchWord = a.split(" ")
        if searchWord[0] in temp[0]:
            freqOfWord+=1

    for i in content:
        temp = i.split('\t')
        if(a.lower() == temp[0].lower()):
            newTemp = int(temp[1].split('\n')[0])
            res = (newTemp+freqOfWord+1)*1.0/(sumOfWord+len(content))
            break
        else:
            res = (1)*1.0/(sumOfWord+len(content))

    return res


a = raw_input("Enter the words: ")
a = a.split(" ")


sums = []
for i in range(len(a)-1):
    sums.append(func(a[i]+" "+a[i+1]))

res = 0
for i in range(len(sums)):
    res+=math.log(sums[i])
print res
