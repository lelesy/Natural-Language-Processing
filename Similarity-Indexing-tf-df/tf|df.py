from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
transformer = TfidfTransformer(smooth_idf=True)


counts = [[4,1,0,2,4],
          [3,2,1,3,0],
          [1,0,0,0,0],
          [0,4,0,0,0],
          [0,0,3,0,0],
          [2,1,5,3,0],
          [0,5,2,0,1]]

copyOfCounts = []
for i in range(len(counts)):
    temp = []
    for j in range(len(counts[0])):
        temp.append(0)
        if(counts[i][j]!=0):
            temp[j] =  float(format((math.log10(counts[i][j]) + 1) * (math.log10( 5.0 /math.log10(counts[i][j] + 1))),'.5f'))
    copyOfCounts.append(temp)
print copyOfCounts




# tfidf = transformer.fit_transform(counts)
# tfidf = tfidf.toarray()
# print tfidf

res = 0
res1 = 0
res2 = 0
for i in range(len(copyOfCounts)):
    res+= (copyOfCounts[i][0] * copyOfCounts[i][3])
    res1+= (copyOfCounts[i][0]) ** 2
    res2+= (copyOfCounts[i][3]) ** 2

res1 = math.sqrt(res1)
res2 = math.sqrt(res2)

mainRes = res*1.0/(res1*res2)
print mainRes
