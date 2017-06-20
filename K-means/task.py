import matplotlib.pyplot as plt
import math
critics = [[1,4],[5,6],[7,5],[7,8],[2,4]]
k_means = [[6,7],[7,8]]


c1_diff = {}
c2_diff = {}
for i in range(len(critics)):
    temp_1 = math.sqrt(((k_means[0][0] - critics[i][0])**2) + ((k_means[0][1]-critics[i][1])**2))
    temp_2 = math.sqrt(((k_means[1][0] - critics[i][0])**2) + ((k_means[1][1]-critics[i][1])**2))
    index_1 ='(' + str(critics[i][0]) + ',' + str(critics[i][1]) + ')'
    index_2 ='(' + str(critics[i][0]) + ',' + str(critics[i][1]) + ')'
    c1_diff[index_1] = temp_1
    c2_diff[index_2] = temp_2

print 'C1:'
print c1_diff
print 'C2:'
print c2_diff


new_c1_diff = {}
new_c2_diff = {}

for i in (c1_diff):
    a = c1_diff[i]
    b = c2_diff[i]
    if(a<b):
        new_c1_diff[i] = c1_diff[i]
    else:
        new_c2_diff[i] = c2_diff[i]

print 'C1: '
print new_c1_diff
print 'C2: '
print new_c2_diff
