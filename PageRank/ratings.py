import math
from scipy import spatial

main_array = [
[1,0,3,0,0,5,0,0,5,0,4,0],
[0,0,5,4,0,0,4,0,0,2,1,3],
[2,4,0,1,2,0,3,0,4,3,5,0],
[0,2,4,0,5,0,0,4,0,0,2,0],
[0,0,4,3,4,2,0,0,0,0,2,5],
[1,0,3,0,3,0,0,2,0,0,4,0]
]

def rank(index,rank):
    similars = {}
    for i in range(len(main_array)):
        result = 1 - spatial.distance.cosine(main_array[index], main_array[i])
        if result>=0.4 and i!=index:
            similars[i] = result
    divider = 0
    division = 0    
    for item in similars:
        divider += main_array[item][rank]*similars[item]
        division += similars[item]

    result = divider*1.0/division
    return index+1,rank+1,result

movie,users,result = rank(0, 9)
print ("For movie = {} and user = {} rank is = {}").format(movie,users,result)
movie,users,result = rank(2, 2)
print ("For movie = {} and user = {} rank is = {}").format(movie,users,result)
movie,users,result = rank(3, 3)
print ("For movie = {} and user = {} rank is = {}").format(movie,users,result)
