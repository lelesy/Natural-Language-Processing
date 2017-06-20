import re

a = "1.2.3333101.91"
b = "5.130.12.190.5"
c = "5.12.1"

s = re.findall(r'([0-9]\.[0-9]+)\.',a)
print s[0]
s = re.findall(r'([0-9]\.[0-9]+)\.',b)
print s[0]
s = re.findall(r'([0-9]\.[0-9]+)\.',c)
print s[0]


f = open("SherlockHolmes.txt","r")
line = f.read()

s = re.findall(r'([A-Z][a-z]+)',line)
print s

s = re.findall(r'([^\.][A-Z][a-z]+)',line)
print s

s = re.findall(r'([A-Z]\. [A-Z][a-z]+)',line)
print s

s = re.findall(r'([A-Z][a-z]+) Street',line)
print s


f = open("file.txt","r")
line = f.read()
s = re.findall(r'Mozilla/[0-9]\.[0-9]+ [(](.*?)[\)]',line)
for i in s:
    print i
    temp = re.findall(r'(.*?);(.*?);(.*?);',i)
    print temp
