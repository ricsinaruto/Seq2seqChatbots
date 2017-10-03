import re
from collections import Counter

in_file=open("cornell_movie_data/movie_lines.txt")
name_vocab=open("cornell_movie_data/name_vocab.txt","w")
name_counter=Counter()

for line in in_file:
	line=line.split(" +++$+++ ")
	name=line[3]
	if name in name_counter:
		name_counter[name]+=1
	else:
		name_counter[name]=1

common_names=name_counter.most_common(3001)
names=[]
for w,c in common_names:
	word=re.sub(" ","_",w)
	if word!="":
		name_vocab.write(word+"\n")

name_vocab.close()
in_file.close()