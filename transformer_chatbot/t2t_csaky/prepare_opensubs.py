import re
from collections import Counter

wc=Counter()
line_dict={}
comma_words=Counter()
in_file=open("valid.txt")

def repl(matchobj):
	return re.sub("'"," '",str(matchobj.group(0)))

def repl_2(matchobj):
	return re.sub("'","",str(matchobj.group(0)))



src=["love"]
trg=[]
# iterate through file
for line in in_file:
	#line=line.split("\t")
	line=line.lower()

	#print(line)
	# keep some special tokens
	line = re.sub("[^a-z .!?'\t\\\]","",line)
	line = re.sub("\\\['] "," '",line)
	line = re.sub("[\\\]"," ",line)
	line =re.sub("[.]"," . ",line)
	line =re.sub("[?]"," ? ",line)
	line =re.sub("[!]"," ! ",line)
	# take care of apostrophes
	line=re.sub("[ ]'[ ]"," ",line)
	#line=re.sub(" '[a-z]",repl_2,line)
	line=re.sub("n't"," n't",line)
	#line=re.sub("[^ n]'[^ t]",repl,line)

	words = line.split()
	for word in words:
		if word in wc:
			wc[word]+=1
		else:
			wc[word]=1

	if len(line.split("\t"))==2:
		trg.append(line.split("\t")[0])
		src.append(line.split("\t")[0])
		trg.append(line.split("\t")[1])
		src.append(line.split("\t")[1])

most_common_words=wc.most_common(32765)
comm_words=[]
#for w,i in most_common_words:
#	comm_words.append(w)
#print(len(wc))
i=0

source_file = open("valSource.txt", "w")
target_file = open("valTarget.txt", "w")
# print vocabulary to a file
if False:
	voc_file=open("vocab_file","w")
	for word,i in wc.most_common(32765):
		voc_file.write(word+'\n')
	# write UNK
	voc_file.write("<UNK>"+'\n')

else:
	voc_file=open("vocab_file")
	for word in voc_file:
		#print(word.strip('\n'))
		comm_words.append(word.strip('\n'))
voc_file.close()

print(len(trg))
i=0
# putting in the unknowns
for line in trg:
	if i%1000==0: print(i)
	words=line.split(" ")
	if "" in words: words.remove("")

	for word in words:
		if word not in comm_words and word!='':
			string=" "+word+" "
			#print(trg[i])
			trg[i]=re.sub(string," <UNK> "," "+trg[i]+" ")
	words=src[i].split(" ")
	if '' in words: words.remove('')
	for word in words:
		if word not in comm_words and word!='':
			string=" "+word+" "
			src[i]=re.sub(string," <UNK> "," "+src[i]+" ")
	source_file.write(src[i]+'\n')
	target_file.write(trg[i]+'\n')
	i+=1

# get the separate dialogs



source_file.close()
target_file.close()

