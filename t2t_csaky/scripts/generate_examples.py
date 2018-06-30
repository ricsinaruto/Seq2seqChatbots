""" annotate the example inputs with speaker names """

f=open("NCM_examples/fullSource.txt")

f1=open("NCM_examples/sep_names/NCM_examples_with_sep_matching_common_names.txt","w")
f2=open("NCM_examples/sep_names/NCM_examples_with_sep_different_common_names.txt","w")

f1r=open("NCM_examples/sep_names/NCM_examples_with_sep_matching_common_names_r.txt","w")
f2r=open("NCM_examples/sep_names/NCM_examples_with_sep_different_common_names_r.txt","w")

names=[("MRS._ROBINSON_m77","BEN_m77"),("BEN_m77","JOE_m100")]

for line in f:
	f1.write(names[0][0]+" "+line.strip("\n")+" "+names[0][1]+"\n")
	f2.write(names[1][0]+" "+line.strip("\n")+" "+names[1][1]+"\n")

	f1r.write(names[0][1]+" "+line.strip("\n")+" "+names[0][0]+"\n")
	f2r.write(names[1][1]+" "+line.strip("\n")+" "+names[1][0]+"\n")