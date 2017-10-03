f=open("NCM_examples.txt")

f1=open("NCM_examples_with_matching_common_names.txt","w")
f2=open("NCM_examples_with_matching_uncommon_names.txt","w")
f3=open("NCM_examples_with_different_common_names.txt","w")
f4=open("NCM_examples_with_different_uncommon_names.txt","w")

f1r=open("NCM_examples_with_matching_common_names_r.txt","w")
f2r=open("NCM_examples_with_matching_uncommon_names_r.txt","w")
f3r=open("NCM_examples_with_different_common_names_r.txt","w")
f4r=open("NCM_examples_with_different_uncommon_names_r.txt","w")

names=[("JACKIE","MAX"),("SAREK","SPOCK"),("JACKIE","BECKER"),
		("SAREK","CALVIN")]

for line in f:
	f1.write(names[0][0]+" "+line.strip("\n")+" "+names[0][1]+"\n")
	f2.write(names[1][0]+" "+line.strip("\n")+" "+names[1][1]+"\n")
	f3.write(names[2][0]+" "+line.strip("\n")+" "+names[2][1]+"\n")
	f4.write(names[3][0]+" "+line.strip("\n")+" "+names[3][1]+"\n")

	f1r.write(names[0][1]+" "+line.strip("\n")+" "+names[0][0]+"\n")
	f2r.write(names[1][1]+" "+line.strip("\n")+" "+names[1][0]+"\n")
	f3r.write(names[2][1]+" "+line.strip("\n")+" "+names[2][0]+"\n")
	f4r.write(names[3][1]+" "+line.strip("\n")+" "+names[3][0]+"\n")