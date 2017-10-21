q=open("NCM_examples.txt")
a=open("NCM_responses.txt")
cornell=open("results/cornell.txt")
cornellS=open("results/cornellS.txt")
opensubs=open("results/opensubs.txt")
opensubsF=open("results/opensubsF.txt")
output=open("latex_table.txt","w")


for l1,l2,l3,l4,l5,l6 in zip(q,a,cornell,cornellS,opensubs,opensubsF):

	string=l1.strip("\n")+" & "+l2.strip("\n")+" & "+l3.strip("\n")+" & "+l4.strip("\n")+" & "+l5.strip("\n")+" & "+l6
	output.write(string)
	string="\\\ \hline\n"
	output.write(string)


