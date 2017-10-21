q=open("NCM_examples.txt")
a=open("results/cornellS.txt")
cornell=open("results/cornellS_match.txt")
cornellS=open("results/cornellS_match_r.txt")
opensubs=open("results/CornellS_diff.txt")
opensubsF=open("results/CornellS_diff_r.txt")
output=open("latex_table_CornellS.txt","w")


for l1,l2,l3,l4,l5,l6 in zip(q,a,cornell,cornellS,opensubs,opensubsF):

	string=l1.strip("\n")+" & "+l2.strip("\n")+" & "+l3.strip("\n")+" & "+l4.strip("\n")+" & "+l5.strip("\n")+" & "+l6
	output.write(string)
	string="\\\ \hline\n"
	output.write(string)


