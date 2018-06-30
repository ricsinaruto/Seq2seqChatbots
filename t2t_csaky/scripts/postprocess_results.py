""" basically prepare a latex table """

q=open("NCM_examples/fullSource.txt")
a=open("NCM_examples/NCM_responses.txt")
cornell=open("NCM_examples/results_for_tdk/cornell.txt")
cornellS=open("NCM_examples/results_for_tdk/cornellS.txt")
opensubs=open("NCM_examples/results_for_tdk/opensubs.txt")
opensubsF=open("NCM_examples/results_for_tdk/opensubsF.txt")
output=open("NCM_examples/results_for_tdk/latex_table.txt","w")


for l1,l2,l3,l4,l5,l6 in zip(q,a,cornell,cornellS,opensubs,opensubsF):

	string=l1.strip("\n")+" & "+l2.strip("\n")+" & "+l3.strip("\n")+" & "+l4.strip("\n")+" & "+l5.strip("\n")+" & "+l6
	output.write(string)
	string="\\\ \hline\n"
	output.write(string)