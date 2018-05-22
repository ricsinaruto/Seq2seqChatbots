def create_table(name="ncm"):
	ncm_in=open(name+".txt")
	ncm_out=open(name+"_latex.txt","w")
	table=[[],[[],[]],[[],[]],[],[[],[]],[],[]]
	# read ncm files
	for i in range(1,7):
		file=open(str(i)+"/"+name+".txt")
		if i==1 or i==2 or i==4:
			over_file=open(str(i)+"/over_"+name+".txt")

			for line in over_file:
				if len(line.split())>10:
					line=" ".join(line.split()[:10])+"...CONT."
				table[i][1].append(line.strip("\n"))
			over_file.close()

		for line in file:
			if len(line.split())>10:
					line=" ".join(line.split()[:10])+"...CONT."
			if i==1 or i==2 or i==4:
				table[i][0].append(line.strip("\n"))
			else:
				table[i].append(line.strip("\n"))
		file.close()

	i=0
	for line in ncm_in:
		ncm_out.write(line.strip("\n")+"&"
					  #+table[1][0][i]+"&"
					  #+table[1][1][i]+"&"
					  +table[2][0][i]+"&"
					  +table[2][1][i]+"&"
					  #+table[3][i]+"&"
					  +table[4][0][i]+"&"
					  +table[4][1][i]+"&"
					  +table[5][i]+"&"
					  +table[6][i]
					  +"\\\ \hline\n")
		i+=1
	ncm_in.close()
	ncm_out.close()

create_table("ncm")
#create_table("icp")