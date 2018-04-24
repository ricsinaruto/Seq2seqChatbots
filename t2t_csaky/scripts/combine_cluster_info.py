clusters=open("data_dir/DailyDialog/base_with_numbers/filtered_data/hash_jaccard/1000_clusters/trainSource_clusters.txt")
cluster_entropies=open("data_dir/DailyDialog/base_with_numbers/filtered_data/hash_jaccard/1000_clusters/trainSource_cluster_entropies.txt")

outfile=open("cluster_elements_and_entropies.txt","w")

cluster_list={}
for line in clusters:
	cluster_list[line.split(":")[0]]=line.split(":")[1]

for line in cluster_entropies:
	num_el=cluster_list[line.split(";")[0]]
	outfile.write(line.strip("\n")+";"+num_el)

outfile.close()
