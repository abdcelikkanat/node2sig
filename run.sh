
edgelistbase="../nodesig_edgelists"
outputbase="../nodesig_embeddings"
walklen=5

datasets=(blogcatalog dblp Homo_sapiens wiki)
#

walk_lens=(1 2 3 4 5)


for dataset in ${datasets[@]}
do
	
	for L in ${walk_lens[@]}
	do
		edge_file_path=${edgelistbase}/${dataset}_newborn.edgelist
		emb_file_path=${outputbase}/${dataset}_newborn_L=${L}.edgelist
		
		./nodesig --edgefile ${edge_file_path} --embfile ${emb_file_path} --walklen ${walklen}
		
	done
done

