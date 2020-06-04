
edgelistbase="../node2sig_edgelists"
outputbase="../node2sig_embeddings"
walklen=5

datasets=(blogcatalog cora dblp Homo_sapiens wiki)
#

walk_lens=(1 2 3 4 5)


for L in ${walk_lens[@]}
do
        for dataset in ${datasets[@]}
	
	do
		edge_file_path=${edgelistbase}/${dataset}_newborn.edgelist
		emb_file_path=${outputbase}/${dataset}_newborn_L=${L}.embedding
		
                echo "---------------"
		echo $dataset, $L
                echo "---------------"

		./nodesig --edgefile ${edge_file_path} --embfile ${emb_file_path} --walklen ${L}
		
	done
done

