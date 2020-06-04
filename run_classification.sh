


gmlbase="../node2sig_edgelists"
embbase="../node2sig_embeddings"
resultbase="../node2sig_results"


datasets=(blogcatalog cora dblp Homo_sapiens wiki)
#

walk_lens=(1 2 3 4 5)


for L in ${walk_lens[@]}
do
        for dataset in ${datasets[@]}
	
	do
		gml_file_path=${gmlbase}/${dataset}_newborn.gml
		emb_file_path=${embbase}/${dataset}_newborn_L=${L}.embedding
		result_file_path=${resultbase}/${dataset}_newborn_L=${L}.result
		
                 echo "---------------"
		 echo $dataset, $L
                 echo "---------------"

		python3 classification.py ${gml_file_path} ${emb_file_path} ${result_file_path} 10 all svm-hamming
		
	done
done

