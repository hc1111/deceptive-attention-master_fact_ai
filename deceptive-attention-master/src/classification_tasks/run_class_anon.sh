

# # Activate your environment
source activate attention

mkdir -p output_emb

# run the pronoun dataset
for s in $(seq 1 5); do
	for model_name in emb-att emb-lstm-att ; do
		for c_hammer in 0.0 ; do
            # on SST+Wiki dataset
			srun -u python main_x.py \
                --num-epochs 15 \
                --num-visualize 5 \
                --loss-hammer $c_hammer \
                --model $model_name \
                --task sst-wiki \
                --anon \
                --seed $s | tee output_emb/logs_sst_wiki_model=$model_name+hammer=$c_hammer+seed=$s+anon;
			# # on pronoun dataset
            srun -u python main_x.py \
                --num-epochs 15  \
                --num-visualize 5 \
                --loss-hammer $c_hammer  \
                --model $model_name \
                --task pronoun \
                --anon \
                --seed $s | tee output_emb/logs_pronoun_model=$model_name+hammer=$c_hammer+seed=$s+anon;
			# occupation classification
     		srun -u python main_x.py \
                --num-epochs 5 \
                --num-visualize 100 \
                --loss-hammer $c_hammer  \
                --model $model_name \
                --task occupation-classification \
                --anon \
                --seed $s \
                --clip-vocab | tee output_emb/logs_occupation_classification=$model_name+hammer=$c_hammer+seed=$s+anon;
		done;
	done;
done;
