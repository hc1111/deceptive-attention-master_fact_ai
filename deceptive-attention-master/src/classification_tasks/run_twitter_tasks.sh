mkdir -p logs
# run the pronoun dataset
export CUDA_VISIBLE_DEVICES=0

for s in $(seq 1 5); do
	for model_name in emb-lstm-att emb-att; do
	    	python main_twitter --num-epochs 15 --num-visualize 5 --loss-hammer $c_hammer  --model $model_name --task twitter  --seed $s | tee -a logs/no_block_logs_twitter_model=$model_name+hammer=$c_hammer+seed=$s;

		for c_hammer in 0.0 0.1 1.0; do
            		# on SST+Wiki dataset
			 python main_twitter.py --num-epochs 15 --num-visualize 5 --loss-hammer $c_hammer  --model $model_name --task twitter --use-block-file --seed $s | tee -a logs/logs_twitter_model=$model_name+hammer=$c_hammer+seed=$s;
			# on pronoun dataset
			# occupation classification
		done;
	done;
done;
