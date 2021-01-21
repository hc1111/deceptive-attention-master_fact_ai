mkdir -p logs
# run the pronoun dataset
export CUDA_VISIBLE_DEVICES=0

for BLOCK in 'none'; do

  for s in $(seq 1 5); do
    for model_name in emb-lstm-att emb-att; do
      for c_hammer in 0.0 0.1 1.0; do
                  # on SST+Wiki dataset
         python main.py --num-epochs 15 --num-visualize 5 --loss-hammer $c_hammer  --model $model_name --task sst-wiki  --seed $s | tee -a logs/logs_sst_wiki_model=$model_name+hammer=$c_hammer+seed=$s+_$BLOCK;
        # on pronoun dataset
                   python main.py --num-epochs 15 --num-visualize 5 --loss-hammer $c_hammer  --model $model_name --task pronoun    --seed $s | tee -a logs/logs_pronoun_model=$model_name+hammer=$c_hammer+seed=$s+_$BLOCK;
        # occupation classification
             python main.py --num-epochs 5 --num-visualize 100 --loss-hammer $c_hammer  --model $model_name   --task occupation-classification --seed $s --clip-vocab | tee logs/logs_occupation_classification=$model_name+hammer=$c_hammer+seed=$s+_$BLOCK;
      done;
    done;
  done;
done;
