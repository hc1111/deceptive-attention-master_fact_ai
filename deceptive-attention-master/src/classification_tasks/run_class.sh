
# Activate your environment
source activate attention

mkdir -p output_emb

# run the pronoun dataset
for model_name in emb-att emb-lstm-att; do
    for s in $(seq 1 5); do
        for c_hammer in 0.0 0.1 1.0; do
            # on SST+Wiki dataset
            unbuffer python main.py --num-epochs 15 --num-visualize 5 --loss-hammer $c_hammer  --model $model_name --task sst-wiki --use-block-file --seed $s | tee -a output_emb/logs_sst_wiki_model=$model_name+hammer=$c_hammer+seed=$s;
            # on pronoun dataset
            unbuffer python main.py --num-epochs 15 --num-visualize 5 --loss-hammer $c_hammer  --model $model_name --task pronoun  --block-words "he" "she" "her" "his" "him" "himself" "herself" --seed $s | tee -a output_emb/logs_pronoun_model=$model_name+hammer=$c_hammer+seed=$s;
            # occupation classification
            unbuffer python main.py --num-epochs 5 --num-visualize 100 --loss-hammer $c_hammer  --model $model_name --block-words "he" "she" "her" "his" "him" "himself" "herself" "hers" "mr" "mrs" "ms" "mr." "mrs." "ms." --task occupation-classification --seed $s --clip-vocab | tee output_emb/logs_occupation_classification=$model_name+hammer=$c_hammer+seed=$s;
        done;
    done;
done;
