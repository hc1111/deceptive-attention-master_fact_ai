
# Activate your environment
source activate attention 

export PYTHONPATH="pytorch-pretrained-BERT"
mkdir -p logs_bert


for SEED in $(seq 1 5); do
    for TASK in "occupation-classification" "pronoun-bert" "sst-wiki-bert"; do
        for OPT in "max" "mean"; do 
            for lambda in 0.0 0.1 1.0; do
                if [ "$TASK" == "sst-wiki-bert" ]; then
                    PROCESSOR="sst-wiki";
                else
                    PROCESSOR="pronoun";
                fi
                if [ "$TASK" == "sst-wiki-bert" ]; then
                    NAME="sst-wiki_bert";
                elif [ "$TASK" == "pronoun-bert" ]; then
                    NAME="pronoun_bert";                
                else
                    NAME="occupation-classification_bert";
                fi
                unbuffer python pytorch-pretrained-BERT/examples/run_classifier.py \
                --name $TASK \
                --data_dir  data/$TASK\
                --bert_model bert-base-uncased \
                --do_train \
                --do_eval \
                --do_lower_case \
                --max_seq_length 256 \
                --train_batch_size 16 \
                --learning_rate 2e-5 \
                --num_train_epochs 4 \
                --output_dir output_bert/$NAME\_$OPT\_$lambda\_$SEED \
                --hammer_coeff $lambda \
                --first_run \
                --input_processor_type $PROCESSOR \
                --seed $SEED \
                --att_opt_func $OPT | tee -a logs_bert/$NAME\_$OPT\_$lambda\_$SEED.log;
            done;
        done;
    done;
done;