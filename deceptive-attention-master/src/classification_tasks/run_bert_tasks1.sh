export PYTHONPATH="pytorch-pretrained-BERT"
mkdir -p logs




for TASK in "occupation-classification" "pronoun-bert" "sst-wiki-bert"; do
  for BLOCK in 'none'; do
    for OPT in "mean" "max"; do 
        for lambda in 0.0; do
            echo "starting for $TASK, lambda = $lambda ($OPT)";

            if [ "$TASK" == "sst-wiki-bert" ]; then
                PROCESSOR="sst-wiki";
            else
                PROCESSOR="pronoun";
            fi
            echo "processor = $PROCESSOR";
            CUDA_VISIBLE_DEVICES=0  python pytorch-pretrained-BERT/examples/run_classifier1.py \
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
              --output_dir output/$TASK\_$OPT_$BLOCK\\_$lambda \
              --hammer_coeff $lambda \
              --first_run \
              --input_processor_type $PROCESSOR \
              --att_opt_func $OPT | tee -a logs/$TASK\_$OPT\_$BLOCK\_$lambda.log;
            echo "Done for $TASK, lambda = $lambda ($OPT) ...";
        done;
    done;
  done;
done;
