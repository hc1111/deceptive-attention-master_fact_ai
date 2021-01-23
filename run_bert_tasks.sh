export PYTHONPATH="pytorch-pretrained-BERT"
mkdir -p logs




for TASK in  "twitter_bert"; do
    for OPT in "mean" ; do
        for lambda in 0.1; do
            echo "starting for $TASK, lambda = $lambda ($OPT)";

            if [ "$TASK" == "twitter_bert" ]; then
                PROCESSOR="sst-wiki";
            else
                PROCESSOR="pronoun";
            fi
            echo "processor = $PROCESSOR";
              python pytorch-pretrained-BERT/examples/run_classifier.py \
              --name $TASK \
              --data_dir  data/$TASK\
              --bert_model bert-base-uncased \
              --do_train \
              --do_eval \
              --do_lower_case \
              --max_seq_length 200 \
              --train_batch_size 1 \
              --learning_rate 2e-5 \
              --num_train_epochs 4 \
              --output_dir output/$TASK\_$OPT\_$lambda \
              --hammer_coeff $lambda \
              --first_run \
              --input_processor_type $PROCESSOR \
              --att_opt_func $OPT | tee -a logs/$TASK\_$OPT\_$lambda.log;
            echo "Done for $TASK, lambda = $lambda ($OPT) ...";
        done;
    done;
done;
