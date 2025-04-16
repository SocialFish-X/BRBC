python run_predicate_classification.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/predicate_classifiction/XMW_classification_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--max_seq_length=128 \
--train_batch_size=64 \
--learning_rate=2e-5 \
--num_train_epochs=200.0 \
--output_dir=./output/predicate_classification_model/epochs6

