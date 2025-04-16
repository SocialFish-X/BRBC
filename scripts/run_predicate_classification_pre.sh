python run_predicate_classification.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/predicate_classifiction/XMW_classification_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/predicate_classification_model/epochs6/model.ckpt-2740 \
  --max_seq_length=128 \
  --output_dir=./output/predicate_infer_out/epochs6/ckpt2740
