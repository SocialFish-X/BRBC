python run_sequnce_labeling.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/subject_object_labeling/sequence_labeling_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/epochs9/model.ckpt-13140 \
  --max_seq_length=128 \
  --output_dir=./output/sequnce_infer_out/epochs9/ckpt13140
