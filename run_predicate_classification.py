# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import sys
import tensorflow as tf
from tqdm import tqdm


import keras
from keras.layers.core import *
from tkan import TKAN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bert")))
from bert import modeling
from bert import optimization
from bert import tokenization
from bert import BayesianLinear
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Bar")))
from Bar import general_utils

from bin.evaluation.evaluate_classifiction import do_evaluation

flags = tf.flags
#设置显卡CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = " 0 "
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "pre_output_dir", None,
    "The output directory where the model checkpoints will be written.")


## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SKE_2019_Multi_Label_Classification_Processor(DataProcessor):
    """Processor for the SKE_2019 data set"""

    # SKE_2019 data from http://lic2019.ccf.org.cn/kg
    def __init__(self):
        self.language = "zh"

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), encoding='utf-8') as token_in_f:
            with open(os.path.join(data_dir, "predicate_out.txt"), encoding='utf-8') as predicate_out_f:
                token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                predicate_label_list = [seq.replace("\n", '') for seq in predicate_out_f.readlines()]
                assert len(token_in_list) == len(predicate_label_list)
                examples = list(zip(token_in_list, predicate_label_list))
                return examples

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, os.path.join("test", "token_in.txt")), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            examples = token_in_list
            return self._create_example(examples, "test")

    def get_labels(self):
        return ['别名','工作频段','研发公司','所属国家','部署平台','组成单元','装备种类','服役单位','研发时间','参加战役',
         '具有功能','测向精度','技术特点','研发背景','实际应用']
    

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_str = line
                predicate_label_str = '组成单元'
            else:
                text_str = line[0]
                predicate_label_str = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_str, text_b=None, label=predicate_label_str))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * len(label_list),
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = example.text_a.split(" ")
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_list = example.label.split(" ")
    label_ids = _predicate_label_to_id(label_list, label_map)

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature

def _predicate_label_to_id(predicate_label, predicate_label_map):
    predicate_label_map_length = len(predicate_label_map)
    predicate_label_ids = [0] * predicate_label_map_length
    for label in predicate_label:
        #print(predicate_label)
        predicate_label_ids[predicate_label_map[label]] = 1
    return predicate_label_ids

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, label_length,
                                is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([label_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]

            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #            BERT-Tkan-贝叶斯网络
    # If you want to use the token-level output, use model.get_sequence_output() get_pooled_output()
    # instead.
    output_layer = model.get_sequence_output()
    tkan_layer_1 = TKAN(units=1000, return_sequences=False, use_bias=False)
    output_layer= tkan_layer_1(output_layer)


    hidden_size = output_layer.shape[-1].value
    bayesian_dense = BayesianLinear.BayesianLinearLayer(num_labels, w_prior=None, b_prior=None)
    outputs = bayesian_dense(output_layer, training=is_training)

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        # 非CRF
        probabilities = tf.sigmoid(outputs)
        label_ids = tf.cast(labels, tf.float32)
        per_example_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=label_ids), axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, outputs, probabilities



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        #tf.logging.info("*** Features ***")
        #for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
            

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        
        #tf.logging.info("**** Trainable Variables ****")
        #for var in tvars:
        #    init_string = ""
        #    if var.name in initialized_variable_names:
        #        init_string = ", *INIT_FROM_CKPT*"
        #    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                    init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 添加一个损失摘要   --Loss图像
            tf.summary.scalar('Predicate_classification', total_loss, family='Loss')
            # 计算总的训练步骤
            global_step = tf.train.get_global_step()
            total_steps = params['train_steps']

            summary_op = tf.summary.merge_all()

            # 创建一个SummarySaverHook来保存摘要
            summary_hook = tf.train.SummarySaverHook(
                save_steps=params['save_summary_steps'],
                output_dir=params['Loss_dir'],
                summary_op=summary_op
            )

            
            # 创建ProgressHook实例
            progress_hook = general_utils.ProgressHook(total_steps=total_steps)
            
            #train_op = optimization.create_optimizer(
            #    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

        ##########################################
            #创建一个学习率调度器
            scheduled_learning_rate = tf.train.polynomial_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=num_train_steps,
                end_learning_rate=0.0,
                power=1.0
            )

            # 创建优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=scheduled_learning_rate)

            # 如果您需要预热步骤，可以使用以下方式结合
            warmup_steps = num_warmup_steps
            warmup_learning_rate = (scheduled_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
            effective_learning_rate = tf.cond(
                global_step < warmup_steps,
                lambda: warmup_learning_rate,
                lambda: scheduled_learning_rate
            )

            # 确保更新 global_step
            train_op = optimizer.minimize(total_loss, global_step=global_step)
      
      
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[progress_hook, summary_hook])
                # 添加进度条hook
                
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):
                predict_ids = tf.cast(probabilities > 0.5, tf.int32)
                label_ids = tf.cast(label_ids, tf.int32)
                elements_equal = tf.cast(tf.equal(predict_ids, label_ids), tf.int32)
                # change [batch_size, class_numbers] to [1, batch_size]
                row_predict_ids = tf.reduce_sum(elements_equal, -1)
                row_label_ids = tf.reduce_sum(tf.ones_like(label_ids), -1)
                accuracy = tf.metrics.accuracy(
                    labels=row_label_ids, predictions=row_predict_ids)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, probabilities, is_real_example])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn
                )  
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)  # 设置日志级别为INFO

    processors = {
        "ske_2019": SKE_2019_Multi_Label_Classification_Processor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()
    label_length = len(label_list)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)


    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.do_predict:
        num_train_steps = num_train_steps
    else :
        num_train_steps = num_train_steps / FLAGS.num_train_epochs * 10
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        params={"train_steps": num_train_steps,
                "save_summary_steps": 100,
                "Loss_dir": "Loss/predicate_classification",
                "eval_dir": "Eval/predicate_classification"
            }
        )

    
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            label_length=label_length,
            is_training=True,
            drop_remainder=True)
      
        # 准备评估数据
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.pre_output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)
        num_actual_predict_examples = len(predict_examples)                                        
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        
        predict_drop_remainder = True

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            label_length=label_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        
        #print("num_actual_predict_examples:", num_actual_predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())
        

        summary_writer = tf.summary.FileWriter("Loss/eval")
        # 训练和评估循环

        output_score_value_file = os.path.join(FLAGS.pre_output_dir, "predicate_score_value.txt")
        output_predicate_predict_file = os.path.join(FLAGS.pre_output_dir, "predicate_predict.txt")
        for epoch in tqdm(range(int(FLAGS.num_train_epochs / 10) + 1), desc="Epochs", unit="epoch", position=0):
            if epoch !=0:
                # 创建ProgressHook实例
                #progress_hook = general_utils.ProgressHook(total_steps=range(int(FLAGS.num_train_epochs)))
                tf.logging.info(f"Starting training epoch {epoch+1}")
                estimator.train(input_fn=train_input_fn, steps=num_train_steps)

            tf.logging.info(f"Starting evaluation for epoch {epoch+1}")
            result = estimator.predict(input_fn=predict_input_fn)

            with tf.gfile.GFile(output_score_value_file, "w") as score_value_writer:
                with tf.gfile.GFile(output_predicate_predict_file, "w") as predicate_predict_writer:
                    num_written_lines = 0
                    tf.logging.info("***** Predict results *****")
                    for (i, prediction) in enumerate(result):
                        probabilities = prediction["probabilities"]
                        if i >= num_actual_predict_examples:
                            break
                        output_line_score_value = " ".join(
                            str(class_probability)
                            for class_probability in probabilities) + "\n"
                        predicate_predict = []
                        for idx, class_probability in enumerate(probabilities):
                            if class_probability > 0.5:
                                predicate_predict.append(label_list[idx])
                        output_line_predicate_predict = " ".join(predicate_predict) + "\n"
                        predicate_predict_writer.write(output_line_predicate_predict)
                        score_value_writer.write(output_line_score_value)
                        num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples

            eval_results = do_evaluation()

            for key, value in eval_results.items():
                # 创建一个摘要操作
                summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
                # 运行摘要操作并写入日志
                summary_writer.add_summary(summary, epoch)

            output_eval_file = os.path.join(FLAGS.pre_output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(eval_results.keys()):
                    tf.logging.info("  %s = %s", key, str(eval_results[key]))
                    writer.write("%s = %s\n" % (key, str(eval_results[key])))

        # 关闭摘要写入器
        summary_writer.close()

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            label_length=label_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        #print("num_actual_predict_examples:", num_actual_predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        #predict_drop_remainder = True if FLAGS.use_tpu else False
        #由于上面语句会导致做预测是batch_size是动态的，为了将batch_size设置为固定值，使用下面：
        predict_drop_remainder = True

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            label_length=label_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_score_value_file = os.path.join(FLAGS.output_dir, "predicate_score_value.txt")
        output_predicate_predict_file = os.path.join(FLAGS.output_dir, "predicate_predict.txt")
        with tf.gfile.GFile(output_score_value_file, "w") as score_value_writer:
            with tf.gfile.GFile(output_predicate_predict_file, "w") as predicate_predict_writer:
                num_written_lines = 0
                tf.logging.info("***** Predict results *****")
                for (i, prediction) in enumerate(result):
                    probabilities = prediction["probabilities"]
                    if i >= num_actual_predict_examples:
                        break
                    output_line_score_value = " ".join(
                        str(class_probability)
                        for class_probability in probabilities) + "\n"
                    predicate_predict = []
                    for idx, class_probability in enumerate(probabilities):
                        if class_probability > 0.5:
                            predicate_predict.append(label_list[idx])
                    output_line_predicate_predict = " ".join(predicate_predict) + "\n"
                    predicate_predict_writer.write(output_line_predicate_predict)
                    score_value_writer.write(output_line_score_value)
                    num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

        


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
