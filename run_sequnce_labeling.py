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
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Lambda, Layer, Input
from keras.models import Sequential
from keras import Model
from keras.layers import GlobalAveragePooling1D
import keras.backend as K
from tqdm import tqdm
import produce_submit_json_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bert")))
from bert import modeling
from bert import optimization
from bert import tokenization
from bert import tf_metrics


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Bar")))
from Bar import general_utils

from bin.evaluation.evaluate_labeling import do_eva
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


flags = tf.flags

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

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string(
    "pre_output_dir", None,
    "The output directory where the model checkpoints will be written.")

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

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

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

    def __init__(self, guid, text_token, token_label):
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
        self.text_token = text_token
        self.token_label = token_label


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
                 token_label_ids,
                 predicate_label_id,
                 sequence_length,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_label_ids = token_label_ids
        self.predicate_label_id = predicate_label_id
        self.sequence_length = sequence_length
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


class SKE_2019_Sequence_labeling_Processor(DataProcessor):
    """Processor for the SKE_2019 data set"""

    # SKE_2019 data from http://lic2019.ccf.org.cn/kg

    def __init__(self):
        self.language = "zh"

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), encoding='utf-8') as token_in_f:
            with open(os.path.join(data_dir, "token_label_and_one_prdicate_out.txt"), encoding='utf-8') as token_label_out_f:
                    token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                    token_label_out_list = [seq.replace("\n", '') for seq in token_label_out_f.readlines()]
                    assert len(token_in_list) == len(token_label_out_list)
                    examples = list(zip(token_in_list, token_label_out_list))
                    return examples

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, os.path.join("test", "token_in_and_one_predicate.txt")), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            examples = token_in_list
            return self._create_example(examples, "test")


    def get_token_labels(self):
        BIO_token_labels = ["[Padding]", "[category]", "[##WordPiece]", "[CLS]", "[SEP]", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "O"]  #id 0 --> [Paddding]
        return BIO_token_labels

    def get_predicate_labels(self):
        return ['别名','工作频段','研发公司','所属国家','部署平台','组成单元','装备种类','服役单位','研发时间','参加战役',
         '具有功能','测向精度','技术特点','研发背景','实际应用']

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_token = line
                token_label = None
            else:
                text_token = line[0]
                token_label = line[1]
            examples.append(
                InputExample(guid=guid, text_token=text_token, token_label=token_label))
        return examples


def convert_single_example(ex_index, example, token_label_list, predicate_label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            token_label_ids=[0] * max_seq_length,
            predicate_label_id = [0],
            sequence_length = [0],
            is_real_example=False)

    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i

    predicate_label_map = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_map[label] = i

    text_token = example.text_token.split("\t")[0].split(" ")
    if example.token_label is not None:
        token_label = example.token_label.split("\t")[0].split(" ")
    else:
        token_label = ["O"] * len(text_token)
    assert len(text_token) == len(token_label)
    print("text_token:", len(text_token))

    sequence_length = len(text_token)
    sequence_length = min(sequence_length, 126)
    
    text_predicate = example.text_token.split("\t")[1]
    if example.token_label is not None:
        token_predicate = example.token_label.split("\t")[1]
    else:
        token_predicate = text_predicate
    assert text_predicate == token_predicate

    tokens_b = [text_predicate] * len(text_token)
    
    predicate_id = predicate_label_map[text_predicate]


    _truncate_seq_pair(text_token, tokens_b, max_seq_length - 3)

    tokens = []
    token_label_ids = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[CLS]"])

    for token, label in zip(text_token, token_label):
        tokens.append(token)
        segment_ids.append(0)
        token_label_ids.append(token_label_map[label])

    tokens.append("[SEP]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #bert_tokenizer.convert_tokens_to_ids(["[SEP]"]) --->[102]
    bias = 1 #1-100 dict index not used
    for token in tokens_b:
      input_ids.append(predicate_id + bias) #add  bias for different from word dict
      segment_ids.append(1)
      token_label_ids.append(token_label_map["[category]"])

    input_ids.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0]) #102
    segment_ids.append(1)
    token_label_ids.append(token_label_map["[SEP]"])

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        token_label_ids.append(0)
        tokens.append("[Padding]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(token_label_ids) == max_seq_length

    #if ex_index < 5:
    #    tf.logging.info("*** Example ***")
    #    tf.logging.info("guid: %s" % (example.guid))
    #    tf.logging.info("tokens: %s" % " ".join(
    #        [tokenization.printable_text(x) for x in tokens]))
    #    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #    tf.logging.info("token_label_ids: %s" % " ".join([str(x) for x in token_label_ids]))
    #    tf.logging.info("predicate_id: %s" % str(predicate_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        token_label_ids=token_label_ids,
        predicate_label_id=[predicate_id],
        sequence_length = [sequence_length],
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, token_label_list, predicate_label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        print(example.text_token)
        print(".............")
        feature = convert_single_example(ex_index, example, token_label_list, predicate_label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
            

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["token_label_ids"] = create_int_feature(feature.token_label_ids)
        features["predicate_label_id"] = create_int_feature(feature.predicate_label_id)
        features["sequence_length"] = create_int_feature(feature.sequence_length)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length,is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "token_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "predicate_label_id": tf.FixedLenFeature([], tf.int64),
        "sequence_length": tf.FixedLenFeature([], tf.int64),
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
                 token_label_ids, predicate_label_id, num_token_labels, num_predicate_labels,
                 use_one_hot_embeddings, sequence_length):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. float Tensor of shape [batch_size, hidden_size]
    predicate_output_layer = model.get_pooled_output()

    intent_hidden_size = predicate_output_layer.shape[-1].value

    # BiLSTM layer
    #bilstm_output = Bidirectional(LSTM(intent_hidden_size, return_sequences=True))(predicate_output_layer)

    # Pooling layer
    #predicate_output_layer = tf.keras.layers.GlobalAveragePooling1D()(bilstm_output)

    #intent_hidden_size = predicate_output_layer.shape[-1].value

    predicate_output_weights = tf.get_variable(
        "predicate_output_weights", [num_predicate_labels, intent_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    predicate_output_bias = tf.get_variable(
        "predicate_output_bias", [num_predicate_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("predicate_loss"):
        if is_training:
            # I.e., 0.1 dropout
            predicate_output_layer = tf.nn.dropout(predicate_output_layer, keep_prob=0.9)

        predicate_logits = tf.matmul(predicate_output_layer, predicate_output_weights, transpose_b=True)
        predicate_logits = tf.nn.bias_add(predicate_logits, predicate_output_bias)
        predicate_probabilities = tf.nn.softmax(predicate_logits, axis=-1)
        predicate_prediction = tf.argmax(predicate_probabilities, axis=-1, output_type=tf.int32)
        predicate_labels = tf.one_hot(predicate_label_id, depth=num_predicate_labels, dtype=tf.float32)
        predicate_per_example_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=predicate_logits, labels=predicate_labels), -1)
        predicate_loss = tf.reduce_mean(predicate_per_example_loss)


    #     """Gets final hidden layer of encoder.
    #
    #     Returns:
    #       float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    #       to the final hidden of the transformer encoder.
    #     """
    token_label_output_layer = model.get_sequence_output()

    token_label_hidden_size = token_label_output_layer.shape[-1].value

     # 自定义动态残差层
    class DynamicResidualLayer(Layer):
        def __init__(self, **kwargs):
            super(DynamicResidualLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            super(DynamicResidualLayer, self).build(input_shape)

        def call(self, inputs, **kwargs):
            x, skip_connection = inputs
            return K.concatenate([x, skip_connection], axis=-1)

        def compute_output_shape(self, input_shape):
            return input_shape[0][0], input_shape[0][1], input_shape[0][2] * 2
    print("token_label_output_layer:", token_label_output_layer.shape)

    # 堆叠式BiLSTM + 动态残差模型
    d = 0.9
    inputs = token_label_output_layer

    # 初始BiLSTM层（输出维度128）
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)  # 注意维度变化
    x = Dropout(d)(x)

    # 第一个动态残差层
    skip_connection1 = Bidirectional(LSTM(64, return_sequences=True))(x)  # 输出维度128
    x = DynamicResidualLayer()([x, skip_connection1])  # 连接后维度256

    # 第二个动态残差层
    skip_connection2 = Bidirectional(LSTM(128, return_sequences=True))(x)  # 输出维度256
    x = DynamicResidualLayer()([x, skip_connection2])  # 连接后维度512

    # 第三个动态残差层
    skip_connection3 = Bidirectional(LSTM(256, return_sequences=True))(x)  # 输出维度512
    x = DynamicResidualLayer()([x, skip_connection3])  # 连接后维度1024

    # 后续保持不变
    x = Dropout(d)(x)
    x = Dense(token_label_hidden_size, activation='relu')(x)
    token_label_output_layer = x




    token_label_output_weights = tf.get_variable(
        "token_label_output_weights", [num_token_labels, token_label_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    token_label_output_bias = tf.get_variable(
        "token_label_output_bias", [num_token_labels], initializer=tf.zeros_initializer()
    )

    with tf.variable_scope("token_label_loss"):
    # 添加CRF转移矩阵参数
        transition_params = tf.get_variable(
            "transitions", 
            shape=[num_token_labels, num_token_labels],
            initializer=tf.zeros_initializer()
        )
    
        if is_training:
            token_label_output_layer = tf.nn.dropout(token_label_output_layer, keep_prob=0.9)
        
        # 保持原有的logits计算
        token_label_output_layer = tf.reshape(token_label_output_layer, [-1, token_label_hidden_size])
        token_label_logits = tf.matmul(token_label_output_layer, token_label_output_weights, transpose_b=True)
        token_label_logits = tf.nn.bias_add(token_label_logits, token_label_output_bias)
        token_label_logits = tf.reshape(token_label_logits, [-1, FLAGS.max_seq_length, num_token_labels])
        
        # 计算序列实际长度（需要确保input_mask已定义）
        sequence_length = tf.reduce_sum(input_mask, axis=1)  # input_mask应为[batch_size, seq_length]
        
        # 替换损失计算为CRF版本
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            inputs=token_label_logits,
            tag_indices=token_label_ids,
            sequence_lengths=sequence_length,
            transition_params=transition_params
        )
        token_label_loss = tf.reduce_mean(-log_likelihood)  # 也可用reduce_sum保持与原损失尺度一致
        
        # 替换预测方法为CRF解码
        token_label_predictions, _ = tf.contrib.crf.crf_decode(
            potentials=token_label_logits,
            transition_params=transition_params,
            sequence_length=sequence_length
        )
        
        # 保持与原代码兼容的返回值
        token_label_per_example_loss = -log_likelihood  # 保持每个样本的损失值

    # 保持总损失计算方式不变
    loss = token_label_loss  # 根据实际需求可能需要加上predicate_loss

    return (loss,
            predicate_loss, predicate_per_example_loss, predicate_probabilities, predicate_prediction,
            token_label_loss, token_label_logits, token_label_predictions)


def model_fn_builder(bert_config,num_token_labels, num_predicate_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        #tf.logging.info("*** Features ***")
        #for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        token_label_ids = features["token_label_ids"]
        predicate_label_id = features["predicate_label_id"]
        sequence_length = features["sequence_length"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(token_label_ids), dtype=tf.float32) #TO DO

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,
         predicate_loss, predicate_per_example_loss, predicate_probabilities, predicate_prediction,
         token_label_loss, token_label_logits, token_label_predictions) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            token_label_ids, predicate_label_id, num_token_labels, num_predicate_labels,
            use_one_hot_embeddings, sequence_length)

        
        # 添加一个损失摘要   --Loss图像
        tf.summary.scalar('Sequence_labeling', total_loss, family='Loss')

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
            #init_string = ""
            #if var.name in initialized_variable_names:
            #    init_string = ", *INIT_FROM_CKPT*"
            #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

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

            def metric_fn(predicate_loss,  predicate_probabilities, token_label_ids, token_label_logits, is_real_example):
                predicate_prediction = tf.argmax(predicate_probabilities, axis=-1, output_type=tf.int32)
                token_label_predictions = tf.argmax(token_label_logits, axis=-1, output_type=tf.int32)
                token_label_pos_indices_list = list(range(num_token_labels))[4:]  # ["[Padding]","[##WordPiece]", "[CLS]", "[SEP]"] + seq_out_set
                pos_indices_list = token_label_pos_indices_list[:-1]  # do not care "O"
                token_label_precision_macro = tf_metrics.precision(token_label_ids, token_label_predictions, num_token_labels,
                                                                   pos_indices_list, average="macro")
                token_label_recall_macro = tf_metrics.recall(token_label_ids, token_label_predictions, num_token_labels,
                                                             pos_indices_list, average="macro")
                token_label_f_macro = tf_metrics.f1(token_label_ids, token_label_predictions, num_token_labels, pos_indices_list,
                                             average="macro")
                token_label_precision_micro = tf_metrics.precision(token_label_ids, token_label_predictions, num_token_labels,
                                                                   pos_indices_list, average="micro")
                token_label_recall_micro = tf_metrics.recall(token_label_ids, token_label_predictions, num_token_labels,
                                                             pos_indices_list, average="micro")
                token_label_f_micro = tf_metrics.f1(token_label_ids, token_label_predictions, num_token_labels, pos_indices_list,
                                             average="micro")
                # token_label_loss = tf.metrics.mean(values=token_label_per_example_loss, weights=is_real_example)
                predicate_loss = tf.metrics.mean(values=predicate_loss)
                return {
                    "eval_predicate_loss": predicate_loss,
                    "predicate_prediction": predicate_prediction,
                    "eval_token_label_precision(macro)": token_label_precision_macro,
                    "eval_token_label_recall(macro)": token_label_recall_macro,
                    "eval_token_label_f(macro)": token_label_f_macro,
                    "eval_token_label_precision(micro)": token_label_precision_micro,
                    "eval_token_label_recall(micro)": token_label_recall_micro,
                    "eval_token_label_f(micro)": token_label_f_micro,
                }

            eval_metrics = (metric_fn,
                            [predicate_loss, predicate_probabilities,
                             token_label_ids, token_label_logits, is_real_example])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"predicate_probabilities": predicate_probabilities,
                             "predicate_prediction":   predicate_prediction,
                             "token_label_predictions": token_label_predictions},
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn




def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ske_2019": SKE_2019_Sequence_labeling_Processor,
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

    token_label_list = processor.get_token_labels()
    predicate_label_list = processor.get_predicate_labels()

    num_token_labels = len(token_label_list)
    num_predicate_labels = len(predicate_label_list)

    token_label_id2label = {}
    for (i, label) in enumerate(token_label_list):
        token_label_id2label[i] = label
    predicate_label_id2label = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_id2label[i] = label

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
        num_token_labels=num_token_labels,
        num_predicate_labels=num_predicate_labels,
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
                "Loss_dir": "Loss/run_sequnce_labeling",
                "eval_dir": "Eval/run_sequnce_labeling"
            })

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, token_label_list, predicate_label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        #estimator.train(input_fn=train_input_fn, steps=num_train_steps)
        


        summary_writer = tf.summary.FileWriter("Loss/eval_sequence_labeling")
        # 训练和评估循环
        for epoch in tqdm(range(int(FLAGS.num_train_epochs / 10) + 1), desc="Epochs", unit="epoch", position=0):
            if epoch !=0:
                # 创建ProgressHook实例
                #progress_hook = general_utils.ProgressHook(total_steps=range(int(FLAGS.num_train_epochs)))
                tf.logging.info(f"Starting training epoch {epoch+1}")
                estimator.train(input_fn=train_input_fn, steps=num_train_steps)
            predict_examples = processor.get_test_examples(FLAGS.data_dir)
            predict_file = os.path.join(FLAGS.pre_output_dir, "predict.tf_record")
            file_based_convert_examples_to_features(predict_examples, token_label_list, predicate_label_list,
                                                    FLAGS.max_seq_length, tokenizer,
                                                    predict_file)
            num_actual_predict_examples = len(predict_examples)
            #if FLAGS.use_tpu:
                # TPU requires a fixed batch size for all batches, therefore the number
                # of examples must be a multiple of the batch size, or else examples
                # will get dropped. So we pad with fake examples which are ignored
                # later on.
            #    while len(predict_examples) % FLAGS.predict_batch_size != 0:
            #        predict_examples.append(PaddingInputExample())

            tf.logging.info("***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(predict_examples), num_actual_predict_examples,
                            len(predict_examples) - num_actual_predict_examples)
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

            predict_drop_remainder = False
            predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)
            tf.logging.info(f"Starting evaluation for epoch {epoch+1}")
            result = estimator.predict(input_fn=predict_input_fn)
            token_label_output_predict_file = os.path.join(FLAGS.pre_output_dir, "token_label_predictions.txt")
            predicate_output_predict_file = os.path.join(FLAGS.pre_output_dir, "predicate_predict.txt")
            predicate_output_probabilities_file = os.path.join(FLAGS.pre_output_dir, "predicate_probabilities.txt")
            with open(token_label_output_predict_file, "w", encoding='utf-8') as token_label_writer:
                with open(predicate_output_predict_file, "w", encoding='utf-8') as predicate_predict_writer:
                    with open(predicate_output_probabilities_file, "w", encoding='utf-8') as predicate_probabilities_writer:
                        num_written_lines = 0
                        tf.logging.info("***** token_label predict and predicate labeling results *****")
                        for (i, prediction) in enumerate(result):
                            token_label_prediction = prediction["token_label_predictions"]
                            predicate_probabilities = prediction["predicate_probabilities"]
                            predicate_prediction = prediction["predicate_prediction"]
                            if i >= num_actual_predict_examples:
                                break
                            token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction) + "\n"
                            token_label_writer.write(token_label_output_line)
                            predicate_predict_line = predicate_label_id2label[predicate_prediction]
                            predicate_predict_writer.write(predicate_predict_line + "\n")
                            predicate_probabilities_line = " ".join(str(sigmoid_logit) for sigmoid_logit in predicate_probabilities) + "\n"
                            predicate_probabilities_writer.write(predicate_probabilities_line)
                            num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples
            
            #####相当于运行run_sequnce_labeling.py文件#####################
            TEST_DATA_DIR = "bin/subject_object_labeling/sequence_labeling_data/test"
            MODEL_OUTPUT_DIR = None
            OUT_RESULTS_DIR = "output/final_text_spo_list_result"
            Competition_Mode = True
            spo_list_manager = produce_submit_json_file.Sorted_relation_and_entity_list_Management(TEST_DATA_DIR, MODEL_OUTPUT_DIR, Competition_Mode=Competition_Mode)
            spo_list_manager.produce_output_file(OUT_RESULTS_DIR=OUT_RESULTS_DIR, keep_empty_spo_list=True)
            #############################################################
            eval_results = do_eva()
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
            eval_examples, token_label_list, predicate_label_list, FLAGS.max_seq_length, tokenizer, eval_file)

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

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
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
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

        file_based_convert_examples_to_features(predict_examples, token_label_list, predicate_label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        token_label_output_predict_file = os.path.join(FLAGS.output_dir, "token_label_predictions.txt")
        predicate_output_predict_file = os.path.join(FLAGS.output_dir, "predicate_predict.txt")
        predicate_output_probabilities_file = os.path.join(FLAGS.output_dir, "predicate_probabilities.txt")
        with open(token_label_output_predict_file, "w", encoding='utf-8') as token_label_writer:
            with open(predicate_output_predict_file, "w", encoding='utf-8') as predicate_predict_writer:
                with open(predicate_output_probabilities_file, "w", encoding='utf-8') as predicate_probabilities_writer:
                    num_written_lines = 0
                    tf.logging.info("***** token_label predict and predicate labeling results *****")
                    for (i, prediction) in enumerate(result):
                        token_label_prediction = prediction["token_label_predictions"]
                        predicate_probabilities = prediction["predicate_probabilities"]
                        predicate_prediction = prediction["predicate_prediction"]
                        if i >= num_actual_predict_examples:
                            break
                        token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction) + "\n"
                        token_label_writer.write(token_label_output_line)
                        predicate_predict_line = predicate_label_id2label[predicate_prediction]
                        predicate_predict_writer.write(predicate_predict_line + "\n")
                        predicate_probabilities_line = " ".join(str(sigmoid_logit) for sigmoid_logit in predicate_probabilities) + "\n"
                        predicate_probabilities_writer.write(predicate_probabilities_line)
                        num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
