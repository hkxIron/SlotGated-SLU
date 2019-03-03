# add crf: https://github.com/InsaneLife/Joint-NLU/blob/master/train.py

import os
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
from model import createModel
from utils import createVocabulary, load_embedding, loadVocabulary, computeF1Score, DataProcessor

# todo: 1. word pre-train embedding, gru, crf, lr decay

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.register("type", "bool", lambda v: v.lower() in ["true", "1"])
# Network
parser.add_argument("--num_units", type=int, default=64, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full(default) | intent_only
                                                                    full: full attention model
                                                                    intent_only: intent attention model""")
parser.add_argument("--use_crf", type="bool", default="true", help="""use crf for seq labeling""")
parser.add_argument("--cell", type=str, default='gru', help="""rnn cell""")

# Training Environment
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--batch_size_add", type=int, default=4, help="Batch size add.")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false', dest='early_stop',
                    help="Disable early stop, which is based on sentence level accuracy.")
parser.add_argument("--patience", type=int, default=10, help="Patience to wait before stop.")
# learn rate param
parser.add_argument("--learning_rate_decay", type=str, default='1', help="learning_rate_decay")
parser.add_argument("--learning_rate", type=float, default=0.001, help="The initial learning rate.")
parser.add_argument("--decay_steps", type=int, default=280*4, help="decay_steps.")
parser.add_argument("--decay_rate", type=float, default=0.9, help="decay_rate.")

# Model and Vocab
parser.add_argument("--dataset", type=str, default='atis', help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")

# Data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")
parser.add_argument("--embedding_path", type=str, default='', help="embedding array's path.")
# embedding_path : ./vocab/google_in_vocab_embedding.npy

arg = parser.parse_args()
# Print arguments
print("args:")
for k, v in sorted(vars(arg).items()):
    print(k, '=', v)
print()
# use full attention or intent only
if arg.model_type == 'full':
    add_final_state_to_intent = True
    remove_slot_attn = False
elif arg.model_type == 'intent_only':
    add_final_state_to_intent = True
    remove_slot_attn = True
else:
    print('unknown model type!')
    exit(1)

# full path to data will be: ./data + dataset + train/test/valid
if arg.dataset == None:
    print('name of dataset can not be None')
    exit(1)
elif arg.dataset == 'snips':
    print('use snips dataset')
elif arg.dataset == 'atis':
    print('use atis dataset')
else:
    print('use own dataset: ', arg.dataset)
full_train_path = os.path.join('./data', arg.dataset, arg.train_data_path)
full_test_path = os.path.join('./data', arg.dataset, arg.test_data_path)
full_valid_path = os.path.join('./data', arg.dataset, arg.valid_data_path)

print("create vocab...")
createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'))
# return map: {'vocab': vocab, 'rev': rev}, vocab: map, rev: array
in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

# Create Training Model
input_data = tf.placeholder(tf.int32, [None, None], name='inputs') # [batch, input_sequence_length]
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length") # [batch]
global_step = tf.Variable(0, trainable=False, name='global_step')
slot_labels = tf.placeholder(tf.int32, [None, None], name='slots') # [batch, input_sequence_length]
slot_weights = tf.placeholder(tf.float32, [None, None], name='slot_weights') # [batch, input_sequence_length]
intent_label = tf.placeholder(tf.int32, [None], name='intent') # [batch]

use_batch_crossent = True
with tf.variable_scope('model'):
    print("create train model")
    training_outputs = createModel(input_data,
                                   len(in_vocab['vocab']),
                                   sequence_length,
                                   len(slot_vocab['vocab']),
                                   len(intent_vocab['vocab']),
                                   remove_slot_attn,
                                   add_final_state_to_intent,
                                   use_crf=arg.use_crf,
                                   layer_size=arg.layer_size,
                                   use_batch_crossent=use_batch_crossent
                                   ) # layer_size:64

# slot_output_logits:[batch * input_sequence_length, slot_size]
# slot_output_logits_crf or use_batch_crossent:[batch, input_sequence_length, slot_size]
slot_output_logits = training_outputs[0]
# intent:[batch, intent_size]
intent_output_logit = training_outputs[1]

with tf.variable_scope('slot_loss'):
    if arg.use_crf:
        """
        log_likelihood: 标量,log-likelihood 
        transition_params: 形状为[num_tags, num_tags] 的转移矩阵
        """
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(inputs=slot_output_logits,
                                                                         tag_indices=slot_labels,
                                                                         sequence_lengths=sequence_length)
        print("log_likelihood:", log_likelihood) # scalar
        # add a training op to tune the parameters.
        slot_loss = tf.reduce_mean(-log_likelihood)
    elif use_batch_crossent:
        """
        交叉熵loss,将所有的sequence展开, 对所有样本求loss平均, 
        loss相对于batch_size是invariable
        这样会有label length bias,因为倾向于将较长的样本预测得更好
        
        Important note: It's worth pointing out that we divide the loss by batch_size, 
        so our hyperparameters are "invariant" to batch_size. 
        Some people divide the loss by (batch_size * num_time_steps),
         which plays down(淡化) the errors made on short sentences. More subtly, 
         our hyperparameters (applied to the former way) can't be used for the latter way.
         For example, if both approaches use SGD with a learning of 1.0, 
         the latter approach(batch_size*num_time_steps) effectively uses a much smaller learning rate of 1 / num_time_steps.
        """
        # slot_labels:[batch, input_sequence_length]
        # slot_outputs:[batch, input_sequence_length, slot_size]
        # crossent: [batch, input_sequence_length]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slot_labels,
                                                                  logits=slot_output_logits)
        print("crossent:", crossent)
        # slot_weights:[batch, input_sequence_length]
        crossent_weighted = crossent*slot_weights
        # mask: [batch_size=8, max_sentence_len=9], 值为bool的矩阵,max_length会根据batch样本里的length自动计算
        mask = tf.sequence_mask(lengths=sequence_length)
        # losses_masked: [ num_of_actual_word_count_in_batch ],为一维的数组
        # 它的shape为一个batch中所有句子的长度之和(除去了padding的元素),每个元素为该句子的ner序列 log(Prob)
        losses_masked = tf.boolean_mask(crossent_weighted, mask)
        slot_loss = tf.reduce_mean(losses_masked) # 如果不用mask,求mean时的个数就会不对
    else:
        """
        (不建议采用)
        交叉熵loss,将所有的batch sequence展开, 
        loss相对于batch*input_sequence_length是invariable
        对所有slot求loss平均,
        这样较长的样本,学习得不太好,因为loss太小
        """
        # [batch, input_sequence_length]
        slot_labels_shape = tf.shape(slot_labels)
        # [batch*input_sequence_length]
        slot_label_vector = tf.reshape(slot_labels, [-1])
        # crossent: [batch_size*input_sequence_length]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slot_label_vector,
                                                                  logits=slot_output_logits)
        crossent = tf.reshape(crossent, slot_labels_shape)
        # slot_weights:[batch, input_sequence_length]
        crossent_weighted = crossent*slot_weights
        # slot_weights:[batch, input_sequence_length]
        slot_loss = tf.reduce_sum(crossent_weighted, axis=1)
        total_size = tf.reduce_sum(slot_weights, axis=1)
        total_size += 1e-12
        slot_loss = slot_loss / total_size

with tf.variable_scope('intent_loss'):
    # intent_label:[batch]
    # intent_output_logit:[batch, intent_size]
    # crossent: [batch]
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent_label,
                                                              logits=intent_output_logit)
    #intent_loss = tf.reduce_sum(crossent) / tf.cast(arg.batch_size, tf.float32)
    intent_loss = tf.reduce_mean(crossent)

params = tf.trainable_variables()
# learning rate decay
learning_rate_decay = tf.train.exponential_decay(learning_rate=arg.learning_rate,
                                                 global_step=global_step,
                                                 decay_steps=arg.decay_steps,
                                                 decay_rate=arg.decay_rate,
                                                 staircase=False)
#妙!可以用decay来初始化Adam
if arg.learning_rate_decay:
    opt = tf.train.AdamOptimizer(learning_rate_decay)
else:
    opt = tf.train.AdamOptimizer(arg.learning_rate)

intent_params = []
slot_params = []
for p in params:
    if not 'slot_' in p.name:
        intent_params.append(p)
    if 'slot_' in p.name \
            or 'bidirectional_rnn' in p.name \
            or 'embedding' in p.name:
        slot_params.append(p)

print("slot params:", slot_params)
print("intent_params:", intent_params)
gradients_slot = tf.gradients(ys=slot_loss, xs=slot_params)
gradients_intent = tf.gradients(ys=intent_loss, xs=intent_params)

clipped_gradients_slot, gradient_norm_slot = tf.clip_by_global_norm(gradients_slot, 5.0)
clipped_gradients_intent, gradient_norm_intent = tf.clip_by_global_norm(gradients_intent, 5.0)

update_slot = opt.apply_gradients(zip(clipped_gradients_slot, slot_params))
update_intent = opt.apply_gradients(zip(clipped_gradients_intent, intent_params), global_step=global_step)

training_outputs = [global_step, slot_loss, update_intent, update_slot, gradient_norm_intent, gradient_norm_slot]
inputs = [input_data, sequence_length, slot_labels, slot_weights, intent_label]

# Create Inference Model
with tf.variable_scope('model', reuse=True):
    print("create infer model")
    inference_output = createModel(input_data,
                                        len(in_vocab['vocab']),
                                        sequence_length,
                                        len(slot_vocab['vocab']),
                                        len(intent_vocab['vocab']),
                                        remove_slot_attn,
                                        add_final_state_to_intent,
                                        use_crf=arg.use_crf,
                                        layer_size=arg.layer_size,
                                        use_batch_crossent=use_batch_crossent,
                                        isTraining=False)

# slot_output_logits:[batch * input_sequence_length, slot_size]
# slot_output_logits_crf or use_batch_crossent:[batch, input_sequence_length, slot_size]
# slot_output_logits = training_outputs[0]

# intent_output_logit:[batch, intent_size]
# intent_output_logit = training_outputs[1]
inference_slot_logits = inference_output[0]
inference_intent_logits = inference_output[1]

if arg.use_crf:
    """
    参数:
    potentials: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor, 
    transition_params: 一个形状为[num_tags, num_tags] 的转移矩阵 
    sequence_length: 一个形状为[batch_size] 的 ,表示batch中每个序列的长度
    
    返回：
    decode_tags:一个形状为[batch_size, max_seq_len] 的tensor,类型是tf.int32.表示最好的序列标记. 
    best_score: 有个形状为[batch_size] 的tensor, 包含每个序列解码标签的分数.
    """
    # inference_slot_logits:[batch, input_sequence_length, slot_size]
    # slot_output_logits_crf or use_batch_crossent:[batch, input_sequence_length, slot_size]
    # inference_slot_output:[batch, input_sequence_length]
    # pred_scores:[batch]
    inference_slot_output, pred_scores = tf.contrib.crf.crf_decode(potentials=inference_slot_logits,
                                                                   transition_params=trans_params,
                                                                   sequence_length=sequence_length)
elif use_batch_crossent:
    # inference_slot_logits:[batch, input_sequence_length, slot_size]
    # inference_slot_output:[batch, input_sequence_length]
    inference_slot_output = tf.nn.softmax(inference_slot_logits, name='slot_output')
else:
    # inference_slot_logits:[batch*input_sequence_length, slot_size]
    # inference_slot_output:[batch*input_sequence_length]
    inference_slot_output = tf.nn.softmax(inference_slot_logits, name='slot_output')

# intent output
# intent_output_logit:[batch, intent_size]
# inference_intent_output:[batch, intent_size]
inference_intent_output = tf.nn.softmax(inference_intent_logits, name='intent_output')

inference_output_list = [inference_intent_output, inference_slot_output]
inference_inputs = [input_data, sequence_length]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

saver = tf.train.Saver()
# gpu setting
gpu_options = tf.GPUOptions(allow_growth=True)

# Start Training
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    logging.info('Training Start ...')

    epochs = 0
    loss = 0.0
    data_processor = None
    line = 0
    num_loss = 0
    step = 0
    no_improve = 0

    # variables to store highest values among epochs, only use 'valid_err' for now
    valid_slot = 0
    test_slot = 0
    valid_intent = 0
    test_intent = 0
    best_valid_score = 0
    test_err = 0

    while True:
        if data_processor == None:
            data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file),
                                           os.path.join(full_train_path, arg.slot_file),
                                           os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab,
                                           intent_vocab)
        in_data, slot_data, slot_weight, length, intents, _, _, _ = data_processor.get_batch(arg.batch_size)
        feed_dict = {input_data.name: in_data,  # 直接用tensor作key也可以
                     slot_labels.name: slot_data,
                     slot_weights.name: slot_weight,
                     sequence_length.name: length,
                     intent_label.name: intents}
        ret = sess.run(training_outputs, feed_dict)
        loss += np.mean(ret[1])

        line += arg.batch_size
        step = ret[0]
        num_loss += 1

        if data_processor.end == 1:
            arg.batch_size += arg.batch_size_add
            line = 0
            data_processor.close()
            data_processor = None
            epochs += 1
            logging.info('Train:')
            logging.info('Step: ' + str(step))
            logging.info('Epochs: ' + str(epochs))
            logging.info('Loss: ' + str(loss / num_loss))
            num_loss = 0
            loss = 0.0

            save_path = os.path.join(arg.model_path, 'train_step_' + str(step) + '_epochs_' + str(epochs) + '.ckpt')
            logging.info("step:{} save model to:{}".format(step,save_path))
            saver.save(sess, save_path)

            def run_validate(in_path, slot_path, intent_path):
                data_processor_valid = DataProcessor(in_path,
                                                     slot_path,
                                                     intent_path,
                                                     in_vocab,
                                                     slot_vocab,
                                                     intent_vocab)

                pred_intents = []
                correct_intents = []
                slot_outputs = []
                correct_slots = []
                input_words = []

                # used to gate
                gate_seq = []
                while True:
                    in_data, slot_data, slot_weight, length, \
                    intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                    if len(in_data) <= 0:
                        break
                    feed_dict = {input_data.name: in_data,
                                 sequence_length.name: length}
                    [infer_intent_out, infer_slot_out] = sess.run(inference_output_list, feed_dict)
                    # infer_intent_output:[batch, intent_size]
                    for input_seq in infer_intent_out:
                        # pred_intents:list(max_intent)
                        pred_intents.append(np.argmax(input_seq))
                    # intent label
                    for input_seq in intents:
                        correct_intents.append(input_seq)
                    # infer_slot_out:[batch, max_seq_length]
                    # pred_slots:[batch, max_seq_length, 1]
                    pred_slots = infer_slot_out.reshape((slot_data.shape[0], slot_data.shape[1], -1))
                    for pred_slot, target_slot, input_seq, length in zip(pred_slots, slot_data, in_data, length):
                        if arg.use_crf or use_batch_crossent:
                            # p:[input_sequence_length,1] => [input_sequence_length]
                            pred_slot = pred_slot.reshape([-1])
                        else:
                            pred_slot = np.argmax(pred_slot, 1)

                        tmp_pred = []
                        tmp_correct = []
                        tmp_input = []
                        for j in range(length):
                            tmp_pred.append(slot_vocab['rev'][pred_slot[j]]) # id->slot_word
                            tmp_correct.append(slot_vocab['rev'][target_slot[j]])
                            tmp_input.append(in_vocab['rev'][input_seq[j]])

                        slot_outputs.append(tmp_pred)
                        correct_slots.append(tmp_correct)
                        input_words.append(tmp_input)

                    if data_processor_valid.end == 1:
                        break
                # 对所有数据N进行计算
                pred_intents = np.array(pred_intents) # [N]
                correct_intents = np.array(correct_intents) # [N]
                accuracy = (pred_intents == correct_intents) # [N]
                semantic_acc = accuracy # intent_acc, [N]
                accuracy = accuracy.astype(float)
                accuracy = np.mean(accuracy) * 100.0

                index = 0
                # correct_slots:[N, input_seq_length]
                # slot_outputs:[N, input_seq_length]
                for target_slot, pred_slot in zip(correct_slots, slot_outputs):
                    # Process Semantic Error
                    if len(target_slot) != len(pred_slot):
                        raise ValueError('Error!!')
                    # target_slot:[input_seq_length]
                    for j in range(len(target_slot)):
                        # TODO:此处计算语义准确率是不是有些严格
                        if pred_slot[j] != target_slot[j]: # 如果有一个slot不对,则将整个句子设为语义错误
                            semantic_acc[index] = False
                            break
                    index += 1
                semantic_acc = semantic_acc.astype(float)
                semantic_acc = np.mean(semantic_acc) * 100.0

                f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
                logging.info('slot f1: ' + str(f1))
                logging.info('intent accuracy: ' + str(accuracy))
                logging.info('semantic Acc(intent, slots are all correct): ' + str(semantic_acc))

                data_processor_valid.close()
                return f1, accuracy, semantic_acc, pred_intents, correct_intents, \
                       slot_outputs, correct_slots, input_words, gate_seq


            logging.info(' ')
            logging.info('Valid:') # 即通常所谓的dev集
            epoch_valid_slot, epoch_valid_intent, epoch_semantic_acc_score, \
            valid_pred_intent, valid_correct_intent, valid_pred_slot, \
            valid_correct_slot, valid_words, valid_gate \
                = run_validate(
                os.path.join(full_valid_path, arg.input_file),
                os.path.join(full_valid_path, arg.slot_file),
                os.path.join(full_valid_path, arg.intent_file))

            logging.info(' ')
            logging.info('Test:')
            epoch_test_slot, epoch_test_intent, epoch_test_err, \
            test_pred_intent, test_correct_intent, test_pred_slot, \
            test_correct_slot, test_words, test_gate = run_validate(
                os.path.join(full_test_path, arg.input_file),
                os.path.join(full_test_path, arg.slot_file),
                os.path.join(full_test_path, arg.intent_file))

            if epoch_semantic_acc_score <= best_valid_score:
                no_improve += 1
            else:
                best_valid_score = epoch_semantic_acc_score
                logging.info('get new best score: Semantic Acc: {}'.format(epoch_semantic_acc_score))
                no_improve = 0

            if epochs == arg.max_epochs:
                break

            if arg.early_stop == True:
                if no_improve > arg.patience:
                    print("no improve for last {} epoch, early stop!".format(arg.patience))
                    break
            logging.info('='*20)
