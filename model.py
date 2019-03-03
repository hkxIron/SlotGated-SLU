import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

def createModel(input_data,
                input_size,
                sequence_length,
                slot_size,
                intent_size,
                remove_slot_attn,
                add_final_state_to_intent,
                use_crf,
                layer_size=128,
                isTraining=True,
                embedding_path=None,
                use_batch_crossent=True
                ):

    #cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    cell_fw = tf.nn.rnn_cell.LSTMCell(layer_size)
    cell_bw = tf.nn.rnn_cell.LSTMCell(layer_size)
    #cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)

    if isTraining == True:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                input_keep_prob=0.5,
                                                output_keep_prob=0.5)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                input_keep_prob=0.5,
                                                output_keep_prob=0.5)
    # embedding layer， [word size, embed size] 724, 64
    if embedding_path:
        embedding_weight = np.load(embedding_path)
        embedding = tf.Variable(embedding_weight, name='embedding', dtype=tf.float32)
    else:
        embedding = tf.get_variable('embedding', [input_size, layer_size])
    # embedding:[vocab_size, embedding_size]
    # input_data:[batch, input_sequence_length]
    # inputs:[batch, input_sequence_length, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, input_data)
    # state_outputs: [batch, nstep, embed size], final_state: [4, bs, embed size] include cell state * 2, hidden state * 2

    #(output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
    # output_fw: [batch, input_sequence_length, num_units],它的值为hidden_state
    # output_bw: [batch, input_sequence_length, num_units],它的值为hidden_state
    # (cell_state_fw, hidden_state_fw) = states_fw
    # cell_state_fw: [batch, num_units]
    # hidden_state_fw: [batch, num_units]
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                 cell_bw=cell_bw,
                                                                 inputs=inputs,
                                                                 sequence_length=sequence_length,
                                                                 dtype=tf.float32)
    (cell_state_fw,hidden_state_fw) = state_fw
    (cell_state_bw,hidden_state_bw) = state_bw
    # [batch, hidden_size * 4]
    final_state = tf.concat([cell_state_fw, hidden_state_fw, cell_state_bw, hidden_state_bw], axis=1)
    # sequence_outputs:[batch, input_sequence_length, hidden_size* 2]
    sequence_outputs = tf.concat([output_fw, output_bw], axis=2)
    print("cell_state_fw:", cell_state_fw, " hidden_state_fw:", hidden_state_fw)
    print("final_state:", final_state)
    print("squence_outputs:", sequence_outputs)

    # tensor.get_shape()返回的是tuple
    sequence_output_shape = sequence_outputs.get_shape() # [batch, input_sequence_length, hidden_size* 2]

    with tf.variable_scope('attention'):
        # state_outputs:[batch, input_sequence_length, hidden_size* 2]
        slot_inputs = sequence_outputs
        if not remove_slot_attn: # 需要slot attention
            with tf.variable_scope('slot_attn'):
                """
                e_{i,k}=V^T*tanh(W_{he}*h_k+W_{ie}*h_i)
                alpha_{i,j} = softmax(e{i,j})
                c_i = sum_{j}(alpha_{i,j}*h_j)
                
                y_i=softmax(W_hy*(h_i+c_i))
               
                其中: 
                W_{he}*h_k:用的是卷积实现
                W_{ie}*h_i:用的是线性映射 _linear()
                """

                """
                W_{he}*h_k:用的是卷积实现
                """
                # attn_size = hidden size * 2
                attn_size = sequence_output_shape[2].value
                # [batch, height=input_sequence_length, width=1, channel=hidden_size * 2]
                hidden_input_conv = tf.expand_dims(sequence_outputs, axis=2)
                # W_he: [filter_height=1, filter_width=1, in_channels=hidden*2, out_channels=hidden*2], 注意: 1*1的核
                W_he = tf.get_variable("slot_AttnW", shape=[1, 1, attn_size, attn_size])
                # hidden_features:[batch, height=input_sequence_length, width=1, channel=hidden_size * 2]
                # 物理意义:对hidden的各维之间进行卷积,等价于: W_{he}*h_k,不过不太清楚为何用卷积来实现
                hidden_features = tf.nn.conv2d(input=hidden_input_conv, filter=W_he, strides=[1, 1, 1, 1], padding="SAME")

                origin_shape = tf.shape(sequence_outputs) # 返回的是tensor
                # hidden_features:[batch,input_sequence_length,hidden_size * 2]
                hidden_features = tf.reshape(hidden_features, origin_shape)
                # hidden_features:[batch, 1,input_sequence_length, hidden_size * 2]
                hidden_features = tf.expand_dims(hidden_features, 1)

                """
                W_{ie}*h_i:用的是线性映射 _linear(), W_{ie}未显式声明,在Linear函数中
                """
                # [batch, input_sequence_length, hidden_size* 2]
                slot_inputs_shape = tf.shape(slot_inputs) #返回tensor
                # slot_inputs:[batch * input_sequence_length, hidden_size * 2]
                slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])
                # [batch * input_sequence_length, hidden_size * 2]
                # W_{ie}*h_i+bias, 注意:这里并没有显式定义W_ie,因为在_linear函数中会自己定义W_ie
                y = core_rnn_cell._linear(slot_inputs, output_size=attn_size, bias=True)
                #y = tf.layers.dense(slot_inputs, attn_size, use_bias=True, activation=None) # 线性函数也可以这样写
                # [batch , input_sequence_length, hidden_size* 2]
                y = tf.reshape(y, slot_inputs_shape)
                # [batch , input_sequence_length, 1, hidden_size* 2]
                y = tf.expand_dims(y, 2)

                """
                e_{i,k}=V^T*tanh(W_{he}*h_k+W_{ie}*h_i)
                注意:
                在seq2seq-attention中,e_{i,k}=g(s_{i-1},h_k), 
                即e_{i,k}是由encoder中的hidden与decoder中的hidden共同作用而来
                但此处的e_{i,k}比较特殊,h_k,h_i都由encoder的hidden隐向量得来
                因此, 这种做法有点类似于 transformer中的query-key-value-attention的query的计算方式
                """
                # [batch , nstep, nstep] = [batch, 1, nstep, hidden_size*2] + [batch , nstep, 1, hidden_size * 2]
                # hidden_features:[batch, 1,input_sequence_length, hidden_size * 2]
                # y:[batch, input_sequence_length,1 , hidden_size* 2]
                # bahdanauAdd:[batch, input_sequence_length, input_sequence_length, hidden_size* 2]
                # 有维度为1的,会自动广播
                bahdanau_activate = tf.tanh(hidden_features + y)
                # V:[attn_size=hidden_size*2]
                V = tf.get_variable("slot_AttnV", [attn_size])
                # logit_i_k:[batch, input_sequence_length, input_sequence_length, hidden_size* 2]
                #   =>[batch, input_sequence_length, input_sequence_length]
                logit_i_k = tf.reduce_sum(V * bahdanau_activate, axis=[3]) # 注意:这里是点乘,不是矩阵相乘
                """
                alpha_{i,j} = softmax(e{i,j})
                c_i = sum_{j}(alpha_{i,j}*h_j)
                """
                # score_i_k:[batch, input_sequence_length, input_sequence_length]
                score_i_k = tf.nn.softmax(logit_i_k, axis=-1)
                # score_i_k:[batch, input_sequence_length=i, input_sequence_length=k, 1]
                score_i_k = tf.expand_dims(score_i_k, axis=-1)
                # hidden=[batch, 1, input_sequence_length, hidden_size* 2]
                hidden = tf.expand_dims(sequence_outputs, axis=1)
                # score_i_k:[batch, input_sequence_length, input_sequence_length, 1]
                # hidden:   [batch, 1, input_sequence_length, hidden_size* 2]
                # slot_attentioned_hidden: [batch, input_sequence_length, hidden_size * 2]
                slot_attentioned_hidden = tf.reduce_sum(score_i_k * hidden, axis=[2])
        else: # 不需attention
            # attn_size = hidden size * 2
            attn_size = sequence_output_shape[2].value
            # [batch*input_sequence_length, hidden_size* 2]
            slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])

        # ===============intent attention ============================
        """
        注意:intent attention是针对最后的hidden state进行的
        """
        # intent_input:[batch, hidden_size * 4]
        intent_input = final_state
        with tf.variable_scope('intent_attn'):
            # attn_size: hidden_size*2
            attn_size = sequence_output_shape[2].value
            # hidden:[batch, input_sequence_length, 1, hidden_size*2]
            hidden = tf.expand_dims(sequence_outputs, 2)
            """
            注意:虽然名字相同, 但与slot-attn中的不是同一个变量!!!
            """

            """
            W_{he}*h_k:用的是卷积实现
            """
            # W_he: [filter_height=1, filter_width=1, in_channels=hidden*2, out_channels=hidden*2], 注意: 1*1的核
            W_he = tf.get_variable("intent_AttnW", shape=[1, 1, attn_size, attn_size]) # 注意:此处与 slot_attention中用的是相同的attention
            # 物理意义:对hidden的各维之间进行卷积,等价于: W_{he}*h_k
            # [batch, input_sequence_length, 1, hidden_size*2]
            hidden_features = tf.nn.conv2d(input=hidden, filter=W_he, strides=[1, 1, 1, 1], padding="SAME")
            V = tf.get_variable("intent_AttnV", shape=[attn_size])


            """
            W_{ie}*h_i:用的是线性映射 _linear() ,W_{ie}未显式声明,在Linear函数中
            """
            # intent_input:[batch, hidden_size*4]
            # y: [batch, attn_size=hidden_size*2]
            y = core_rnn_cell._linear(intent_input, output_size=attn_size, bias=True)
            print("intent-attn, attn_size:",attn_size, " y:", y)
            # [batch, 1, 1, hidden_size * 2]
            y = tf.reshape(y, shape=[-1, 1, 1, attn_size])

            """
            e_{i,k}=V^T*tanh(W_{he}*h_k+W_{ie}*h_i)
            """
            # V:[batch, input_sequence_length, 1, hidden_size*2]
            # hidden_features:[batch, input_sequence_length, 1, hidden_size*2]
            # y:[batch, 1, 1, hidden_size * 2]
            # bahdanau_activate:[batch, input_sequence_length, 1, hidden_size*2]
            bahdanau_activate = V * tf.tanh(hidden_features + y)
            # logit_i_k:[batch, input_sequence_length]
            logit_i_k = tf.reduce_sum(bahdanau_activate, axis=[2, 3])

            """
            alpha_{i,j} = softmax(e{i,j})
            c_i = sum_{j}(alpha_{i,j}*h_j)
            """
            # [batch, input_sequence_length]
            score_i_k = tf.nn.softmax(logit_i_k)
            # [batch, input_sequence_length, 1]
            score_i_k = tf.expand_dims(score_i_k, axis=-1)
            # score_i_k:[batch, input_sequence_length, 1, 1]
            score_i_k = tf.expand_dims(score_i_k, axis=-1)
            # hidden:[batch, input_sequence_length, 1, hidden_size*2]
            # intent_attentioned_hidden:[batch, hidden_size*2]
            intent_attentioned_hidden = tf.reduce_sum(score_i_k * hidden, axis=[1, 2])

            if add_final_state_to_intent == True:
                # intent_input:[batch, hidden_size * 4]
                # intent_attentioned_hidden:[batch, hidden_size*2]
                # intent_output:[batch, hidden_size* 2 + hidden_size * 4]
                intent_output = tf.concat([intent_attentioned_hidden, intent_input], 1)
            else:
                # intent_attentioned_hidden:[batch, hidden_size*2]
                intent_output = intent_attentioned_hidden

        with tf.variable_scope('slot_gated'):
            # intent_gate:[batch, hidden_size * 2]
            intent_gate = core_rnn_cell._linear(intent_output, output_size=attn_size, bias=True)
            embed_size = intent_gate.get_shape()[1].value
            # [batch, 1, hidden_size * 2]
            intent_gate = tf.reshape(intent_gate, [-1, 1, embed_size])
            # V_gate:[hidden_size*2]
            V_gate = tf.get_variable("gateV", [attn_size])
            if not remove_slot_attn: # 需要slot attention
                # slot_attentioned_hidden: [batch, input_sequence_length, hidden_size * 2]
                # intent_gate:[batch, 1, hidden_size * 2]
                # slot_gate:[batch, input_sequence_length, hidden_size * 2]
                slot_gate = V_gate * tf.tanh(slot_attentioned_hidden + intent_gate)
            else:
                # sequence_outputs:[batch, input_sequence_length, hidden_size * 2]
                # intent_gate:[batch, 1, hidden_size * 2]
                # slot_gate:[batch, input_sequence_length, hidden_size * 2]
                slot_gate = V_gate * tf.tanh(sequence_outputs + intent_gate)
            # slot_gate:[batch, input_sequence_length]
            slot_gate = tf.reduce_sum(slot_gate, axis=[2])
            # slot_gate:[batch, input_sequence_length, 1]
            slot_gate = tf.expand_dims(slot_gate, -1)
            if not remove_slot_attn: # 需要slot attention
                # slot_attentioned_hidden: [batch, input_sequence_length, hidden_size * 2]
                # slot_gate:[batch, input_sequence_length, 1]
                # slot_gate:[batch, input_sequence_length, hidden_size* 2]
                slot_gate = slot_attentioned_hidden * slot_gate
            else:
                # sequence_outputs:[batch, input_sequence_length, hidden_size* 2]
                # slot_gate:[batch, input_sequence_length, hidden_size* 2]
                slot_gate = sequence_outputs * slot_gate
            # slot_gate:[batch * input_sequence_length, attn_size=hidden_size*2]
            slot_gate = tf.reshape(slot_gate, [-1, attn_size])
            # slot_gate:[batch * input_sequence_length, attn_size=hidden_size*2]
            # slot_inputs:[batch * input_sequence_length, hidden_size * 2]
            # slot_output:[batch * input_sequence_length, hidden_size * 4]
            slot_output = tf.concat([slot_gate, slot_inputs], 1)

    with tf.variable_scope('intent_proj'):
        # intent_output:[batch, hidden_size* 2 + hidden_size * 4]
        # intent_logits:[batch, intent_size]
        intent_logits = core_rnn_cell._linear(intent_output, output_size=intent_size, bias=True)

    with tf.variable_scope('slot_proj'):
        # slot_output:[batch * input_sequence_length, hidden_size * 4]
        # slot_logits:[batch * input_sequence_length, slot_size]
        slot_logits = core_rnn_cell._linear(slot_output, output_size=slot_size, bias=True)
        if use_crf or use_batch_crossent:
            # sequence_outputs:[batch, input_sequence_length, hidden_size* 2]
            nstep = tf.shape(sequence_outputs)[1]
            # slot_logits:[batch, input_sequence_length, slot_size]
            slot_logits = tf.reshape(slot_logits, [-1, nstep, slot_size])

    return [slot_logits, intent_logits]

