from src2.encoders_and_decoders import forward_encoder, backward_encoder, decoder
from src2.score_and_align import align
import tensorflow as tf
import numpy as np

# Some Globals
_size_of_hidden_layer = 500
K = 5
range_width = 10


def model(_tf_text, _tf_seq_len, _tf_output_len, _word_vec_dim, _vocab_len,
          _np_embedding_limit, _SOS_position):
    # PARAMETERS
    # 1.1 FORWARD ENCODER PARAMETERS

    initial_hidden_f = tf.zeros([1, _size_of_hidden_layer], dtype=tf.float32)
    cell_f = tf.zeros([1, _size_of_hidden_layer], dtype=tf.float32)
    wf_f = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    uf_f = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bf_f = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    wi_f = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    ui_f = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bi_f = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    wo_f = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    uo_f = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bo_f = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    wc_f = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    uc_f = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bc_f = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    _w_attention_f = tf.Variable(tf.zeros([K, 1]), dtype=tf.float32)

    # 1.2 BACKWARD ENCODER PARAMETERS

    initial_hidden_b = tf.zeros([1, _size_of_hidden_layer], dtype=tf.float32)
    cell_b = tf.zeros([1, _size_of_hidden_layer], dtype=tf.float32)
    wf_b = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    uf_b = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bf_b = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    wi_b = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    ui_b = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bi_b = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    wo_b = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    uo_b = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bo_b = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    wc_b = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, _size_of_hidden_layer], stddev=0.01))
    uc_b = tf.Variable(np.eye(_size_of_hidden_layer), dtype=tf.float32)
    bc_b = tf.Variable(tf.zeros([1, _size_of_hidden_layer]), dtype=tf.float32)
    _w_attention_b = tf.Variable(tf.zeros([K, 1]), dtype=tf.float32)

    # 2 ATTENTION PARAMETERS

    Wp = tf.Variable(tf.truncated_normal(shape=[2 * _size_of_hidden_layer, 50], stddev=0.01))
    Vp = tf.Variable(tf.truncated_normal(shape=[50, 1], stddev=0.01))
    Wa = tf.Variable(tf.truncated_normal(shape=[2 * _size_of_hidden_layer, 2 * _size_of_hidden_layer], stddev=0.01))
    Wc = tf.Variable(tf.truncated_normal(shape=[4 * _size_of_hidden_layer, 2 * _size_of_hidden_layer], stddev=0.01))

    # 3 DECODER PARAMETERS

    Ws = tf.Variable(tf.truncated_normal(shape=[2 * _size_of_hidden_layer, _vocab_len], stddev=0.01))

    cell_d = tf.zeros([1, 2 * _size_of_hidden_layer], dtype=tf.float32)
    wf_d = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, 2 * _size_of_hidden_layer], stddev=0.01))
    uf_d = tf.Variable(np.eye(2 * _size_of_hidden_layer), dtype=tf.float32)
    bf_d = tf.Variable(tf.zeros([1, 2 * _size_of_hidden_layer]), dtype=tf.float32)
    wi_d = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, 2 * _size_of_hidden_layer], stddev=0.01))
    ui_d = tf.Variable(np.eye(2 * _size_of_hidden_layer), dtype=tf.float32)
    wo_d = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, 2 * _size_of_hidden_layer], stddev=0.01))
    uo_d = tf.Variable(np.eye(2 * _size_of_hidden_layer), dtype=tf.float32)
    wc_d = tf.Variable(tf.truncated_normal(shape=[_word_vec_dim, 2 * _size_of_hidden_layer], stddev=0.01))
    uc_d = tf.Variable(np.eye(2 * _size_of_hidden_layer), dtype=tf.float32)
    bc_d = tf.Variable(tf.zeros([1, 2 * _size_of_hidden_layer]), dtype=tf.float32)

    hidden_residuals_d = tf.TensorArray(size=K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)
    hidden_residuals_d = hidden_residuals_d.unstack(tf.zeros([K, 2 * _size_of_hidden_layer], dtype=tf.float32))

    _w_attention_d = tf.Variable(tf.zeros([K, 1]), dtype=tf.float32)

    output = tf.TensorArray(size=_tf_output_len, dtype=tf.float32)

    # BI-DIRECTIONAL LSTM

    hidden_forward = forward_encoder(_tf_text,
                                     initial_hidden_f, cell_f,
                                     wf_f, uf_f, bf_f,
                                     wi_f, ui_f, bi_f,
                                     wo_f, uo_f, bo_f,
                                     wc_f, uc_f, bc_f,
                                     _w_attention_f,
                                     _tf_seq_len,
                                     _word_vec_dim, K, _size_of_hidden_layer)

    hidden_backward = backward_encoder(_tf_text,
                                       initial_hidden_b, cell_b,
                                       wf_b, uf_b, bf_b,
                                       wi_b, ui_b, bi_b,
                                       wo_b, uo_b, bo_b,
                                       wc_b, uc_b, bc_b,
                                       _w_attention_b,
                                       _tf_seq_len,
                                       _word_vec_dim, K, _size_of_hidden_layer)

    encoded_hidden = tf.concat([hidden_forward, hidden_backward], 1)

    # ATTENTION MECHANISM AND DECODER

    decoded_hidden = encoded_hidden[0]
    decoded_hidden = tf.reshape(decoded_hidden, [1, 2 * _size_of_hidden_layer])
    _w_attention_d_normalized = tf.nn.softmax(_w_attention_d)
    _tf_embedding_limit = tf.convert_to_tensor(_np_embedding_limit)

    y = tf.convert_to_tensor(_SOS_position)  # initial decoder token <SOS> vector
    y = tf.reshape(y, [1, _word_vec_dim])

    j = K

    hidden_residuals_stack = hidden_residuals_d.stack()

    RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j - K:j], _w_attention_d_normalized), 0)
    RRA = tf.reshape(RRA, [1, 2 * _size_of_hidden_layer])

    decoded_hidden_next, cell_d = decoder(y, decoded_hidden, cell_d,
                                          wf_d, uf_d, bf_d,
                                          wi_d, ui_d, bf_d,
                                          wo_d, uo_d, bf_d,
                                          wc_d, uc_d, bc_d,
                                          RRA)
    decoded_hidden = decoded_hidden_next

    hidden_residuals_d = hidden_residuals_d.write(j, tf.reshape(decoded_hidden, [2 * _size_of_hidden_layer]))

    j = j + 1

    i = 0

    def attention_decoder_cond(__i):
        return __i < _tf_output_len

    def attention_decoder_body(__i, __j, __decoded_hidden, __cell_d,
                               __hidden_residuals_d, __output):

        # LOCAL ATTENTION
        G, pt = align(encoded_hidden, __decoded_hidden, Wp, Vp, Wa, _tf_seq_len)
        local_encoded_hidden = encoded_hidden[pt - range_width:pt + range_width + 1]
        weighted_encoded_hidden = tf.multiply(local_encoded_hidden, G)
        context_vector = tf.reduce_sum(weighted_encoded_hidden, 0)
        context_vector = tf.reshape(context_vector, [1, 2 * _size_of_hidden_layer])

        attended_hidden = tf.tanh(tf.matmul(tf.concat([context_vector, __decoded_hidden], 1), Wc))

        # DECODER

        __y = tf.matmul(attended_hidden, Ws)

        __output = __output.write(__i, tf.reshape(__y, [_vocab_len]))
        # Save probability distribution as output

        __y = tf.nn.softmax(__y)

        __y_index = tf.cast(tf.argmax(tf.reshape(__y, [_vocab_len])), tf.int32)
        __y = _tf_embedding_limit[__y_index]
        __y = tf.reshape(__y, [1, _word_vec_dim])

        # setting next decoder input token as the word_vector of maximum probability
        # as found from previous attention-decoder output.

        __hidden_residuals_stack = __hidden_residuals_d.stack()

        __RRA = tf.reduce_sum(tf.multiply(__hidden_residuals_stack[__j - K:__j], _w_attention_d_normalized), 0)
        __RRA = tf.reshape(__RRA, [1, 2 * _size_of_hidden_layer])

        __decoded_hidden_next, __cell_d = decoder(__y, __decoded_hidden, __cell_d,
                                                  wf_d, uf_d, bf_d, wi_d, ui_d, bf_d,
                                                  wo_d, uo_d, bf_d, wc_d, uc_d, bc_d,
                                                  __RRA)

        __decoded_hidden = __decoded_hidden_next

        __hidden_residuals_d = \
            tf.cond(tf.equal(__j, _tf_output_len - 1 + K + 1),
                    lambda: __hidden_residuals_d,
                    lambda: __hidden_residuals_d.write(__j,
                                                       tf.reshape(__decoded_hidden,
                                                                  [2 * _size_of_hidden_layer])))

        return __i + 1, __j + 1, __decoded_hidden, __cell_d, __hidden_residuals_d, __output

    i, j, decoded_hidden, cell_d, hidden_residuals_d, output = tf.while_loop(attention_decoder_cond,
                                                                             attention_decoder_body,
                                                                             [i, j, decoded_hidden, cell_d,
                                                                              hidden_residuals_d, output])
    hidden_residuals_d.close().mark_used()

    output = output.stack()

    return output
