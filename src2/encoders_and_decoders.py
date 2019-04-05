import tensorflow as tf


def forward_encoder(_array_of_vectors_of_texts, _hidden_layer, _cell, _wf, _uf, _bf,
                    _wi, _ui, _bi, _wo, _uo, _bo, _wc, _uc, _bc, _w_attention,
                    _sequence_length, _input_dimensions, _K, hidden_size):
    _w_attention = tf.nn.softmax(_w_attention, 0)
    hidden_forward = tf.TensorArray(size=_sequence_length, dtype=tf.float32)

    hidden_residuals = tf.TensorArray(
        size=_K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)

    hidden_residuals = hidden_residuals.unstack(
        tf.zeros([_K, hidden_size], dtype=tf.float32))

    _i = 0
    _j = _K

    def _condition(_i, _sequence_length):
        return _i < _sequence_length

    def body(__i, __j, _hidden, __cell, __hidden_forward, __hidden_residuals):
        x = tf.reshape(_array_of_vectors_of_texts[__i], [1, _input_dimensions])

        hidden_residuals_stack = __hidden_residuals.stack()

        __RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[__j - _K:__j], _w_attention), 0)
        __RRA = tf.reshape(__RRA, [1, hidden_size])

        # LSTM with RRA
        _fg = tf.sigmoid(tf.matmul(x, _wf) + tf.matmul(_hidden, _uf) + _bf)
        _ig = tf.sigmoid(tf.matmul(x, _wi) + tf.matmul(_hidden, _ui) + _bi)
        _og = tf.sigmoid(tf.matmul(x, _wo) + tf.matmul(_hidden, _uo) + _bo)
        __cell = tf.multiply(_fg, __cell) + tf.multiply(_ig, tf.tanh(tf.matmul(x, _wc) + tf.matmul(_hidden, _uc) + _bc))
        _hidden = tf.multiply(_og, tf.tanh(__cell + __RRA))

        __hidden_residuals = tf.cond(tf.equal(__j, _sequence_length - 1 + _K),
                                     lambda: __hidden_residuals,
                                     lambda: __hidden_residuals.write(__j, tf.reshape(_hidden, [hidden_size])))

        __hidden_forward = __hidden_forward.write(__i, tf.reshape(_hidden, [hidden_size]))

        return __i + 1, __j + 1, _hidden, __cell, __hidden_forward, __hidden_residuals

    _, _, _, _, hidden_forward, hidden_residuals = tf.while_loop(
        _condition, body, [_i, _j, _hidden_layer, _cell, hidden_forward, hidden_residuals])

    hidden_residuals.close().mark_used()

    return hidden_forward.stack()


def backward_encoder(_array_of_vectors_of_texts, _hidden_layer, _cell, _wf, _uf, _bf,
                     _wi, _ui, _bi, _wo, _uo, _bo, _wc, _uc, _bc,
                     _w_attention, _sequence_length, _input_dimensions, _K, _hidden_size):
    _w_attention = tf.nn.softmax(_w_attention, 0)
    _hidden_backward = tf.TensorArray(size=_sequence_length, dtype=tf.float32)

    _hidden_residuals = tf.TensorArray(
        size=_K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)

    _hidden_residuals = _hidden_residuals.unstack(
        tf.zeros([_K, _hidden_size], dtype=tf.float32))

    _i = _sequence_length - 1
    _j = _K

    def __condition(__i):
        return __i > -1

    def body(__i, __j, __hidden_layer, __cells, __hidden_backward, __hidden_residuals):
        x = tf.reshape(_array_of_vectors_of_texts[__i], [1, _input_dimensions])

        hidden_residuals_stack = __hidden_residuals.stack()

        RRA = tf.reduce_sum(tf.multiply(
            hidden_residuals_stack[__j - _K:__j], _w_attention), 0)
        RRA = tf.reshape(RRA, [1, _hidden_size])

        # LSTM with RRA
        fg = tf.sigmoid(tf.matmul(x, _wf) + tf.matmul(__hidden_layer, _uf) + _bf)
        ig = tf.sigmoid(tf.matmul(x, _wi) + tf.matmul(__hidden_layer, _ui) + _bi)
        og = tf.sigmoid(tf.matmul(x, _wo) + tf.matmul(__hidden_layer, _uo) + _bo)
        __cells = tf.multiply(fg, __cells) + tf.multiply(ig, tf.tanh(tf.matmul(x, _wc) +
                                                                     tf.matmul(__hidden_layer, _uc) + _bc))
        __hidden_layer = tf.multiply(og, tf.tanh(__cells + RRA))

        __hidden_residuals = tf.cond(
            tf.equal(__j, _sequence_length - 1 + _K),
            lambda: __hidden_residuals,
            lambda: __hidden_residuals.write(
                __j, tf.reshape(__hidden_layer, [_hidden_size])))

        __hidden_backward = __hidden_backward.write(
            __i, tf.reshape(__hidden_layer, [_hidden_size]))

        return __i - 1, __j + 1, __hidden_layer, __cells, __hidden_backward, __hidden_residuals

    _, _, _, _, _hidden_backward, _hidden_residuals = \
        tf.while_loop(__condition, body, [_i, _j, _hidden_layer, _cell,
                                          _hidden_backward, _hidden_residuals])

    _hidden_residuals.close().mark_used()

    return _hidden_backward.stack()


def decoder(_x, _hidden_layers, _cell, _wf, _uf, _bf, _wi, _ui, _bi,
            _wo, _uo, _bo, _wc, _uc, _bc, _RRA):

    # LSTM with RRA
    _fg = tf.sigmoid(tf.matmul(_x, _wf) + tf.matmul(_hidden_layers, _uf) + _bf)
    _ig = tf.sigmoid(tf.matmul(_x, _wi) + tf.matmul(_hidden_layers, _ui) + _bi)
    _og = tf.sigmoid(tf.matmul(_x, _wo) + tf.matmul(_hidden_layers, _uo) + _bo)
    _cell_next = tf.multiply(_fg, _cell) + tf.multiply(_ig, tf.tanh(tf.matmul(_x, _wc) + tf.matmul(_hidden_layers, _uc)
                                                                    + _bc))
    _hidden_next = tf.multiply(_og, tf.tanh(_cell + _RRA))

    return _hidden_next, _cell_next
