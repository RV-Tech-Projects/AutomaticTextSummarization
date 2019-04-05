import tensorflow as tf

range_width = 10
window_size = 2 * range_width + 1


def score(_hs, _ht, _Wa, _sequence_length):
    return tf.reshape(tf.matmul(tf.matmul(_hs, _Wa),
                                tf.transpose(_ht)), [_sequence_length])


def align(_hs, _ht, _Wp, _Vp, _Wa, _tf_sequence_length):

    _pd = tf.TensorArray(size=(2 * range_width + 1), dtype=tf.float32)

    _positions = tf.cast(_tf_sequence_length - 1 - 2 * range_width, dtype=tf.float32)

    _sigmoid_multiplier = tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(_ht, _Wp)), _Vp))
    _sigmoid_multiplier = tf.reshape(_sigmoid_multiplier, [])

    _pt_float = _positions * _sigmoid_multiplier

    _pt = tf.cast(_pt_float, tf.int32)
    _pt = _pt + range_width

    _sigma = tf.constant(range_width / 2, dtype=tf.float32)

    _i = 0
    _pos = _pt - range_width

    def body(__i, __pos, __pd):
        __comp_1 = tf.cast(tf.square(__pos - _pt), tf.float32)
        __comp_2 = tf.cast(2 * tf.square(_sigma), tf.float32)

        __pd = __pd.write(__i, tf.exp(-(__comp_1 / __comp_2)))

        return __i + 1, __pos + 1, __pd

    _i, _pos, _pd = tf.while_loop(lambda __i: __i < window_size, body, [_i, _pos, _pd])

    __local_hs = _hs[(_pt - range_width):(_pt + range_width + 1)]

    __normalized_scores = tf.nn.softmax(score(__local_hs, _ht, _Wa, 2 * range_width + 1))

    _pd = _pd.stack()

    __G = tf.multiply(__normalized_scores, _pd)
    __G = tf.reshape(__G, [2 * range_width + 1, 1])

    return __G, _pt
