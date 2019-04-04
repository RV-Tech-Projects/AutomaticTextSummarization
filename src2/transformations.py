from src2.word_vector_conversions import vector_to_word
import numpy as np


def transformation_of_output_summary(_text_vector, _reviews_vocabulary, _reviews_embeddings):
    """
    This function takes in a text-vector and returns the np-array of indices.
    These indices are the index of the word in _reviews_vocabulary.

    :param _text_vector: vector form of all the words in the text
    :param _reviews_vocabulary: for searching the word
    :param _reviews_embeddings: used because of a compulsory argument for a function
    :return: np-array of indices
    """

    # get the length to avoid calculating in the future
    _length = len(_text_vector)

    # initialize the transformed_output with np array having values = 0
    _transformed_output = np.zeros(_length, dtype=np.int32)

    # for every word-vector of the input-text-vector, we assign its index (from the
    # long list of vocabulary) as the transformed_output of the input text_vector
    for _i in range(_length):
        _transformed_output[_i] = \
            _reviews_vocabulary.\
            index(vector_to_word(_text_vector[_i], _reviews_vocabulary, _reviews_embeddings))

    # return all the indices of the words from input-text in reviews_vocabulary
    return _transformed_output
