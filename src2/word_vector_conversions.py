import numpy as np


def nearest_neighbor_using_numpy(_vector, _vocabulary, _embeddings):
    """
    This function accepts a vector and then returns an array in embeddings(positions) that is
    most similar to the vector. Metric used is cosine similarity.

    :param _vector: word's vector
    :param _vocabulary: from different files
    :param _embeddings: from different files
    :return: vector of the word that is most similar to the input word's vector
    """

    # Multiply(Dot product) _vector to all vectors in embeddings
    _dot_product_of_embeddings_n_vector = np.multiply(_embeddings, _vector)

    # returns an array of elements, where each element is sum of all values in the corresponding vector
    # eg. a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; np.sum(a, axis=1) => [6, 15, 24]
    _sum_of_vector_values = np.sum(_dot_product_of_embeddings_n_vector, axis=1)

    # x -> Input

    # square of individual elements
    _x_square = np.square(_vector)
    # sums all the elements
    _x_sum = np.sum(_x_square, 0)
    # x-length -> square root of the sum of squares
    _x_length = np.sqrt(_x_sum)

    # y -> Output

    # square of individual elements
    _y_square = np.square(_embeddings)
    # sums all the elements
    _y_sum = np.sum(_y_square, 1)
    # y-length -> square root of the sum of squares
    _y_length = np.sqrt(_y_sum)

    # element-wise multiplication
    _product = np.multiply(_x_length, _y_length)

    # Find the cosine-similarities
    _cosine_similarities = np.divide(_sum_of_vector_values, _product)

    return _embeddings[np.argmax(_cosine_similarities)]


def word_to_vector(_word, _vocabulary, _embeddings):
    """
    Accepts a word and returns its vector form, from the embeddings(positions) list.
    If the word is nt in our vocabulary, then we return the vector form of unknown word
    given by "unk"

    :param _word: word
    :param _vocabulary: from different files
    :param _embeddings: from different files
    :return: vector form of the word
    """

    # If word exists in our vocabulary, then return its vector form
    if _word in _vocabulary:
        return _embeddings[_vocabulary.index(_word)]

    # If word doesn't exist in our vocabulary, then return vector form of unknown word
    return _embeddings[_vocabulary.index('unk')]


def vector_to_word(_vector, _vocabulary, _embeddings):
    """
    This function accepts a vector and then returns its corresponding word. But if the vector
    does not exist in the embeddings(position), it returns the word, whose vector is closest
    to the input vector. The closeness is found by the cosine similarity

    :param _vector: vector array
    :param _embeddings: from different files
    :param _vocabulary: from different files
    :return: corresponding word, else, closest word
    """

    # We iterate over all the vectors
    for _possible_match_vector in _embeddings:

        # if the possible match vector matches the input vector
        if np.array_equal(_possible_match_vector, np.asarray(_vector)):

            # We return its corresponding word
            return _vocabulary[list(_embeddings).index(_possible_match_vector)]

    # If no vector in the embeddings matches the input vector, then we return the word
    # whose vector is closest to the input vector according to cosine similarity.
    return vector_to_word(nearest_neighbor_using_numpy(np.asarray(_vector), _vocabulary, _embeddings)
                          , _vocabulary, _embeddings)
