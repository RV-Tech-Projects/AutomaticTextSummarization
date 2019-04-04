from src2.load_glove import load_global_vectors
import numpy as np
import csv
from nltk import word_tokenize
import pickle

# Global variables for cross-file access
vocabulary = None
positions = None  # positions and embeddings are same
embeddings = None
dimension_of_word_vector = None

summaries = list()
texts = list()


def removal_of_non_printable_characters(_text) -> str:
    """
    This function removes all the non-printable characters from the text

    :param _text: text variable
    :return: cleaned text
    """

    # string.printable has all the character that are printable
    from string import printable as _printable

    # We only choose those characters that are printable
    return str(filter(lambda _character: _character in _printable, _text))


if __name__ == '__main__':
    # GloVe file name
    glove_file_name = '../glove.6B/glove.6B.50d.txt'

    # load the glove file
    vocabulary, positions = load_global_vectors(glove_file_name)

    # convert positions to np array and change their data-type to float32
    embeddings = np.asarray(positions)
    embeddings = embeddings.astype(np.float32)

    # The dimensions of all vectors will be same, so we just use the 1st vector
    # to find the dimensions
    dimension_of_word_vector = len(embeddings[0])

    # variable to hold the location of the dataset
    _location = '../datasets/amazon_fine_food_reviews/Reviews.csv'
    with open(_location, 'r') as _file:

        # Read data as a dictionary. Column name is key, and its row-values is value.
        reviews = csv.DictReader(_file)

        # We iterate over all the reviews
        for _review in reviews:

            # clean the text
            _cleaned_text = removal_of_non_printable_characters(_review['Text'])

            # clean the summary of the text
            _cleaned_summary = removal_of_non_printable_characters(_review['Summary'])

            # separate the words from the text
            _words_in_text = word_tokenize(_cleaned_text)

            # separate the words from the summary
            _words_in_summary = word_tokenize(_cleaned_summary)

            # push the text in the texts list
            texts.append(_words_in_text)

            # push the summary in the summaries list
            summaries.append(_words_in_summary)

    print("Loaded all reviews on texts and summaries list!")

    # # Reducing the data to execute the next steps faster
    # DATA_LIMIT = 50000
    #
    # # Some stats
    # print("Total data = " + str(len(texts)))
    # print("Limited to =", DATA_LIMIT)
    #
    # # Remove extra texts
    # texts = texts[:DATA_LIMIT]
    #
    # # Remove extra summaries
    # summaries = summaries[:DATA_LIMIT]

    # These variables will hold the vocabulary words from the reviews and their
    # positions
    _reviews_vocabulary = list()
    _reviews_embeddings = list()

    # Import word vector conversions file
    import src2.word_vector_conversions as converter

    # we iterate over all the texts
    for _text in texts:

        # While doing so,  we iterate over each word in the text
        for _word in _text:

            # If the word doesn't exist in our reviews vocabulary
            if _word not in _reviews_vocabulary:

                # but the word exists in our model's vocabulary(GLOVE)
                if _word in vocabulary:

                    # then we append the word to the reviews vocabulary
                    _reviews_vocabulary.append(_word)

                    # and its vector value is appended to the reviews embeddings
                    _reviews_embeddings.append(converter.word_to_vector(_word, vocabulary, embeddings))

    print("Added all texts non-stop words to vocabulary")

    # we iterate over all the summaries
    for _summary in summaries:

        # While doing so, we iterate over each word int the summary
        for _word in _summary:

            # If the word doesn't exist in our reviews vocabulary
            if _word not in _reviews_vocabulary:

                # but the word exists in our model's vocabulary(GLOVE)
                if _word in vocabulary:

                    # then we append the word to the reviews vocabulary
                    _reviews_vocabulary.append(_word)

                    # and its vector value is appended to the reviews embeddings
                    _reviews_embeddings.append(converter.word_to_vector(_word, vocabulary, embeddings))

    print("Added all summaries non-stop words to vocabulary")

    # We need 2 tokens: eos(end of sentence) and unk(unknown), in our
    # vocabulary to make some decisions or set some boundaries

    # If eos is not in our vocabulary
    if 'eos' not in _reviews_vocabulary:

        # We append it to the vocabulary list
        _reviews_vocabulary.append('eos')

        # and also append its vector value to the embeddings list
        _reviews_embeddings.append(converter.word_to_vector('eos', vocabulary, embeddings))

    # If eos is not in our vocabulary
    if 'unk' not in _reviews_vocabulary:

        # We append it to the vocabulary list
        _reviews_vocabulary.append('unk')

        # and also append its vector value to the embeddings list
        _reviews_embeddings.append(converter.word_to_vector('unk', vocabulary, embeddings))

    # We generally process in sequences of a fixed length. If a particular
    # sequence doesn't have the sufficient length, we add some padding(trash
    # characters) to the sequence to make its length equal to the fixed length.
    # The pad character is represented by <PAD> in the word embeddings convention.
    # Also, its vector value is 0 so that it doesn't have any effect on the
    # processing.

    # Initialize the pad vector to have all 0's
    _pad_vector = np.zeros(dimension_of_word_vector)

    # append the pad character to our vocabulary
    _reviews_vocabulary.append('<PAD>')

    # append the pad character's vector to the embeddings(vector list)
    _reviews_embeddings.append(_pad_vector)

    # converting all the words of the summaries in the vector form
    summaries_in_vector_form = []

    # for every summary in summaries
    for _summary in summaries:

        # we construct a different list for every summary
        # this list will be a collection of vectors of the
        # words in this summary
        _summary_vector = []

        # for every word in the summary
        for word in _summary:

            # we append its vector form to the summary_vector
            _summary_vector.append(converter.word_to_vector(word, vocabulary, embeddings))

        # we add the vector form of the eos character as well
        _summary_vector.append(converter.word_to_vector('eos', vocabulary, embeddings))

        # then we convert is to np-array and change its data-type to float-32
        _summary_vector = np.asarray(_summary_vector).astype(np.float32)

        # then we finally append the group of vectors for 1 summary in the
        # all_summaries: summaries_in_vector_form, list. It will be a list
        # of list of vector forms of words for each summary.
        summaries_in_vector_form.append(_summary_vector)

    print("Converted all summaries in vector form!")

    # converting all the words of the text in the vector form
    texts_in_vector_form = []

    # for every word in the texts
    for _text_ in texts:

        # we append its vector form to the text_vector
        _text_vector = []

        # for every word in the text
        for word in _text_:

            # we append its vector form to the text_vector
            _text_vector.append(converter.word_to_vector(word, vocabulary, embeddings))

        # then we convert is to np-array and change its data-type to float-32
        _text_vector = np.asarray(_text_vector).astype(np.float32)

        # then we finally append the group of vectors for 1 text in the
        # all_texts: texts_in_vector_form, list. It will be a list
        # of list of vector forms of words for each text.
        texts_in_vector_form.append(_text_vector)

    print("Converted all texts in vector form!")

    # Saving the processed data in some file to re-use it later

    with open('../processed_data/amazon_reviews/_reviews_vocabulary', 'wb') as fp:
        pickle.dump(_reviews_vocabulary, fp)
    with open('../processed_data/amazon_reviews/_reviews_embeddings', 'wb') as fp:
        pickle.dump(_reviews_embeddings, fp)
    with open('../processed_data/amazon_reviews/_summaries_in_vector_form', 'wb') as fp:
        pickle.dump(summaries_in_vector_form, fp)
    with open('../processed_data/amazon_reviews/_texts_in_vector_form', 'wb') as fp:
        pickle.dump(texts_in_vector_form, fp)

    print("Final stage complete!")
