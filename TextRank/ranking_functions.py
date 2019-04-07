from TextRank.text_processing import separate_words_from_text
from collections import Counter


# GLOBALS
_stop_words = set()
_ideal_number_of_words_in_a_sentence = 20


def getter_keyword_frequency(_text: str) -> dict:
    """
    This functions chooses the top 10 keywords from the words in _text by
    their frequency. If number of words, in the text is less than 10, then
    all the words are returned.

    :param _text: Article in the form of text
    :return keywords: Top 10 words and their frequencies
    """

    # Split the text into a list of words
    _words = separate_words_from_text(_text_to_split=_text)

    # number of words in text before removing the stop words
    _num_of_words_including_stop_words = len(_words)

    # removing the stop words from our list of words
    # stop words = useless words

    # nltk has a builtin list of stop words
    from nltk.corpus import stopwords as nltk_stopwords

    # We only import the english stop words
    global _stop_words
    # setA.update(setB) => add contents of B to A
    _stop_words.update(set(nltk_stopwords.words('english')))

    # We than add our list of stop words to the builtin stop words
    # our list of stop words are stored in the text file that we import below
    with open("../assets/stopwords.txt", 'r') as stop_words_file:
        _word_on_line = stop_words_file.readline()
        while _word_on_line:
            _stop_words.add(_word_on_line)
            _word_on_line = stop_words_file.readline()

    # removing stop words from the list of words
    _words = [_word for _word in _words if _word not in _stop_words]

    # We than calculate the frequency of words
    _frequency_of_words = Counter(_words)

    # We choose top 10 words. If number of words is less than 10,
    # then we choose all the words
    _number_of_words_chosen = min(10, len(_frequency_of_words))

    # top 10 or n, if n <= 10, words
    _top_n_words = _frequency_of_words.most_common(_number_of_words_chosen)

    # choosing top n words where n = number of words chosen
    _keywords = {_word: _frequency for _word, _frequency in _top_n_words}

    # Scoring the top words chosen, by normalizing using some offsets (1, 1.5)
    for _word in _keywords:
        _keywords[_word] = 1 + \
                          _keywords[_word] * 1.5 / _num_of_words_including_stop_words

    # Finally, return top 10 words with a score value
    return _keywords


def summation_based_selection_score(_words: list, _keywords: dict) -> float:
    """
    This function returns the score by first adding all the normalized keyword frequency of all
    keywords and then returns the normalized final score

    :param _words: List of words from text
    :param _keywords: Dictionary of keywords with their frequency as value
    :return score: Normalized form of the sum of all the scores of each keyword
    """

    # initialize score with 0
    _score = 0

    # return 0 if no words are left after removal of stop words
    if len(_words) == 0:
        return 0

    # iterate over all words in the list of words, one by one
    for _word in _words:

        # If the word is a keyword, then we add it's normalized keyword frequency to the score
        if _word in _keywords:
            _score += _keywords[_word]

    # Normalization of the final score and then return it
    _score = _score / (10 * len(_words))
    return _score


def density_based_selection_score(_words: list, _keywords: dict) -> float:
    """
    This function finds the density of the keyword scores

    :param _words: List of all words
    :param _keywords: (DICT) Has the top 10 words chosen based on their
                            frequency and value = Normalized score
    :return: Normalized density score
    """

    # If no words exits, we return a score of 0
    if len(_words) == 0:
        return 0

    # initialize score with 0
    _sum = 0

    # _previous will be the list having position and frequency of the previous keyword
    # encountered in the list of words
    _previous = None

    # _current will be the list having position and frequency of the current keyword
    # encountered in the list of words
    _current = None

    # We iterate over each word int he list of words and also keep track of its position
    # in the list of words using the enumerate function
    for _i, _word in enumerate(_words):

        # if the word is a keyword
        if _word in _keywords:

            # Store its normalized frequency score in a variable
            _score = _keywords[_word]

            # If it is the first keyword encountered in the list of words, we do nothing and
            # store its position and score to be used as _previous for the next keyword that
            # will be encountered
            if _current is None:
                _current = [_i, _score]

            # else, we do the following procedure
            else:

                # We store the previous keyword's position and score
                _previous = _current.copy()

                # Then _current has current keyword's position and score
                _current = [_i, _score]

                # We then find the difference of the position of the current keyword and
                # the previously encountered keyword
                _difference_in_position = _current[0] - _previous[0]

                # Add the product of the scores of the curr and prev keyword divided by
                # the square of the difference in their positions
                _sum += (_current[1] * _previous[1]) / pow(_difference_in_position, 2)

    # number of keywords
    _number_of_keywords = len(_keywords.keys())

    # return the density of the score by the following normalization
    return _sum / ((_number_of_keywords + 1) * (_number_of_keywords + 1))


def scoring_the_title(_title_as_a_list_of_words: list, _sentence_as_a_list_of_words: list) -> float:
    """
    This function assigns score to a title based on the appearance of every word,
    in title, in the sentence passed as argument

    :param _title_as_a_list_of_words: Title as a list of words
    :param _sentence_as_a_list_of_words: Sentence as a list of words
    :return: normalized title score for a sentence
    """

    # removing the stop words from the title
    _title_as_a_list_of_words = [_word for _word in _title_as_a_list_of_words if _word not in _stop_words]

    # if title, after removing the stop words, is empty, then score is 0
    if len(_title_as_a_list_of_words) == 0:
        return 0

    # initialize the score to be 0
    _score = 0

    # iterating over each word in the sentence
    for _word in _sentence_as_a_list_of_words:

        # if word is not a stop word and if the word is in the title, then we increment the score
        if _word not in _stop_words and _word in _title_as_a_list_of_words:
            _score += 1

    # return the normalized value of the score
    return _score / len(_title_as_a_list_of_words)


def probability_of_a_sentence_being_important_based_on_position(_position_in_text: int
                                                                , _number_of_sentences: int) -> float:
    """
    Position of a sentence in the text affects its importance

    :param _position_in_text: index
    :param _number_of_sentences: Total number of sentences in the text
    :return: probability of being important
    """

    # Finding the normalized position of the sentence in the text
    _normalized_position = _position_in_text / _number_of_sentences

    # the range in which the normalized position of the sentence lies,
    # decides the probability
    if 0 < _normalized_position <= 0.1:
        return 0.17
    elif 0.1 < _normalized_position <= 0.2:
        return 0.23
    elif 0.2 < _normalized_position <= 0.3:
        return 0.14
    elif 0.3 < _normalized_position <= 0.4:
        return 0.08
    elif 0.4 < _normalized_position <= 0.5:
        return 0.05
    elif 0.5 < _normalized_position <= 0.6:
        return 0.04
    elif 0.6 < _normalized_position <= 0.7:
        return 0.06
    elif 0.7 < _normalized_position <= 0.8:
        return 0.04
    elif 0.8 < _normalized_position <= 0.9:
        return 0.04
    elif 0.9 < _normalized_position <= 1.0:
        return 0.15

    # if the normalized position is greater than 1, then its probability of
    # being important is 0. Also, after normalizing the position, the position
    # should lie in 0 <= pos <= 1. If for some reason it doesn't, then the following
    # code is reached and executed.
    return 0


def score_of_sentence_for_its_word_length(_sentence_as_a_list_of_words: list) -> float:
    """
    This function calculates the score of sentence's word's length

    :param _sentence_as_a_list_of_words: List of words forming a sentence
    :return: score based on number of words in the sentence
    """

    # deviation from the ideal number of words in a sentence
    _off_from_ideal = abs(_ideal_number_of_words_in_a_sentence - len(_sentence_as_a_list_of_words))

    # return the normalized score
    return 1 - (_off_from_ideal / _ideal_number_of_words_in_a_sentence)


def score_on_all_factors(_list_of_sentences: list, _title_as_a_list_of_words: list, _keywords: dict) -> Counter:
    """
    This function scores the sentences based of different factors like title, length, position in the text,
    summation score, density score, etc.

    :param _list_of_sentences: List of all the sentences
    :param _title_as_a_list_of_words: List of all the words of the title
    :param _keywords: Dictionary of top 10 keywords along with their normalized frequency score
    :return: Final score that is calculated based on many factors
    """

    # this variable stores the total number of sentences
    _number_of_sentences = len(_list_of_sentences)

    # We create a Counter object to be used a a dictionary.
    # We use Counter instead of dictionary because, Counter objects have a method attached to them
    # named: most_common. This method can help us get the top N sentences without having to write
    # a piece of code to do it for us.
    _ranks = Counter()

    # We then iterate over all the sentences to score them individually.
    for _pos, _sentence in enumerate(_list_of_sentences):

        # First convert sentence to a list of words.
        _sentence_split_into_list_of_words = separate_words_from_text(_sentence)

        # Scoring the sentence based on the title.
        _title_score_for_a_sentence = scoring_the_title(_title_as_a_list_of_words,
                                                        _sentence_split_into_list_of_words)

        # Scoring the sentence based on the number of words it has.
        _sentence_score_based_on_number_of_words = \
            score_of_sentence_for_its_word_length(_sentence_split_into_list_of_words)

        # Scoring the sentence based on where it is located in the text.
        _sentence_score_based_on_its_position = \
            probability_of_a_sentence_being_important_based_on_position(_pos + 1, _number_of_sentences)

        # Adding all the normalized frequency of keywords in the sentence and returning normalized score.
        _summation_based_selection_score = \
            summation_based_selection_score(_sentence_split_into_list_of_words, _keywords)

        # Scoring based on the density of the keywords.
        _density_based_selection_score = \
            density_based_selection_score(_sentence_split_into_list_of_words, _keywords)

        # Net score of SBS and DBS features.
        _net_selection_score = 10 * (_summation_based_selection_score + _density_based_selection_score) / 2

        # weighted average of scores from four categories.
        _final_score = (_title_score_for_a_sentence * 1.5 + _net_selection_score * 2.0 +
                        _sentence_score_based_on_number_of_words * 1.0 +
                        _sentence_score_based_on_its_position * 1.0) / 4.0

        # Storing the score of the sentence in the Counter object.
        _ranks[_sentence] = _final_score

    # Returning the scores of all the objects in a Counter object.
    return _ranks


# [ FOR DEBUGGING ] Only run this code if this file is run independently
if __name__ == '__main__':

    # Sample Text
    text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, ' \
           'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.' \
           ' Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris ' \
           'nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in ' \
           'reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. ' \
           'Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt ' \
           'mollit anim id est laborum.'

    # Test what the function returns
    print(getter_keyword_frequency(text))
