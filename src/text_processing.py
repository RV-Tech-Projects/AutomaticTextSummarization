def separate_sentences_from_text(_text_to_split):
    """
    This function takes in a text as a string and then, separates
    sentences from it.

    :param _text_to_split:
    :return: list of sentences
    """

    # Natural Language Toolkit has many NLP functions
    import nltk

    # nltk.tokenize.sent_tokenize() function splits the text into a
    # list of sentences.
    _separated_sentences_list = nltk.tokenize.sent_tokenize(_text_to_split)

    # return the list of separated sentences
    return _separated_sentences_list


def split_words(_text_to_split):
    """
    This function takes in a text as a string and then, separates
    words from it.

    :param _text_to_split:
    :return: list of words
    """

    # Natural Language Toolkit has many NLP functions
    import nltk

    # nltk.tokenize.word_tokenize() function splits the text into a
    # list of words
    _possible_separated_words = nltk.tokenize.word_tokenize(_text_to_split)

    # The list of words also contains some special characters as
    # different words. We don't need those.
    _separated_words = []

    # For every word in the list of words produced, we check if it is a
    # special character
    for _possible_word in _possible_separated_words:

        # We select the word only if it is not a special character
        if not (_possible_word in ".,!?\"<>;:/\\]}[{+=-_~`@#$%^&*()"):
            _separated_words.append(_possible_word)

    # We return the final list of pure words
    return _separated_words


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
    print(separate_sentences_from_text(text))
    print(split_words(text))
