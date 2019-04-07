# function for getting the data from url
from TextRank.article_from_url import get_article_from_link
from TextRank.text_processing import separate_sentences_from_text,\
    separate_words_from_text
from TextRank.ranking_functions import getter_keyword_frequency, score_on_all_factors


def summarize(_title, _text):
    _summary = []
    _list_of_sentences = separate_sentences_from_text(_text)
    _keywords = getter_keyword_frequency(_text)
    _title_split_into_list_of_words = separate_words_from_text(_title)

    if len(_list_of_sentences) <= 5:
        return _list_of_sentences

    # Scoring sentences, and using the top 5
    # most_common returns a list of tuples, where each tuple = (key, value), of most common keys
    _final_ranks = score_on_all_factors(_list_of_sentences,
                                        _title_split_into_list_of_words, _keywords).most_common(5)

    # Iterate over all key value pairs of the top 5 ranks
    for _sentence_and_its_rank in _final_ranks:

        # append the key, which is the first element of the tuple, to the summary.
        # Here, key is the sentence.
        _summary.append(_sentence_and_its_rank[0])

    # Return the list of top 5 sentences as the summary
    return _summary


def summarize_data_from_url(_local_url):
    """
    If the user inputs-> URL instead of article, then we need this function
    to collect the data from the URL and then pass the collected article and title
    for the summarize() function to generate the summary and return it.
    If the user inputs-> article_title and article_data, then we don't need this
    functions and we can directly use the summarize() function

    :param _local_url: URL provided by the user
    :return summary: The final output
    """

    # get article from the URL
    _article = get_article_from_link(_local_url)

    # Proceed only is if we get the article data
    if _article:

        # get the article text content
        _cleaned_article = _article.text

        # get the title of the article
        _cleaned_title = _article.title

        # for debugging
        # print(_cleaned_article)
        # print(_cleaned_title)

        # proceed only if article and title is available
        if _cleaned_article and _cleaned_title:

            # find the summary
            _generated_summary = summarize(_cleaned_title, _cleaned_article)

            # return summary
            return _generated_summary

    # If receiving article fails, then return None to signal exit
    print("Cannot proceed further as no article is available to process on.."
          "Exiting !")
    return None
