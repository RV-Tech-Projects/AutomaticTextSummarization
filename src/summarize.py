# function for getting the data from url
from src.article_from_url import get_article_from_link


# TODO: Yet to complete
def summarize(_title, _text):
    return "something, for now, to avoid warnings"


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
