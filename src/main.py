from src.summarize import summarize_data_from_url, summarize


def log_summary_for_url(fetch_url):
    """
    TO log the summary of an article to be extracted from an URL.
    JUST FOR DEBUGGING

    :param fetch_url:
    :return: Nothing or None
    """

    # fetch the summary
    summary = summarize_data_from_url(fetch_url)

    # if summary is None, then it means article couldn't be fetched. Hence,
    # log the error and Halt.
    if summary is None:
        print("Program halted: no article found!")
        exit(0)

    # else, Log the summary to the console
    else:
        for _s in summary:
            print(_s)
        return None


def log_summary(input_title, input_text):
    """
    TO log the summary of an article.
    JUST FOR DEBUGGING

    :param input_title: title of the article
    :param input_text: contents of the article
    :return: Nothing or None
    """

    # fetch the summary
    summary = summarize(input_title, input_text)

    # if summary is None, then it means article couldn't be fetched. Hence,
    # log the error and Halt.
    if summary is None:
        print("Program halted: no article found!")
        exit(0)

    # else, Log the summary to the console
    else:
        for _s in summary:
            print(_s)
        return None


if __name__ == '__main__':

    url = "https://www.bbc.com/news/world-asia-india-47721497"
    log_summary_for_url(url)
