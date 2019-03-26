def get_article_from_link(local_url):
    """
    The function accepts a single parameter, URL, then we use the library newspaper
    to extract the article from that URL and return that article if extraction is
    successful, else, return None

    :param local_url: URL of the site containing the article
    :return article: If fetching of article is successful,
            None: If fetching or article fails
    """
    # import the newspaper module to use one of its functions to extract the data
    import newspaper

    try:
        # create the newspaper object with the given URL as its member
        article = newspaper.Article(local_url)

        # download the URL web-page
        article.download()

        # parse the article data from the page
        article.parse()

        # return article
        return article

    # if the extraction of article from the URL fails, it raises a Value
    # error which is accepted by the following block
    except newspaper.article.ArticleException:

        # Log the error message then return None
        print('Failed to extract article from the given URL')
        return None
