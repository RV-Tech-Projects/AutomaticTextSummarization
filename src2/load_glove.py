def load_global_vectors(glove_file):
    """
    This function loads the GloVe file and returns a tuple whose 1st element is
    a list of all the vocabulary words and the 2nd element is the list of all the
    vector positions in the N-D plane.

    :param glove_file: Name of the file
    :return: Tuple of vocab and their positions
    """

    # Initialize empty vocabulary list
    _vocabulary = []
    # Initialize empty positions list
    _positions = []

    # Open the file
    with open(glove_file, 'r') as file:

        # iterate over each of its lines
        for _vector in file.readlines():

            # Split the line to get the vocabulary word and its position
            # separately
            _parsed_vector = _vector.strip().split(' ')

            # Push the vocabulary word in the vocabulary's list
            _vocabulary.append(_parsed_vector[0])

            # Push the position of the word in the position's list
            _positions.append(_parsed_vector[1:])

        # Log the message to the console: FOR DEBUGGING
        print('GloVe Loading Complete!')

    # Return the tuple
    return _vocabulary, _positions
