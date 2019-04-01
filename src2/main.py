from src2.load_glove import load_global_vectors
import numpy as np


# Global variables for cross-file access
vocabulary = None
positions = None
embeddings = None
dimension_of_word_vector = None


if __name__ == '__main__':
    # Run Pre-processing file first .

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

