# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np 
import math
from collections import Counter, defaultdict

def handle_unknown_words(t, documents): 
    """
    Replaces tokens in the given documents with <unk> (unknown) tokens if they occur 
    less frequently than a certain threshold, based on the provided parameter 't'. 
    Tokens are ordered first by frequency then alphabetically, so the tokens 
    replaced are the least frequent tokens and earliest alphabetically.
    
    Input:
        t (float):
            A value between 0 and 1 representing the threshold for token frequency.
            The int(t * total_unique_tokens) least frequent tokens will be replaced.
        documents (list of lists):
            A list of documents, where each document is represented as a list of tokens.
    Output:
        new_documents (list of lists):
            A list of processed documents where the int(t * total_unique_tokens) least
            frequent tokens have been replaced with <unk> tokens and no other changes. 
        vocab (list):
            A list of tokens representing the vocabulary, including both the most common tokens
            and the <unk> token.
    Example:
    t = 0.3
    documents = [["apple", "banana", "apple", "orange"],
                 ["apple", "cherry", "banana", "banana"],
                 ["cherry", "apple", "banana"]]
    new_documents, vocab = handle_unknown_words(t, documents)
    # new_documents:
    # [['apple', 'banana', 'apple', '<unk>'],
    #  ['apple', 'cherry', 'banana', 'banana'],
    #  ['cherry', 'apple', 'banana']]
    # vocab: ['banana', 'apple', 'cherry', '<unk>']
    """
    new_documents = documents.copy()
    
    # Dict of token to list of all (i,j) at which the given token appears within documents
    token_position = {}

    # Create position dict    
    for i in range(len(documents)):
        sen = documents[i]

        for j in range(len(sen)):
            token = sen[j]

            # Add this token's coordinates to the token position dictionary
            if token in token_position.keys():
                token_position[token].append((i,j))

            else:
                token_position[token] = [(i,j)]

    # The vocabulary is simply the keys of the token position dict
    vocab = [t for t in token_position.keys()]

    # Calculate the number of <unk> tokens
    n_unk = int(t * len(vocab))

    # The list of (token, count of token)
    counts = [(token, len(token_position[token])) for token in vocab]

    # Sort the counts list first by count then by alphabetical order of token
    counts.sort(key=lambda pair: (pair[1], pair[0]))

    # Replace words in new_documents with <unk> until we have replaced n_unk words
    while n_unk > 0:

        (token_to_replace, token_count) = counts.pop(0)

        vocab.remove(token_to_replace)

        # The list of (i,j) coordinates within documents at which token_to_replace occurs
        positions = token_position[token_to_replace]

        # Go through all the positions this token occurs at and change each corresponding position in new_documents to <unk>
        for (i,j) in positions:
            new_documents[i][j] = "<unk>"

        # Decrement num unkown words
        n_unk -= 1

    # Add <unk> to vocab
    vocab.append('<unk>')

    return new_documents, vocab

def apply_smoothing(k, observation_counts, unique_obs):
    """
    Apply add-k smoothing to state-observation counts and return the log smoothed observation 
    probabilities log[P(observation | state)].

    Input:
        k (float): 
            A float number to add to each count (the k in add-k smoothing)
            Observation here can be either an NER tag or a word, 
            depending on if you are applying_smoothing to transition_matrix or emission_matrix
        observation_counts (Dict[Tuple[str, str], float]): 
            A dictionary containing observation counts for each state.
            Keys are state-observation pairs and values are numbers of occurrences of the key.
            Keys should contain  all possible combinations of (state, observation) pairs. 
            i.e. if a `(NER tag, word)` doesn't appear in the training data, you should still include it as `observation_counts[(NER tag, word)]=0`
        unique_obs (List[str]):
            A list of string containing all the unique observation in the dataset. 
            If you are applying smoothing to the transition matrix, unique_obs contains all the possible NER tags in the dataset.
            If you are applying smoothing to the emission matrix, unique_obs contains the vocabulary in the dataset

    Output:
        Dict<key Tuple[String, String]: value Float>
            A dictionary containing log smoothed observation **probabilities** for each state.
            Keys are state-observation pairs and values are the log smoothed 
            probability of occurrences of the key.
            The output should be the same size as observation_counts.

    Note that the function will be applied to both transition_matrix and emission_matrix. 
    """

    # Dict of state --> count of all occurences of the state (over all observations)
    state_counts = {}

    # Sum up all occurences of the state (over all observations)
    for ((state, _), count) in observation_counts.items():

        if state in state_counts.keys():
            state_counts[state] += count
        else:
            state_counts[state] = count
    
    log_smoothed_p = {}
    
    for ((state, obs), count) in observation_counts.items():
        log_smoothed_p[(state, obs)] = np.log((count + k) / (state_counts[state] + (k*len(unique_obs))))

    return log_smoothed_p


