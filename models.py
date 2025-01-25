# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
from collections import defaultdict
from nltk import classify
from nltk import download
from nltk import pos_tag
import numpy as np

class HMM: 

  def __init__(self, documents, labels, vocab, all_tags, k_t, k_e, k_s, smoothing_func): 
    """
    Initializes HMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
      vocab: List[String], dataset vocabulary
      all_tags: List[String], all possible NER tags 
      k_t: Float, add-k parameter to smooth transition probabilities
      k_e: Float, add-k parameter to smooth emission probabilities
      k_s: Float, add-k parameter to smooth starting state probabilities
      smoothing_func: (Float, Dict<key Tuple[String, String] : value Float>, List[String]) -> 
      Dict<key Tuple[String, String] : value Float>
    """
    self.documents = documents
    self.labels = labels
    self.vocab = vocab
    self.all_tags = all_tags
    self.k_t = k_t
    self.k_e = k_e
    self.k_s = k_s
    self.smoothing_func = smoothing_func
    self.emission_matrix = self.build_emission_matrix()
    self.transition_matrix = self.build_transition_matrix()
    self.start_state_probs = self.get_start_state_probs()

  def build_transition_matrix(self):
    """
    Returns the transition probabilities as a dictionary mapping all possible
    (tag_{i-1}, tag_i) tuple pairs to their corresponding smoothed 
    log probabilities: log[P(tag_i | tag_{i-1})]. 
    
    Note: Consider all possible tags. This consists of everything in 'all_tags', but also 'qf' our end token.
    Use the `smoothing_func` and `k_t` fields to perform smoothing.

    Output: 
      transition_matrix: Dict<key Tuple[String, String] : value Float>
    """

    # Dict of (state1, state2) --> count of times this particular transition occured 
    observation_counts = {}

    # Initialize observation counts with all possible (state1, state2) pairs INCLUDING (state, 'qf') set to a count of 0
    for t1 in self.all_tags:
      for t2 in self.all_tags:
        observation_counts[(t1, t2)] = 0

      observation_counts[(t1, 'qf')] = 0

    # Loop over the sentences in the data and count the occurences of each tag1-->tag2 transition
    for sen in self.labels:
      
      # The tag being transitioned FROM
      prev_tag = sen[0]

      for i in range(1, len(sen)):

        # The tag being transitioned TO
        cur_tag = sen[i]

        transition = (prev_tag, cur_tag)

        # Increment the count of this transition
        observation_counts[transition] += 1

        # Set prev_tag to cur_tag for the next iteration of the loop
        prev_tag = cur_tag

      # At this point prev_tag holds the LAST tag in the sentence tf
      # Increment the count of tf --> qf
      final_transition = (prev_tag, 'qf')
      observation_counts[final_transition] += 1
      
    tag_with_qf = self.all_tags.copy()
    tag_with_qf.append('qf')

    return self.smoothing_func(self.k_t, observation_counts, tag_with_qf)

  def build_emission_matrix(self): 
    """
    Returns the emission probabilities as a dictionary, mapping all possible 
    (tag, token) tuple pairs to their corresponding smoothed log probabilities: 
    log[P(token | tag)]. 
    
    Note: Consider all possible tokens from the list `vocab` and all tags from 
    the list `all_tags`. Use the `smoothing_func` and `k_e` fields to perform smoothing.
  
    Output:
      emission_matrix: Dict<key Tuple[String, String] : value Float>
      Its size should be len(vocab) * len(all_tags).
    """
    # Dict of (state, token) --> count of times this particular emission occurrenced
    observation_counts = {}

    # Initialize observation counts with all possible (state, token) pairs set to a count of 0
    for tag in self.all_tags:
      for token in self.vocab:
        observation_counts[(tag, token)] = 0

    # Loop over the sentences in the data and count the occurrences of each tag-->token emission
    for i in range(len(self.documents)):
      for j in range(len(self.documents[i])):
        tag = self.labels[i][j]
        token = self.documents[i][j]

        observation_counts[(tag, token)] += 1

    return self.smoothing_func(self.k_e, observation_counts, self.vocab)

  def get_start_state_probs(self):
    """
    Returns the starting state probabilities as a dictionary, mapping all possible 
    tags to their corresponding smoothed log probabilities. Use `k_s` smoothing
    parameter to manually perform smoothing.
    
    Note: Do NOT use the `smoothing_func` function within this method since 
    `smoothing_func` is designed to smooth state-observation counts. Manually
    implement smoothing here.

    Output: 
      start_state_probs: Dict<key String : value Float>
    """
    # Dict of tag-->count of number of times this tag started a sentence in the documents
    start_counts = {}

    # Since there is 1 starting label per sentence, the total number of starting labels is qual to the number of sentences
    tot_start = len(self.documents)

    # Increment tot_start by 1 addition of k_e for each tag in all_tags
    tot_start += (self.k_s * len(self.all_tags))

    # Initialize start counts with all states initially set to a count of 0
    for tag in self.all_tags:
      start_counts[tag] = 0

    # Loop over the sentences in labels and increment the count of the first label
    for sen in self.labels:
      start_counts[sen[0]] += 1

    # Perform Smoothing on the counts
    log_smoothed_p = {}

    for (tag, count) in start_counts.items():
      log_smoothed_p[tag] = np.log((count+self.k_s)/tot_start)

    return log_smoothed_p

  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the `previous_tag` and `document` at index `i`.
    
    For HMM, this would be the sum of the smoothed log emission probabilities and 
    log transition probabilities: 
    log[P(predicted_tag | previous_tag))] + log[P(document[i] | predicted_tag)].
    
    Note: Treat unseen tokens as an <unk> token.
    Note: Make sure to handle the case where we are dealing with the first word. Is there a transition probability for this case?
    Note: Make sure to handle the case where the predicted tag is an end token. Is there an emission probability for this case?
  
    Input: 
      predicted_tag: String, predicted tag for token at index `i` in `document`
      previous_tag: String, previous tag for token at index `i` - 1
      document: List[String]
      i: Int, index of the `document` to compute probabilities 
    Output: 
      result: Float
    """

    # Replace unseen tokens with <unk>
    cur_token = None
    if predicted_tag != 'qf':
      cur_token = document[i]
    
    # Replace token with <unk> if it is not in vocab
    if not(cur_token in self.vocab):
      cur_token = "<unk>"

    # Log smoothed prob of the previous tag transitioning to the predicted tag
    #   Pt is the start state log smoothed probability if i == 0, 
    #   Otherwise Pt is the transition matrix log smoothed probability
    p_t = self.start_state_probs[predicted_tag] if i == 0 else self.transition_matrix[(previous_tag, predicted_tag)]

    # Log smoothed prob of the predicted tag emitting the token 
    #   Pe is 0 if the predicted tag is the end token 
    #   Otherwise Pe is the emission matrix log smoothed probability
    p_e = 0 if predicted_tag == 'qf' else self.emission_matrix[(predicted_tag, cur_token)]

    return p_t + p_e

    

 

################################################################################
################################################################################



class MEMM: 

  def __init__(self, documents, labels): 
    """
    Initializes MEMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
    """
    self.documents = documents
    self.labels = labels
    self.classifier = self.generate_classifier()


  def extract_features_token(self, document, i, previous_tag):
    """
    Returns a feature dictionary for the token at document[i].

    Input: 
      document: List[String], representing the document at hand
      i: Int, representing the index of the token of interest
      previous_tag: string, previous tag for token at index `i` - 1

    Output: 
      features_dict: Dict<key String: value Any>, Dictionaries of features 
                    (e.g: {'Is_CAP':'True', . . .})
    """

    features_dict = {}

    # Prev Token
    features_dict["PREV_TOKEN"] = "START_TOKEN"
    if i == None:
      features_dict["PREV_TOKEN"] = document[-1]
    elif i != 0:
      features_dict["PREV_TOKEN"] = document[i-1]
    
    # Prev Tag
    features_dict["PREV_TAG"] = previous_tag

    # Identity
    if i != None:
      features_dict["IDENTITY"] = document[i]
    else:
      features_dict["IDENTITY"] = "END_TOKEN"
      
    # First letter of this token is capitalized AND the first letter of the NEXT token is capitalized
    features_dict["IS_CAP"] = False
    if i != None:
      features_dict["IS_CAP"] = document[i][0].isupper()

    return features_dict

  def generate_classifier(self):
    """
    Returns a trained MaxEnt classifier for the MEMM model on the featurized tokens.
    Use `extract_features_token` to extract features per token.

    Output: 
      classifier: nltk.classify.maxent.MaxentClassifier 
    """

    feature_dicts = []

    for i in range(len(self.documents)):
      sen = self.documents[i]

      for j in range(len(sen)):

        prev_tag = 'q0'
        if j != 0:
          prev_tag = self.labels[i][j]

        feature_dicts.append((self.extract_features_token(sen, j, prev_tag), self.labels[i][j]))

    classifier = classify.MaxentClassifier.train(feature_dicts)
    
    return classifier
  
  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the features of the token of `document` at 
    index `i`.
    
    For MEMM, this would be the log classifier output log[P(predicted_tag | features_i)].
  
    Input: 
      predicted_tag: string, predicted tag for token at index `i` in `document`
      previous_tag: string, previous tag for token at index `i` - 1
      document: string
      i: index of the `document` to compute probabilities 
    Output: 
      result: Float
    """ 

    test_feature_dict = self.extract_features_token(document, i, previous_tag)

    return self.classifier.prob_classify(test_feature_dict).logprob(predicted_tag)

