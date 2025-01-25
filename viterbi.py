# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np

def viterbi(model, observation, tags):
  """
  Returns the model's predicted tag sequence for a particular observation.
  Use `get_trellis_arc` method to obtain model scores at each iteration.

  Input: 
    model: HMM/MEMM model
    observation: List[String]
    tags: List[String]
  Output:
    predictions: List[String]
  """
  v = np.zeros((len(tags), len(observation)))
  vb = np.zeros((len(tags), len(observation)))

  # Initialize V & Vb
  for i in range(len(tags)):
    tag = tags[i]

    v[i][0] = model.get_trellis_arc(tag, None, observation, 0)
    vb[i][0] = -1



  # Fill in Viterbi Matrices

  # For every observation
  for i in range(1, len(observation)):
    obs = observation[i]

    # For every possible tag we could assign to this observation
    for j in range(len(tags)):
      predicted_tag = tags[j]

      # Keep track of max log smoothed p and the tag index that yields the max p
      max_p = float('-inf')
      max_idx = -1

      # For every possible previous tag whose path we couldve taken to get to this (predicted tag, observation) pair
      for k in range(len(tags)):
        prev_tag = tags[k]

        log_smoothed_p = v[k][i-1] + model.get_trellis_arc(predicted_tag, prev_tag, observation, i)

        if log_smoothed_p > max_p:
          max_p = log_smoothed_p
          max_idx = k

      # Update the Viterbi matrix with the value of max_p at the jth row and ith col
      v[j][i] = max_p

      # Update the Viterbi Backpointer matrix with the value of max_idx at the same position
      vb[j][i] = max_idx



  # Backtracking

  # Keep track of the max log smoothed p and the idx of the tag that yields that maximum
  max_final_p = float('-inf')
  max_final_idx = -1

  # Loop over all final entries in v and calculate/add the probability that that tag-->qf
  for i in range(len(tags)):
    tag = tags[i]

    v[i][-1] = v[i][-1] + model.get_trellis_arc('qf', tag, observation, None)

    if v[i][-1] > max_final_p:
      max_final_p = v[i][-1]
      max_final_idx = i

  obs_idx = len(observation) - 1
  pred_tag_idx = max_final_idx

  predictions = []

  while obs_idx >= 0:
    
    # Insert the predicted tag at the start of the list
    predictions.insert(0, tags[pred_tag_idx])

    # Check vb for the next predicted tag for next iteration
    pred_tag_idx = int(vb[pred_tag_idx][obs_idx])
    
    # Decrement vb for next iteration
    obs_idx -= 1

  return predictions