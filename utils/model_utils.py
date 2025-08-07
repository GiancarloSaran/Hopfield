from numpy import random as npr
import numpy as np
from .model import async_evolve, noisify

def retrieval_ratio(patterns, W, threshold, noise=0, overlap=1):
  P, N = patterns.shape
  init = noisify(patterns, 1, noise)
  retrieval = 0
  for mu in range(P):
    final, _ = async_evolve(init[0, mu, :], W, threshold)
    retrieval += (act_overlap(final, patterns)[mu] >= overlap)
  return retrieval / P


def act_overlap(act, patterns):
    #Returns an array of shape (P,)
    P, N = patterns.shape
    overlaps = np.zeros(P)
    overlaps = np.mean(act*patterns, axis=1)
    return overlaps

def overlap_matrix(patterns):
    #Returns a matrix of shape (P, P)
    P, N = patterns.shape
    overlaps = patterns@patterns.T/N
    return overlaps

def rand_patterns(P, N):
    #Returns an array of shape (P, N)
    patterns = npr.randint(0, 2, size=(P, N))
    return 2*patterns - 1

def init_weights(N):
    #Gaussian random weights, could add variance as parameter
    W = npr.randn(N, N)
    for i in range(N):
      W[i, i] = 0
    return (W+W.T)/2