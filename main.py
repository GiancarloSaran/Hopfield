import numpy as np
import numpy.random as npr
import argparse
#import pytorch as torch
#from numba import jit, njit
import matplotlib.pyplot as plt
from tqdm import tqdm
npr.seed(1982)

def rand_patterns(P, N):
    #Returns an array of shape (P, N)
    return 2*npr.randint(2, size=(P, N)) - 1

def init_weights(N):
    #Gaussian random weights, could add variance as parameter
    W = npr.randn(N, N)
    W[np.diag_indices(N)] = 0
    return (W+W.T)/2

def true_weights(patterns):
    #Returns a matrix of shape (N, N)
    P, N = patterns.shape
    W = patterns.T@patterns/N
    W[np.diag_indices(N)] = 0
    return W

def act_overlap(act, patterns):
    #Returns an array of shape (P,)
    P, N = patterns.shape
    overlaps = np.mean(act*patterns, axis=1)
    return overlaps

def overlap_matrix(patterns):
    #Returns a matrix of shape (P, P)
    P, N = patterns.shape
    overlaps = patterns@patterns.T/N
    return overlaps

def noisify(patterns, n_copies, noise_fraction):
    #Returns an array of shape (n_copies, P, N)
    P, N = patterns.shape
    noisy_patterns = np.repeat(np.expand_dims(patterns, axis=0), n_copies, axis=0)
    n_flips = int(noise_fraction * N)
    for i in range(n_copies):
        for j in range(P):
            idx = npr.choice(N, n_flips, replace=False)
            noisy_patterns[i, j, idx] *= -1
    return noisy_patterns

def convergence(patterns, conv_copies, val_frac, W, thresh, n_steps):
    #When all the patterns are stable, return True
    P, N = patterns.shape
    train_batch = noisify(patterns, conv_copies, val_frac)
    is_converging = True
    #Could add a second convergence criterion based on the dW curve.
    for a in range(conv_copies):
        for mu in range(P):
            final = async_evolve(train_batch[a, mu, :], n_steps, W, thresh)
            if not np.allclose(final, patterns[mu]):
                is_converging = False
                break
    return is_converging

#Consider splitting wake train and sleep train
def train(W, patterns, n_copies, conv_copies, noise_frac, val_frac, n_epochs,
          stim, thresh, wake_lr, wake_steps, eval_steps, alpha=1):
    P, N = patterns.shape
    noisy_patterns = noisify(patterns, n_copies, noise_frac) #Noisy patterns, functioning as training set
    curr_act = np.ones(N) #Initialize activations
    curr_batch = np.ones((wake_steps, N))
    tW = true_weights(patterns)
    norm_true = np.zeros(n_epochs)
    norm_prev = np.zeros(n_epochs)
    epoch = 0
    pbar = tqdm(total=n_epochs, initial=epoch, position=0, leave=True)
    stop = False
    #Wake phase
    while not stop and epoch < n_epochs:
        pbar.set_description(f'Epoch {epoch}')
        a, mu = npr.randint(n_copies), npr.randint(P)
        stimulus = noisy_patterns[a, mu, :]
        #Stimulation
        for t in range(wake_steps):
            order = npr.permutation(N)
            for i in order:
                field = W@curr_act + stim*stimulus
                curr_act[i] = np.sign(field[i] + thresh[i])
            curr_batch[t] = curr_act
        #Hebbian weight update with weight decay
        S_mat = curr_batch[:, :, np.newaxis]@curr_batch[:, np.newaxis, :]
        dW = wake_lr/N**2*np.sum(S_mat, axis=0) - alpha*wake_lr*W/N
        W += dW
        W[np.diag_indices(N)] = 0
        norm_true[epoch] = np.linalg.norm(W - tW) #Track distance from true weights
        norm_prev[epoch] = np.linalg.norm(dW) #Track distance from previous weights
        epoch += 1
        pbar.update(1)
        if convergence(patterns, conv_copies, val_frac, W, thresh, eval_steps):
            stop = True
    pbar.close()
    return W, norm_true[:epoch], norm_prev[:epoch]

def sleep(W, sleep_steps, sleep_lr):
    #Dream phase
    N = W.shape[0]
    norm_w = np.zeros(sleep_steps) #Sometimes it diverges
    print(f'Non mi svegliate, ve ne prego!')
    for k in range(sleep_steps):
        W += sleep_lr/(1 + sleep_lr*k)*(W - W@W)
        W[np.diag_indices(N)] = 0
        norm_w[k] = np.linalg.norm(W)
        if k > 0 and norm_w[k] > norm_w[k-1]:
            print(f'Stopping early at step {k} due to increase in norm')
            break
    print(f'All done! I dreamt for {k} steps') #Can't be witty early in the morning
    return W, norm_w[:k]

def async_evolve(input, steps, W, thresh):
    #Testing/Inference function
    N = input.size
    act = input.copy()
    for t in range(steps):
        order = npr.permutation(N)
        for i in order:
            field = W@act
            act[i] = np.sign(field[i] + thresh[i])
    return act

#Utility/Observables functions to be written later
#Energy, overlaps, correlations, hamming distance, crosstalk, etc.

def main():
    #Decide how to pass parameters later (argparse, config file, etc.)
    n_copies = 30
    conv_copies = 15
    noise_frac = 0.05
    val_frac = 0.1
    n_epochs = 2500
    stim = 10
    wake_lr = 0.5
    wake_steps = 15
    eval_steps = 150
    alpha = 1

    P, N = 2, 15
    patterns = rand_patterns(P, N)
    threshold = 0 * np.ones(N)
    W = init_weights(N)
    W, norm_true, norm_prev = train(W, patterns, n_copies, conv_copies, noise_frac, val_frac, n_epochs, stim, \
               threshold, wake_lr, wake_steps, eval_steps, alpha)
    sleep_lr = 1e-06
    sleep_steps = 40000
    W, norm_w = sleep(W, sleep_steps, sleep_lr)
    P2 = 2
    conv_copies = 10
    new_patterns = rand_patterns(P2, N)
    tot_patterns = np.concatenate((patterns, new_patterns), axis=0)
    W2, norm_true, norm_prev = train(W.copy(), tot_patterns, n_copies, conv_copies, noise_frac, val_frac, n_epochs, stim, \
                threshold, wake_lr, wake_steps, eval_steps, alpha) #Maybe present tot_patterns
    sleep_lr = 1e-05
    sleep_steps = 75000
    W2, norm_w = sleep(W2, sleep_steps, sleep_lr)
    plt.plot(norm_w)
    print(overlap_matrix(tot_patterns))

if __name__ == "__main__":
    main()