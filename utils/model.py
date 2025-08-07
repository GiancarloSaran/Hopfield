from numba import njit
import numpy as np
import numpy.random as npr

@njit
def true_weights(patterns):
    #Returns a matrix of shape (N, N)
    P, N = patterns.shape
    W = np.zeros((N, N))
    for mu in range(P):
      W += patterns[mu, :][:, np.newaxis]*patterns[mu, :]
    for i in range(N):
      W[i, i] = 0
    return W/N

#Could be parallelized in copies and patterns
@njit
def noisify(patterns, n_copies, noise_fraction):
    #Returns an array of shape (n_copies, P, N)
    P, N = patterns.shape
    noisy_patterns = np.zeros((n_copies, P, N))
    n_flips = int(noise_fraction * N)
    for a in range(n_copies):
        for mu in range(P):
            noisy_patterns[a, mu, :] = patterns[mu]
            idx = npr.choice(N, n_flips, replace=False)
            noisy_patterns[a, mu, idx] *= -1
    return noisy_patterns

@njit
def convergence(patterns, conv_copies, val_frac, W, thresh):
    P, N = patterns.shape
    train_batch = noisify(patterns, conv_copies, val_frac)
    is_converging = True
    for a in range(conv_copies):
        for mu in range(P):
            final, _ = async_evolve(train_batch[a, mu, :], W, thresh)
            if not np.all(final - patterns[mu] < 1e-04):
                is_converging = False
                break
    return is_converging

@njit(cache=True)
def train(W, patterns, n_copies, conv_copies, noise_frac, val_frac, n_epochs,
          stim, thresh, wake_lr, wake_steps, eval_steps, alpha=1, early_stop=True):
    P, N = patterns.shape
    noisy_patterns = noisify(patterns, n_copies, noise_frac) #Training set
    curr_act = np.ones(N)
    S_mat = np.zeros((wake_steps, N, N))
    tW = true_weights(patterns) #Hebbian weights, just for tracking performance, never used to inform training
    norm_true = np.zeros(n_epochs)
    norm_prev = np.zeros(n_epochs)
    epoch = 0
    stop = False
    while not stop and epoch < n_epochs:
        a, mu = npr.randint(n_copies), npr.randint(P)
        stimulus = noisy_patterns[a, mu, :]
        #Stimulation
        for t in range(wake_steps):
            order = npr.permutation(N)
            for i in order:
                field = W@curr_act + stim*stimulus
                curr_act[i] = np.sign(field[i] + thresh[i])
            S_mat[t] = np.outer(curr_act, curr_act)
        #Weight update
        dW = wake_lr/N**2*np.sum(S_mat, axis=0) - alpha*wake_lr*W/N
        for i in range(N):
          dW[i, i] = 0
        W = W + dW
        norm_true[epoch] = np.linalg.norm(W - tW) #Track distance from true weights
        norm_prev[epoch] = np.linalg.norm(dW) #Track distance from previous weights
        epoch += 1
        if epoch % (max(n_epochs // 100, 1)) == 0:
            print(f'Epoch {epoch} done')
        if convergence(patterns, conv_copies, val_frac, W, thresh) and early_stop:
            stop = True
    return W, norm_true[:epoch], norm_prev[:epoch]

#Could be parallelized (sum)
@njit(cache=True)
def sleep(W, sleep_steps, sleep_lr):
    """
    Unlearning rule from 2019 paper Dreaming Neural Network
    """
    N = W.shape[0]
    for k in range(sleep_steps):
        W = W + sleep_lr/(1 + sleep_lr*k)*(W - W@W)
        for i in range(N):
          W[i, i] = 0
        curr_norm = np.linalg.norm(W)
        W = W / curr_norm
    return W

@njit(cache=True)
def sleep_v2(W, threshold, patterns, sleep_steps, norm_steps, sleep_lr):
    """
    Unlearning rule from 2024 paper Daydreaming neural networks
    """
    P, N = patterns.shape
    for t in range(sleep_steps//norm_steps):
      for u in range(norm_steps):
        mu = npr.randint(P)
        pattern = patterns[mu]
        init = 2*npr.randint(0, 1, size=N) - 1.0
        fx_point, _ = async_evolve(init, W, threshold)
        dW = (np.outer(pattern, pattern) - np.outer(fx_point, fx_point))/sleep_lr/N
        for i in range(N):
          dW[i, i] = 0
        W = W + dW
      W = W/np.linalg.norm(W) #Not necessarily the right norm
    return W

@njit
def async_evolve(input, W, thresh, until_conv=True, max_steps=0):
    #Either go on until convergence or stop within a number of steps
    N = input.size
    act = input.copy()
    pre_act = input.copy()
    t = 0
    conv = False
    while not conv:
      order = npr.permutation(N)
      for i in order:
          field = W@act
          act[i] = np.sign(field[i] + thresh[i])
      if np.all(act - pre_act < 1e-04):
          conv = True
      pre_act = act.copy()
      t += 1
      if t >= max_steps and not until_conv:
          break
    return act, t
