import numpy as np

EPS = 1e-7  # tiny number so we never take log(0)

def sigmoid(z):
    """1 / (1 + e^-z), written to avoid overflow for large negative z."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def relu(z):
    """max(0, z): keeps positive values, turns negatives into 0."""
    return np.maximum(0.0, z)


def activation(z, name):
    """Apply the hidden-layer activation the user chose."""
    return relu(z) if name == "ReLU" else sigmoid(z)


def activation_grad(z, a, name):
    """Derivative dA/dZ, needed for back-propagation."""
    if name == "ReLU":
        return (z > 0).astype(float)   # 1 where the neuron was active, else 0
    return a * (1 - a)                 # derivative of sigmoid

def bce_loss(y_true, y_prob):
    """Binary cross-entropy, averaged over both outputs and all rows."""
    p = np.clip(y_prob, EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def binary_accuracy(y_true, y_prob, threshold=0.5):
    """Fraction of (row, label) pairs predicted correctly."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))

class NeuralNetwork:
    def __init__(self, layer_sizes=(3, 5, 4, 2), hidden_activation="ReLU", seed=42):
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.seed = seed
        self.init_params()

    def init_params(self):
        """Random weights from a fixed seed, biases start at zero."""
        rng = np.random.default_rng(self.seed)
        self.W = []   # W[0] is 3x5, W[1] is 5x4, W[2] is 4x2
        self.b = []   # b[0] has 5, b[1] has 4, b[2] has 2
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))          # Glorot/Xavier uniform
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.epochs_trained = 0

    def forward(self, X):
        """Z = A_prev @ W + b, then activation. Saves every Z and A."""
        cache = {"A0": X}
        A = X
        L = len(self.W)                      # 3 weight layers
        for l in range(L):
            Z = A @ self.W[l] + self.b[l]    # weighted sum
            if l < L - 1:
                A = activation(Z, self.hidden_activation)   # hidden layers
            else:
                A = sigmoid(Z)               # output layer: always sigmoid
            cache[f"Z{l+1}"] = Z
            cache[f"A{l+1}"] = A
        return A, cache

    def predict_proba(self, X):
        """Just the two output probabilities."""
        return self.forward(X)[0]

    def backward(self, cache, Y):
        """Gradients of the BCE loss for every W and b."""
        L = len(self.W)
        m, k = Y.shape                        # m rows, k = 2 outputs
        grads_W = [None] * L
        grads_b = [None] * L

        # Sigmoid + BCE together give this simple starting gradient:
        dZ = (cache[f"A{L}"] - Y) / (m * k)

        for l in reversed(range(L)):          # go backwards: 2, 1, 0
            A_prev = cache[f"A{l}"]
            grads_W[l] = A_prev.T @ dZ        # how much each weight caused the error
            grads_b[l] = dZ.sum(axis=0)       # how much each bias caused the error
            if l > 0:
                dA_prev = dZ @ self.W[l].T    # send the error back one layer
                dZ = dA_prev * activation_grad(cache[f"Z{l}"], cache[f"A{l}"],
                                               self.hidden_activation)
        return grads_W, grads_b

    def train(self, X_tr, Y_tr, X_val, Y_val, lr=0.01, epochs=100, batch_size=16):
        """Mini-batch SGD. Records real losses after every epoch."""
        rng = np.random.default_rng(self.seed + 1000 + self.epochs_trained)
        history = {"epoch": [], "train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": [], "grad_norm": []}
        weight_snaps = [[w.copy() for w in self.W]]      # weights before training
        n = len(X_tr)

        for ep in range(epochs):
            idx = rng.permutation(n)                     # shuffle every epoch
            norms = []
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                _, cache = self.forward(X_tr[batch])
                gW, gb = self.backward(cache, Y_tr[batch])

                for l in range(len(self.W)):             # gradient descent step
                    self.W[l] -= lr * gW[l]
                    self.b[l] -= lr * gb[l]

                norms.append(np.sqrt(sum(np.sum(g**2) for g in gW + gb)))

            # after each epoch: measure on ALL training and validation rows
            self.epochs_trained += 1
            p_tr = self.predict_proba(X_tr)
            p_val = self.predict_proba(X_val)
            history["epoch"].append(self.epochs_trained)
            history["train_loss"].append(bce_loss(Y_tr, p_tr))
            history["val_loss"].append(bce_loss(Y_val, p_val))
            history["train_acc"].append(binary_accuracy(Y_tr, p_tr))
            history["val_acc"].append(binary_accuracy(Y_val, p_val))
            history["grad_norm"].append(float(np.mean(norms)))
            weight_snaps.append([w.copy() for w in self.W])

        return history, weight_snaps