from data_utils import load_data, prepare
from nn_core import NeuralNetwork, bce_loss

data = prepare(load_data("sensor_dataset.csv"))

# --- Test 1: shapes ---
net = NeuralNetwork(seed=42)
print("Weight shapes:", [w.shape for w in net.W])
print("Bias lengths: ", [len(b) for b in net.b])

# --- Test 2: are the gradients correct? ---
X, Y = data["X_train"][:10], data["Y_train"][:10]
_, cache = net.forward(X)
gW, gb = net.backward(cache, Y)
h = 1e-5
net.W[1][2, 3] += h;     loss_plus = bce_loss(Y, net.predict_proba(X))
net.W[1][2, 3] -= 2 * h; loss_minus = bce_loss(Y, net.predict_proba(X))
net.W[1][2, 3] += h
print("\nBackprop gradient:", round(gW[1][2, 3], 8))
print("Numeric gradient: ", round((loss_plus - loss_minus) / (2 * h), 8))

# --- Test 3: does it learn? ---
for act in ["ReLU", "Sigmoid"]:
    net = NeuralNetwork(hidden_activation=act, seed=42)
    hist, _ = net.train(data["X_train"], data["Y_train"],
                        data["X_val"], data["Y_val"], lr=0.01, epochs=100)
    print(f"\n{act}: train loss {hist['train_loss'][0]:.4f} -> {hist['train_loss'][-1]:.4f}"
          f" | val loss {hist['val_loss'][-1]:.4f} | val acc {hist['val_acc'][-1]:.1%}")