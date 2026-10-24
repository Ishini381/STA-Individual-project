"""Plain-language interpretation of a recorded training history."""
import numpy as np


def describe_history(h):
    tr = np.array(h["train_loss"]); va = np.array(h["val_loss"]); ep = np.array(h["epoch"])
    if len(tr) < 3:
        return ["Not enough epochs recorded to interpret."]
    msgs = []
    drop_tr = (tr[0] - tr[-1]) / tr[0] * 100
    drop_va = (va[0] - va[-1]) / va[0] * 100
    msgs.append(f"Training loss went from {tr[0]:.3f} to {tr[-1]:.3f} ({drop_tr:+.1f}% change); "
                f"validation loss went from {va[0]:.3f} to {va[-1]:.3f} ({drop_va:+.1f}%).")

    # learning still improving or slowing? compare first and last 20% of epochs
    k = max(2, len(tr) // 5)
    early_rate = (tr[0] - tr[k]) / k
    late_rate = (tr[-k - 1] - tr[-1]) / k
    if drop_tr < 2:
        msgs.append("**Barely learning:** the loss hardly moved. The learning rate may be too small "
                    "or there are too few epochs.")
    elif late_rate > 0.5 * early_rate:
        msgs.append("**Still improving:** the training curve is still falling steeply at the end, "
                    "so more epochs (or a larger learning rate) could help.")
    else:
        msgs.append("**Slowing down:** the curve is flattening, the per-epoch improvement at the end is "
                    f"about {late_rate / early_rate * 100 if early_rate > 0 else 0:.0f}% of the early rate. "
                    "Gradient descent is approaching a minimum region.")

    # oscillation: sign changes in successive differences of training loss
    d = np.diff(tr[1:])
    if len(d) > 4:
        flips = np.mean(np.sign(d[1:]) != np.sign(d[:-1]))
        ups = np.mean(d > 0)
        if ups > 0.25:
            msgs.append(f"**Oscillating:** training loss increased in {ups * 100:.0f}% of epochs, "
                        "a sign the step size is large relative to the loss surface.")

    # overfitting: val minimum well before the end while train keeps falling
    best = int(np.argmin(va))
    if best < len(va) - 1 and va[-1] - va[best] > 0.005 and tr[-1] < tr[best]:
        msgs.append(f"**Possible overfitting:** validation loss was lowest ({va[best]:.3f}) at epoch "
                    f"{ep[best]} and has risen to {va[-1]:.3f}, while training loss kept falling. "
                    f"Stopping near epoch {ep[best]} would generalise better.")
    else:
        msgs.append("**No clear overfitting:** validation loss is at or near its lowest value at the end.")
    gap = va[-1] - tr[-1]
    msgs.append(f"Generalisation gap (val − train loss) at the end: {gap:+.3f}. "
                "With only 27 validation rows, small differences are noisy.")
    return msgs
