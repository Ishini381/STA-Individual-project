from data_utils import load_data, prepare, FEATURES

df = load_data("sensor_dataset.csv")
print("Full data:", df.shape)

data = prepare(df)
print("X_train:", data["X_train"].shape, " Y_train:", data["Y_train"].shape)
print("X_val:  ", data["X_val"].shape,   " Y_val:  ", data["Y_val"].shape)
print("X_test: ", data["X_test"].shape,  " Y_test: ", data["Y_test"].shape)

print("\nTraining means used for scaling:", data["scaler"].mean_.round(2))
print("Training stds used for scaling: ", data["scaler"].std_.round(2))

print("\nScaled X_train mean (should be ~0):", data["X_train"].mean(axis=0).round(4))
print("Scaled X_train std  (should be ~1):", data["X_train"].std(axis=0).round(4))

print("\nFirst training machine:", data["train_df"].loc[0, "machine_id"])