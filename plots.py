import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load your CSV file
# ---------------------------
expert = "A"
window = 1
chunk = 20
filename = f"epoch_metrics_avg_expert_{expert}_W{window}_C{chunk}_new.csv"

df = pd.read_csv(filename)

# ---------------------------
# Split training vs validation
# ---------------------------
train_df = df[df["dataset"] == "train"]
val_df = df[df["dataset"] == "val"]

# ---------------------------
# Plot loss per epoch
# ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_df["epoch"], train_df["loss"], color="red", label="Training Loss")
plt.plot(val_df["epoch"], val_df["loss"], color="blue", label="Validation Loss")
plt.title(f"Loss per Epoch - Expert {expert}, W{window}, C{chunk}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
