import csv
import matplotlib.pyplot as plt

steps = []
train_losses = []
val_losses = []

with open("out/loss_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        train_losses.append(float(row["train_loss"]))
        val_losses.append(float(row["val_loss"]))

plt.plot(steps, train_losses, label="Train Loss")
plt.plot(steps, val_losses, label="Validation Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.show()