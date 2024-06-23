import matplotlib as plt
results_file = 'runs/train/spfd2/results.csv'

epochs = []
train_loss = []
val_loss = []
with open(results_file, 'r') as file:
    for line in file:
        values = line.split()
        epochs.append(int(values[0]))
        train_loss.append(float(values[1]))
        val_loss.append(float(values[3]))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()