import pickle
import matplotlib.pyplot as plt

model = "100_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_20_sched"

with open(f"./cotraining_output/{model}/cnn_train_losses.pkl", "rb") as f:
  cnn_train_losses = pickle.load(f)

with open(f"./cotraining_output/{model}/lstm_train_losses.pkl", "rb") as f:
  lstm_train_losses = pickle.load(f)

with open(f"./cotraining_output/{model}/cnn_valid_losses.pkl", "rb") as f:
  cnn_valid_losses = pickle.load(f)

with open(f"./cotraining_output/{model}/lstm_valid_losses.pkl", "rb") as f:
  lstm_valid_losses = pickle.load(f)


plt.plot(cnn_train_losses)
plt.plot(lstm_train_losses)
plt.title('Train loss')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.legend(['CNN', 'LSTM'], loc='upper right')
plt.savefig("train_loss.png")

plt.clf()
plt.plot(cnn_valid_losses)
plt.plot(lstm_valid_losses)
plt.title('Validation loss')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.legend(['CNN', 'LSTM'], loc='upper right')
plt.savefig("validation_loss.png")
