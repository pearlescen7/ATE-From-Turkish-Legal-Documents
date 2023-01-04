import matplotlib.pyplot as plt
import numpy as np
import torch

gt_terimler = []

with open("./bert_dataset/terimler_filtreli.txt", encoding="utf-8") as gt_file:
  gt_terimler = gt_file.readlines()

with open("./bert_dataset/terimler_updated.txt", encoding="utf-8") as gt_file:
  gt_terimler.extend(gt_file.readlines())

gt_terimler = [terim.strip().lower() for terim in gt_terimler]
gt_terimler_set = set(gt_terimler)

# print(gt_terimler)
print(len(gt_terimler))

cnn_model = "100_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_20_sched"
lstm_model = "30_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_10_sched"
# lstm_model = "100_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_20_sched"

cnn_words = []
lstm_words = []
with open(f"./cotraining_output/{cnn_model}/cnn_1.txt", "r", encoding="utf-8") as fp:
    for line in fp.readlines():
        cnn_words.append(line.split(" ")[0])

with open(f"./cotraining_output/{lstm_model}/lstm_1.txt", "r", encoding="utf-8") as fp:
    for line in fp.readlines():
        lstm_words.append(line.split(" ")[0])

# print(cnn_words)

label_words = []
with open("./embeddings/labeled/label_words_updated.txt", "r") as fp:
  for line in fp.readlines():
    label_words.append(line.lstrip().rstrip())

print(len(label_words))
lwordset = set(label_words[:1000])

cnnDictCount = 0
lstmDictCount = 0

cnn_true_counts = [0]
lstm_true_counts = [0]

for idx, word in enumerate(cnn_words):
    if word in lwordset:
        cnnDictCount += 1
    if word in gt_terimler:
        cnn_true_counts.append(cnn_true_counts[idx]+1)
    else:
        cnn_true_counts.append(cnn_true_counts[idx])

for idx, word in enumerate(lstm_words):
    if word in lwordset:
        lstmDictCount += 1
    if word in gt_terimler:
        lstm_true_counts.append(lstm_true_counts[idx]+1)
    else:
        lstm_true_counts.append(lstm_true_counts[idx])

# print(cnn_true_counts)

for idx,t in enumerate(cnn_true_counts):
    if idx == 0:
        continue
    cnn_true_counts[idx] = 100*cnn_true_counts[idx]/idx

for idx,t in enumerate(lstm_true_counts):
    if idx == 0:
        continue
    lstm_true_counts[idx] = 100*lstm_true_counts[idx]/idx
    
print(f"cnn words in training set: {cnnDictCount}")
print(f"lstm words in training set: {lstmDictCount}")

cnn_true_counts = cnn_true_counts[:250]
lstm_true_counts = lstm_true_counts[:250]

plt.grid()
plt.plot(np.arange(5,len(cnn_true_counts)), cnn_true_counts[5:])
plt.plot(np.arange(5,len(lstm_true_counts)), lstm_true_counts[5:])
plt.title('Inference Accuracy of Models in Top N Extracted Terms')
plt.ylabel('Accuracy')
plt.xlabel('Nth Guessed Word')
plt.legend(['CNN', 'LSTM'], loc='upper right')
plt.savefig("top_accuracy_ngram.png")