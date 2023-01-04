import heapq
import matplotlib.pyplot as plt

gt_terimler = []

with open("./bert_dataset/terimler_filtreli.txt", encoding="utf-8") as gt_file:
  gt_terimler = gt_file.readlines()

gt_terimler = [terim.strip().lower() for terim in gt_terimler]
gt_terimler_set = set(gt_terimler)

print(gt_terimler)
print(len(gt_terimler))

frequent_words = []

with open("./bert_dataset/zemberek_full.txt", encoding="utf-8") as fp:
  for line in fp.readlines():
    token = line.rstrip().lstrip().lower().split(" ")
    token[0], token[1] = int(token[1]), token[0]
    heapq.heappush(frequent_words, token)

freq_pair = heapq.nlargest(2000, frequent_words)

freq_nk = []
for term_pair in freq_pair:
  if term_pair[1] not in gt_terimler_set:
    freq_nk.append(term_pair[1])
    
print(freq_nk)
print(len(freq_nk))

total_labeled = len(freq_nk) + len(gt_terimler)
per_gt = 100*len(gt_terimler)/total_labeled
per_freq = 100*len(freq_nk)/total_labeled

graph_labels = ["Terms", "Non-Terms"]

fig = plt.figure(figsize = (10, 5))

plt.bar(graph_labels, [per_gt, per_freq], width=0.4)
 
# plt.xlabel("Label")
plt.ylabel("Distribution (%)")
plt.title("Distribution of the Labeled N-grams")
plt.savefig("data_distribution.png")