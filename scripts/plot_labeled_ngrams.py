
import matplotlib.pyplot as plt
import torch
 
temp_labeled_embeddings = torch.load("./embeddings/labeled/labeled_embeddings.pt")
labels = torch.load("./embeddings/labeled/labels.pt")

label_words = []
with open("./embeddings/labeled/label_words.txt", "r") as fp:
  for line in fp.readlines():
    label_words.append(line.lstrip().rstrip())

labeled_embeddings = []
for i in range(len(temp_labeled_embeddings)):
  labeled_embeddings.append([temp_labeled_embeddings[i], labels[i], label_words[i]])

# creating the dataset
ngram_counts = [0, 0, 0, 0]
graph_labels = ["unigram", "bigram", "trigram", "4-gram"]
for i in range(len(labeled_embeddings)):
    ngram_counts[len(label_words[i].split(" "))-1] +=1

fig = plt.figure(figsize = (10, 5))

ngram_percentage = list(map(lambda x: x*100/len(labeled_embeddings), ngram_counts))

# creating the bar plot
plt.bar(graph_labels, ngram_counts, width = 0.4)
 
plt.xlabel("Word count")
plt.ylabel("Number of ngrams")
plt.title("Distribution of ngrams in the labeled set")
plt.savefig("ngram_counts_old.png")

print(ngram_counts)
#updated [6768, 700, 299, 39]
#old [3102, 299, 127, 18]
# plt.show()