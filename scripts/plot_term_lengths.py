
import matplotlib.pyplot as plt
 
terms = []
with open("./bert_dataset/terimler_filtreli.txt", "r") as fp:
  for line in fp.readlines():
    terms.append(line.lstrip().rstrip())

# creating the dataset
ngram_counts = [0, 0, 0, 0]
graph_labels = ["unigram", "bigram", "trigram", "4-gram"]
for i in range(len(terms)):
    idx = len(terms[i].split(" "))-1
    if idx < 4:
        ngram_counts[idx] +=1

fig = plt.figure(figsize = (10, 5))

ngram_percentage = list(map(lambda x: x*100/len(terms), ngram_counts))

# creating the bar plot
# plt.bar(graph_labels, ngram_percentage, width = 0.4)
plt.bar(graph_labels, ngram_counts, width = 0.4)
 
plt.xlabel("Word count")
plt.ylabel("Number of terms")
plt.title("Distribution of Number of Words in Terms")
plt.savefig("term_counts.png")

print(ngram_counts)
#updated [6768, 700, 299, 39]
#old [3102, 299, 127, 18]
# plt.show()