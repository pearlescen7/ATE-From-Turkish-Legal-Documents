######################################
# SECTION
# LIBRARY IMPORTS
######################################

import nltk
import os, re, random, heapq

#Install NLTK packages
nltk.download('punkt')


######################################
# SECTION
# LOAD TERMS CRAWLED FROM DICTIONARY
######################################

gt_terimler = []

with open("./bert_dataset/terimler_filtreli.txt", encoding="utf-8") as gt_file:
  gt_terimler = gt_file.readlines()

with open("./bert_dataset/terimler_updated.txt", encoding="utf-8") as gt_file:
  gt_terimler.extend(gt_file.readlines())

gt_terimler = [terim.strip().lower() for terim in gt_terimler]
gt_terimler_set = set(gt_terimler)

print(gt_terimler)
print(len(gt_terimler))
#3888 terms

######################################
# SECTION
# LOAD NON-TERMS
######################################

frequent_words = []

with open("./bert_dataset/zemberek_full.txt", encoding="utf-8") as fp:
  for line in fp.readlines():
    token = line.rstrip().lstrip().lower().split(" ")
    token[0], token[1] = int(token[1]), token[0]
    heapq.heappush(frequent_words, token)

freq_pair = heapq.nlargest(4000, frequent_words)

freq_nk = []
for term_pair in freq_pair:
  if term_pair[1] not in gt_terimler_set:
    freq_nk.append(term_pair[1])
    
print(freq_nk)
print(len(freq_nk))

######################################
# SECTION
# CREATE THE LABELED DATASET
######################################

labels = [[1,0] for terim in gt_terimler]
labels.extend([[0,1] for word in freq_nk])

labeled_words = []
labeled_words.extend(gt_terimler)
labeled_words.extend(freq_nk)

labeled_data = [[labeled_words[i], labels[i]]for i in range(len(labeled_words))]
random.shuffle(labeled_data)

######################################
# SECTION
# IMPORT DOMAIN-ADAPTED BERT MODEL
######################################

#Import BERT
import torch
from transformers import BertTokenizer, AutoModel

bert_t = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
bert_m = AutoModel.from_pretrained("./bert-turkish-legal-pretrained")
bert_m.cuda()
# bert_m.device

######################################
# SECTION
# GENERATE EMBEDDINGS FOR LABELED DATA
######################################

# Get embeddings for labeled data
labeled_embeddings = []

with torch.no_grad():

  for i in range(len(labeled_data)):
    candidate_words = labeled_data[i][0].split(" ")
    candidate_len = len(candidate_words)
    cur_embedding = torch.Tensor([])
    if candidate_len <= 4:
      for j in range(4-candidate_len):
        cur_embedding = torch.cat( (cur_embedding, torch.zeros((1,768))) )

      for j in range(candidate_len):
        encoded_tokens = bert_t(candidate_words[j], return_tensors='pt', padding=True, truncation=True).to("cuda:0")
        output = bert_m(**encoded_tokens)
        embedding = torch.mean(output.last_hidden_state.to("cpu"), dim=1)
        cur_embedding = torch.cat( (embedding, cur_embedding) )
        # print(output.last_hidden_state)
        # print(output.last_hidden_state.shape)
      

      # embedding = torch.mean(output.last_hidden_state, dim=1)
      labeled_embeddings.append([cur_embedding, labeled_data[i][1], labeled_data[i][0]])

#Embeddings are in shape (4, 768)

######################################
# SECTION
# SAVE EMBEDDINGS FOR LABELED DATA
######################################

labeled_emb_t = torch.Tensor(labeled_embeddings[0][0].unsqueeze(dim=0))
label_t = torch.Tensor([labeled_embeddings[0][1]])
label_word = []

for i in range(len(labeled_embeddings)):
  if i != 0:
    labeled_emb_t = torch.cat( (labeled_emb_t, labeled_embeddings[i][0].unsqueeze(dim=0)) )
    label_t = torch.cat((label_t, torch.Tensor([labeled_embeddings[i][1]])), 0)
  label_word.append(labeled_embeddings[i][2])

torch.save(labeled_emb_t, "./embeddings/labeled/labeled_embeddings_updated.pt")
torch.save(label_t, "./embeddings/labeled/labels_updated.pt")

with open("./embeddings/labeled/label_words_updated.txt", "w", encoding="utf-8") as fp:
  for i in range(len(label_word)):
    fp.write(label_word[i]+"\n")

print(labeled_emb_t.shape)
print(label_t.shape)
print(len(label_word))