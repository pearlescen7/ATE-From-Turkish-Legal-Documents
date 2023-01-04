######################################
# SECTION
# LOAD CANDIDATES
######################################

candidates = []

with open("./bert_dataset/pos_candidates_without_tags.txt", "r", encoding="utf-8") as f:
  for line in f.readlines():
    candidates.append(line.rstrip().lstrip())

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
# GENERATE EMBEDDINGS FOR UNLABELED DATA
######################################

# Get embeddings for unlabeled data
unlabeled_data = candidates
unlabeled_embeddings = torch.Tensor([])
unlabeled_words = []

start_idx = 57561

with torch.no_grad():
  for i in range(start_idx, len(unlabeled_data)):
    
    if (i % 1000 == 0) and (i != 0):
      print(i, " tokens done")
      print(unlabeled_embeddings.shape)

    candidate_words = unlabeled_data[i].split(" ")
    candidate_len = len(candidate_words)
    candidate_embeddings = torch.Tensor([])

    if candidate_len <= 4:
      for j in range(4-candidate_len):
        candidate_embeddings = torch.cat( (candidate_embeddings, torch.zeros((1,768))) )
      
      for j in range(candidate_len):
        encoded_tokens = bert_t(candidate_words[j], return_tensors='pt', padding=True, truncation=True).to("cuda:0")
        output = bert_m(**encoded_tokens)
        embedding = torch.mean(output.last_hidden_state.to("cpu"), dim=1)
        candidate_embeddings = torch.cat( (embedding, candidate_embeddings) )
      
      unlabeled_words.append(unlabeled_data[i])
      unlabeled_embeddings = torch.cat( (unlabeled_embeddings, candidate_embeddings.unsqueeze(dim=0) ), 0)

    if (len(unlabeled_embeddings) == 50000) or (i == len(unlabeled_data)-1):
      torch.save(unlabeled_embeddings, f"./embeddings/unlabeled_pos/unlabeled_embeddings_{i+1}.pt")

      with open(f"./embeddings/unlabeled_pos/unlabeled_words_{i+1}.txt", "w", encoding="utf-8") as fp:
        for j in range(len(unlabeled_words)):
          fp.write(unlabeled_words[j] + "\n")

      unlabeled_embeddings = torch.Tensor([])
      unlabeled_words = []  
