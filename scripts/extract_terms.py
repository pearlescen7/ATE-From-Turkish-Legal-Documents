import torch
from models import ExtractorCNN, ExtractorLSTM
from itertools import count
import heapq

torch.cuda.empty_cache()

u50 = torch.load("./embeddings/unlabeled/unlabeled_embeddings_50000.pt")
u100 = torch.load("./embeddings/unlabeled/unlabeled_embeddings_100000.pt")

# u50 = torch.load("./embeddings/unlabeled_pos/unlabeled_embeddings_57561.pt")
# u100 = torch.load("./embeddings/unlabeled_pos/unlabeled_embeddings_101684.pt")

unlabeled_embeddings = torch.cat((u50, u100), 0)

unlabeled_words = []
with open("./embeddings/unlabeled/unlabeled_words_50000.txt", "r") as fp:
  for line in fp.readlines():
    unlabeled_words.append(line.lstrip().rstrip())

with open("./embeddings/unlabeled/unlabeled_words_100000.txt", "r") as fp:
  for line in fp.readlines():
    unlabeled_words.append(line.lstrip().rstrip())

# with open("./embeddings/unlabeled_pos/unlabeled_words_57561.txt", "r", encoding="utf-8") as fp:
#   for line in fp.readlines():
#     unlabeled_words.append(line.lstrip().rstrip())

# with open("./embeddings/unlabeled_pos/unlabeled_words_101684.txt", "r", encoding="utf-8") as fp:
#   for line in fp.readlines():
#     unlabeled_words.append(line.lstrip().rstrip())

unlabeled_emb_idx2word = {}
unlabeled_word2idx = {}
for i in range(len(unlabeled_embeddings)):
  unlabeled_emb_idx2word[i] = unlabeled_words[i]
  unlabeled_word2idx[unlabeled_words[i]] = i

model = "100_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_20_sched"

cnn_m = torch.load(f"./cotraining_output/{model}/cnn_model_final.pt")
lstm_m = torch.load(f"./cotraining_output/{model}/lstm_model_final.pt")
__batch_size = 8

cnn_m.batch_size = __batch_size
lstm_m.batch_size = __batch_size

cnn_m.cuda()
lstm_m.cuda()

cnn_m.eval()
lstm_m.eval()

with torch.no_grad():

    cnn_predictions = []
    lstm_predictions = []
    ch_cnt = count()
    lh_cnt = count()


    for i in range(0, len(unlabeled_embeddings), __batch_size):
      if i+__batch_size < len(unlabeled_embeddings):
        __input = torch.Tensor([]).cuda()
        for j in range(__batch_size):
          __input = torch.cat((__input, unlabeled_embeddings[i+j].unsqueeze(dim=0).cuda()), 0)

        cnn_prediction = cnn_m(__input.unsqueeze(dim=1)).to("cpu")
        lstm_prediction = lstm_m(__input).to("cpu")

        for j in range(__batch_size):
          cnn_prediction[j] = torch.exp(cnn_prediction[j])
          lstm_prediction[j] = torch.exp(lstm_prediction[j])

        # print(cnn_prediction)
        # print(cnn_prediction.shape)

        # print(lstm_prediction)
        # print(lstm_prediction.shape)
        # print(cnn_prediction[j][0])

        for j in range(__batch_size):
          #The candidate is a term
          if(cnn_prediction[j][0] > cnn_prediction[j][1]):
            heapq.heappush(cnn_predictions, [cnn_prediction[j][0].item(), next(ch_cnt), unlabeled_embeddings[i+j], 'p', unlabeled_words[i+j]])
          else:
            heapq.heappush(cnn_predictions, [cnn_prediction[j][1].item(), next(ch_cnt), unlabeled_embeddings[i+j], 'n', unlabeled_words[i+j]])

        for j in range(__batch_size):
          #The candidate is a term
          if(lstm_prediction[j][0] > lstm_prediction[j][1]):
            heapq.heappush(lstm_predictions, [lstm_prediction[j][0].item(), next(lh_cnt), unlabeled_embeddings[i+j], 'p', unlabeled_words[i+j]])
          else:
            heapq.heappush(lstm_predictions, [lstm_prediction[j][1].item(), next(lh_cnt), unlabeled_embeddings[i+j], 'n', unlabeled_words[i+j]])

        if (i%1000==0):
            print(i, " ngrams guessed")

with open(f"./cotraining_output/{model}/cnn_extracted_terms_ngram.txt", "w", encoding="utf-8") as fp:
    for i in range(len(cnn_predictions)):
        if cnn_predictions[i][3] == 'p':
            fp.write(cnn_predictions[i][-1] + " " + str(cnn_predictions[i][0]) + "\n")

with open(f"./cotraining_output/{model}/lstm_extracted_terms_ngram.txt", "w", encoding="utf-8") as fp:
    for i in range(len(lstm_predictions)):
        if lstm_predictions[i][3] == 'p':
            fp.write(lstm_predictions[i][-1] + " " + str(lstm_predictions[i][0]) + "\n")