######################################
# SECTION
# LIBRARY IMPORTS
######################################
import torch
import random, heapq
from itertools import count

######################################
# SECTION
# LOAD PREPROCESSED DATA
######################################

torch.cuda.empty_cache()

# u50 = torch.load("./embeddings/unlabeled/unlabeled_embeddings_50000.pt")
# u100 = torch.load("./embeddings/unlabeled/unlabeled_embeddings_100000.pt")
# u150 = torch.load("./embeddings/unlabeled/unlabeled_embeddings_150000.pt")
# u200 = torch.load("./embeddings/unlabeled/unlabeled_embeddings_200000.pt")
# u250 = torch.load("./embeddings/unlabeled/unlabeled_embeddings_250000.pt")
# unlabeled_embeddings = torch.cat((u50, u100, u150, u200, u250), 0)

u50 = torch.load("./embeddings/unlabeled_pos/unlabeled_embeddings_57561.pt")
u100 = torch.load("./embeddings/unlabeled_pos/unlabeled_embeddings_101684.pt")

unlabeled_embeddings = torch.cat((u50, u100), 0)

unlabeled_words = []
# with open("./embeddings/unlabeled/unlabeled_words_50000.txt", "r") as fp:
#   for line in fp.readlines():
#     unlabeled_words.append(line.lstrip().rstrip())

# with open("./embeddings/unlabeled/unlabeled_words_100000.txt", "r") as fp:
#   for line in fp.readlines():
#     unlabeled_words.append(line.lstrip().rstrip())

with open("./embeddings/unlabeled_pos/unlabeled_words_57561.txt", "r", encoding="utf-8") as fp:
  for line in fp.readlines():
    unlabeled_words.append(line.lstrip().rstrip())

with open("./embeddings/unlabeled_pos/unlabeled_words_101684.txt", "r", encoding="utf-8") as fp:
  for line in fp.readlines():
    unlabeled_words.append(line.lstrip().rstrip())

for i in range(len(unlabeled_words)):
  new_idx = random.randint(0, len(unlabeled_words)-1)
  unlabeled_words[i], unlabeled_words[new_idx] = unlabeled_words[new_idx], unlabeled_words[i]
  temp = unlabeled_embeddings[i].clone()
  unlabeled_embeddings[i] = unlabeled_embeddings[new_idx]
  unlabeled_embeddings[new_idx] = temp

temp_labeled_embeddings = torch.load("./embeddings/labeled/labeled_embeddings_updated.pt")
labels = torch.load("./embeddings/labeled/labels_updated.pt")

label_words = []
with open("./embeddings/labeled/label_words_updated.txt", "r") as fp:
  for line in fp.readlines():
    label_words.append(line.lstrip().rstrip())

labeled_embeddings = []
for i in range(len(temp_labeled_embeddings)):
  labeled_embeddings.append([temp_labeled_embeddings[i], labels[i], label_words[i]])

######################################
# SECTION
# EXPERIMENTAL: PASS ONLY A PART OF LABELED DATA 
######################################

TRAIN_SIZE = 1000
VALIDATION_SIZE = 1000
# TEST_SIZE = 5800
test_labeled_embeddings = labeled_embeddings[TRAIN_SIZE+VALIDATION_SIZE:]
valid_labeled_embeddings = labeled_embeddings[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
labeled_embeddings = labeled_embeddings[:TRAIN_SIZE]

with open("./bert_dataset/used_labels.txt", 'w', encoding="utf-8") as fp:
  for e in labeled_embeddings:
    fp.write(e[2] + "\n")

######################################
# SECTION
# LSTM MODEL
######################################

#LSTM with 
#Sequence length = 4  <= Check this part
#Input size = 768
#Batch size = __batch_size
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

class ExtractorLSTM(torch.nn.Module):
    def __init__(self, input_size=768, hidden_size=768*2, batch_size=1, batch_first=True):
        super(ExtractorLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        self.activation = torch.nn.Mish()
        self.linear = torch.nn.Linear(hidden_size, 2)
        self.softmax = torch.nn.LogSoftmax(dim=0)
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
      output = torch.Tensor([]).cuda()
      hidden = torch.zeros(1,1,self.hidden_size).cuda()
      cell = torch.zeros(1,1,self.hidden_size).cuda()
      for i in range(self.batch_size):
        curr_output, (hidden, cell) = self.lstm(x[i].unsqueeze(dim=0), (hidden, cell))
        curr_output = curr_output.squeeze()[-1]
        curr_output = self.activation(curr_output)
        curr_output = self.linear(curr_output)
        curr_output = self.softmax(curr_output)
        #curr_output.shape => [2]
        #to stack unsqueeze dim 0 => [1,2] => will accumulate into => [8,2]
        output = torch.cat((output, curr_output.unsqueeze(dim=0)), 0)

      # print(output)
      return output


######################################
# SECTION
# CNN MODEL
######################################

class ExtractorCNN(torch.nn.Module):
    def __init__(self, n_filter=100, batch_size=1):
        super(ExtractorCNN, self).__init__()
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_filter, (2, 768)),
            # torch.nn.ReLU()
            torch.nn.LeakyReLU()
            # torch.nn.Mish()
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_filter, (3, 768)),
            # torch.nn.ReLU()
            torch.nn.LeakyReLU()
            # torch.nn.Mish()
        )
        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_filter, (4, 768)),
            # torch.nn.ReLU()
            torch.nn.LeakyReLU()
            # torch.nn.Mish()
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_filter*3, 2)
        )
        self.softmax = torch.nn.LogSoftmax(dim=0)

        self.n_filter = n_filter
        self.batch_size = batch_size

    def forward(self, x):
      output = torch.Tensor([]).cuda()
      for i in range(self.batch_size):
        out1 = self.cnn1(x[i].unsqueeze(dim=0))
        out2 = self.cnn2(x[i].unsqueeze(dim=0))
        out3 = self.cnn3(x[i].unsqueeze(dim=0))

        out1_maxed = torch.max(out1, dim=2).values
        out2_maxed = torch.max(out2, dim=2).values
        out3_maxed = torch.max(out3, dim=2).values
        curr_output = torch.cat((out1_maxed, out2_maxed, out3_maxed), dim=1)
        curr_output = curr_output.squeeze()

        curr_output = self.linear(curr_output)
        curr_output = self.softmax(curr_output)

        #curr_output.shape => [2]
        #to stack unsqueeze dim 0 => [1,2] => will accumulate into => [8,2]
        output = torch.cat((output, curr_output.unsqueeze(dim=0)), 0)

      # print(output)
      return output  

######################################
# SECTION
# MODEL DEFINITIONS AND HYPERPARAMETERS
######################################

#Settings and inits
__batch_size = 200
COTRAINING_ITER = 100
ch_cnt = count()
lh_cnt = count()
# EPOCHS = 1
P = 1000
G = 25
LR = 0.001

cnn_m = ExtractorCNN(batch_size=__batch_size)
# cnn_optimizer = torch.optim.SGD(cnn_m.parameters(), lr=LR, momentum=0.9, weight_decay=0.001)
cnn_optimizer = torch.optim.AdamW(cnn_m.parameters(), lr=LR)
cnn_scheduler = torch.optim.lr_scheduler.StepLR(cnn_optimizer, step_size=20, gamma=0.1)
cnn_criterion = torch.nn.NLLLoss()

cnn_m.cuda()
# print(cnn_m.device)

lstm_m = ExtractorLSTM(batch_size=__batch_size)
# lstm_optimizer = torch.optim.SGD(lstm_m.parameters(), lr=LR, momentum=0.9, weight_decay=0.001)
lstm_optimizer = torch.optim.AdamW(lstm_m.parameters(), lr=LR)
lstm_scheduler = torch.optim.lr_scheduler.StepLR(lstm_optimizer, step_size=20, gamma=0.1)
lstm_criterion = torch.nn.NLLLoss()

lstm_m.cuda()


######################################
# SECTION
# GENERATE DICTIONARIES FOR COTRAINING
######################################

unlabeled_emb_idx2word = {}
# unlabeled_word2emb = {}
unlabeled_word2idx = {}

for i in range(len(unlabeled_embeddings)):
  unlabeled_emb_idx2word[i] = unlabeled_words[i]
  unlabeled_word2idx[unlabeled_words[i]] = i
  # unlabeled_word2emb[unlabeled_words[i]] = unlabeled_embeddings[i]

######################################
# SECTION
# COTRAINING LOOP
######################################

guessed_idx = set([])
all_guessed_terms = []

cnn_train_losses = []
lstm_train_losses = []

cnn_valid_losses = []
lstm_valid_losses = []

for idx in range(COTRAINING_ITER):
  
  cnn_train_loss = 0.0
  lstm_train_loss = 0.0

  cnn_valid_loss = 0.0
  lstm_valid_loss = 0.0

  if idx % 10 == 0:
    print(f"iter {idx} done")

  U_hat = []
  U_hat_words = []

  # fill_idx = 0
  # added_count = 0
  # while added_count < P:
  #   if fill_idx not in guessed_idx:
  #     U_hat.append(unlabeled_embeddings[fill_idx])
  #     U_hat_words.append(unlabeled_emb_idx2word[fill_idx])
  #     added_count += 1
  #   fill_idx += 1

  # print("added count: ", added_count)

  U_hat = unlabeled_embeddings[idx*P:(idx+1)*P]
  U_hat_words = unlabeled_words[idx*P:(idx+1)*P]

  cnn_m.train()
  lstm_m.train()
  #Train the model first
  for i in range(0, len(labeled_embeddings), __batch_size):

    if i+__batch_size < len(labeled_embeddings)+1:
      #Get input here
      __input = torch.Tensor([]).cuda()
      __label = torch.Tensor(labeled_embeddings[i][1][0]).unsqueeze(dim=0).cuda()
      for j in range(__batch_size):
        __input = torch.cat((__input, labeled_embeddings[i+j][0].unsqueeze(dim=0).cuda()), 0)
        # __input = torch.cat((__input, labeled_embeddings[i+j][0].cuda()), 0)
        if j != 0:
          #note: loss expects an index since the labels are in the form of [1,0] setting the 
          # second value as target works 
          __label = torch.cat((__label, labeled_embeddings[i+j][1][1].unsqueeze(dim=0).cuda()), 0)


      # print("training input shape: ", __input.shape)
      # print("training label shape: ", __label.shape)

      cnn_optimizer.zero_grad()
      cnn_output = cnn_m(__input.unsqueeze(dim=1))

      # for j in range(__batch_size):
      #   cnn_output[j] = torch.exp(cnn_output[j])

      # print(cnn_output)
      # print(__label)

      cnn_loss = cnn_criterion(cnn_output, __label.long())

      cnn_train_loss += cnn_loss.item()
      # if i%100==0:
      #   print("cnn_loss: ", cnn_loss)
      cnn_loss.backward()
      cnn_optimizer.step()

      lstm_optimizer.zero_grad()
      lstm_output = lstm_m(__input)

      # # for j in range(__batch_size):
      # #   lstm_output[j] = torch.exp(lstm_output[j])

      # # print("lstm output shape: ", lstm_output.shape)

      lstm_loss = lstm_criterion(lstm_output, __label.long())

      lstm_train_loss += lstm_loss.item()

      # # if i%100==0:
      #   # print("lstm_loss: ", lstm_loss)
      lstm_loss.backward()
      lstm_optimizer.step()

  random.shuffle(labeled_embeddings)  

  labeled_emb_word2idx = {}
  for i in range(len(labeled_embeddings)):
    labeled_emb_word2idx[labeled_embeddings[i][2]] = i


  cnn_train_loss = cnn_train_loss/(len(labeled_embeddings)/__batch_size)
  lstm_train_loss = lstm_train_loss/(len(labeled_embeddings)/__batch_size)
  print("cnn_train_loss: ", cnn_train_loss)
  print("lstm_train_loss: ", lstm_train_loss)
  
  cnn_train_losses.append(cnn_train_loss)
  lstm_train_losses.append(lstm_train_loss)

  cnn_scheduler.step()
  lstm_scheduler.step()

  cnn_m.eval()
  lstm_m.eval()
  with torch.no_grad():
    #Validate the model
    for i in range(0, len(valid_labeled_embeddings), __batch_size):

      if i+__batch_size < len(valid_labeled_embeddings):
        #Get input here
        __input = torch.Tensor([]).cuda()
        __label = torch.Tensor(valid_labeled_embeddings[i][1][0]).unsqueeze(dim=0).cuda()
        for j in range(__batch_size):
          __input = torch.cat((__input, valid_labeled_embeddings[i+j][0].unsqueeze(dim=0).cuda()), 0)
          if j != 0:
            #note: loss expects an index since the labels are in the form of [1,0] setting the 
            # second value as target works 
            __label = torch.cat((__label, valid_labeled_embeddings[i+j][1][1].unsqueeze(dim=0).cuda()), 0)

        cnn_output = cnn_m(__input.unsqueeze(dim=1))
        lstm_output = lstm_m(__input)

        cnn_loss = cnn_criterion(cnn_output, __label.long())
        cnn_valid_loss += cnn_loss.item()

        lstm_loss = lstm_criterion(lstm_output, __label.long())
        lstm_valid_loss += lstm_loss.item()

    cnn_valid_loss = cnn_valid_loss/(len(labeled_embeddings)/__batch_size)
    lstm_valid_loss = lstm_valid_loss/(len(labeled_embeddings)/__batch_size)
    print("cnn_valid_loss: ", cnn_valid_loss)
    print("lstm_valid_loss: ", lstm_valid_loss)
    
    cnn_valid_losses.append(cnn_valid_loss)
    lstm_valid_losses.append(lstm_valid_loss)

    if(idx % 20 == 0 and idx != 0):
      torch.save(cnn_m, f"./cotraining_output/cnn_model_{idx}.pt")
      torch.save(cnn_m.state_dict(), f"./cotraining_output/cnn_state_dict_{idx}.pt")

      torch.save(lstm_m, f"./cotraining_output/lstm_model_{idx}.pt")
      torch.save(lstm_m.state_dict(), f"./cotraining_output/lstm_state_dict_{idx}.pt")


    #Then expand the ground truth
    cnn_predictions = []
    lstm_predictions = []

    # g elements must be guessed 
    for i in range(0, len(U_hat), __batch_size):
      if i+__batch_size < len(U_hat):
        __input = torch.Tensor([]).cuda()
        for j in range(__batch_size):
          __input = torch.cat((__input, U_hat[i+j].unsqueeze(dim=0).cuda()), 0)
          # __input = torch.cat((__input, U_hat[i+j].cuda()), 0)

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

        #TODO: Heappush is wrong here, nlargest searches again as if the list is not sorted
        for j in range(__batch_size):
          #The candidate is a term
          if(cnn_prediction[j][0] > cnn_prediction[j][1]):
            heapq.heappush(cnn_predictions, [cnn_prediction[j][0].item(), next(ch_cnt), U_hat[i+j], 'p', U_hat_words[i+j]])
          else:
            heapq.heappush(cnn_predictions, [cnn_prediction[j][1].item(), next(ch_cnt), U_hat[i+j], 'n', U_hat_words[i+j]])

        for j in range(__batch_size):
          #The candidate is a term
          if(lstm_prediction[j][0] > lstm_prediction[j][1]):
            heapq.heappush(lstm_predictions, [lstm_prediction[j][0].item(), next(lh_cnt), U_hat[i+j], 'p', U_hat_words[i+j]])
          else:
            heapq.heappush(lstm_predictions, [lstm_prediction[j][1].item(), next(lh_cnt), U_hat[i+j], 'n', U_hat_words[i+j]])

    cnn_guesses = heapq.nlargest(G, cnn_predictions)
    for guess in cnn_guesses:
      guessed_idx.add( unlabeled_word2idx[guess[4]] )
      if guess[4] not in labeled_emb_word2idx:
        if guess[3] == 'p':
          labeled_embeddings.append([guess[2], torch.Tensor([1, 0]), guess[4]])
        else:
          labeled_embeddings.append([guess[2], torch.Tensor([0, 1]), guess[4]])

    lstm_guesses = heapq.nlargest(G, lstm_predictions)
    for guess in lstm_guesses:
      guessed_idx.add( unlabeled_word2idx[guess[4]] )
      if guess[4] not in labeled_emb_word2idx:
        if guess[3] == 'p':
          labeled_embeddings.append([guess[2], torch.Tensor([1, 0]), guess[4]])
        else:
          labeled_embeddings.append([guess[2], torch.Tensor([0, 1]), guess[4]])
    
    all_guessed_terms.extend(cnn_guesses)
    all_guessed_terms.extend(lstm_guesses)

    # print(all_guessed_terms[4])

######################################
# SECTION
# SAVE MODELS, LOSSES AND GUESSED TERMS
######################################

torch.save(cnn_m, "./cotraining_output/cnn_model_final.pt")
torch.save(lstm_m, "./cotraining_output/lstm_model_final.pt")

torch.save(cnn_m.state_dict(), "./cotraining_output/cnn_state_dict_final.pt")
torch.save(lstm_m.state_dict(), "./cotraining_output/lstm_state_dict_final.pt")

import pickle

with open("./cotraining_output/cnn_train_losses.pkl", "wb") as f:
  pickle.dump(cnn_train_losses, f)

with open("./cotraining_output/lstm_train_losses.pkl", "wb") as f:
  pickle.dump(lstm_train_losses, f)

with open("./cotraining_output/cnn_valid_losses.pkl", "wb") as f:
  pickle.dump(cnn_valid_losses, f)

with open("./cotraining_output/lstm_valid_losses.pkl", "wb") as f:
  pickle.dump(lstm_valid_losses, f)

with open("./cotraining_output/all_guessed_terms.txt", "w", encoding="utf-8") as f:
  all_guessed_terms.sort()
  for guess in all_guessed_terms:
    f.write(guess[4] + " " + guess[3] + "\n")

print("COTRAINING_ITER: ", COTRAINING_ITER)
print("TRAIN_SIZE: ", TRAIN_SIZE)
print("VALID_SIZE: ", VALIDATION_SIZE)
print("P: ", P)
print("G: ", G)
print("LR: ", LR)

######################################
# SECTION
# TEST ACCURACY AND RESULTS
######################################

# test_dict = {}
# for i in range(len(test_labeled_embeddings)):
#   if int(test_labeled_embeddings[i][1][0]) == 1:
#     test_dict[test_labeled_embeddings[i][2]] = 'p'
#   else:
#     test_dict[test_labeled_embeddings[i][2]] = 'n'


# guessed_dict = {}
# for guess in all_guessed_terms:
#   # Also replaces the guessed term with the newest label if there is duplicate
#   guessed_dict[guess[4]] = guess[3]


# terms_found = 0
# tp = 0
# fp = 0
# tn = 0
# fn = 0

# for key in guessed_dict.keys():
#   if key in test_dict:
#     terms_found += 1
#     if test_dict[key] == guessed_dict[key]:
#       if guessed_dict[key] == 'p':
#         tp += 1
#       else:
#         tn += 1
#     else:
#       if guessed_dict[key] == 'p':
#         fp += 1
#       else:
#         fn += 1

# recognized_terms = terms_found/len(test_dict.keys()) * 100
# precision = tp/(tp+fp) * 100
# recall    = tp/(tp+fn) * 100
# accuracy  = (tp+tn)/(tp+tn+fp+fn) * 100

# print("Recognized terms: ", recognized_terms)
# print("Precision: ", precision)
# print("Recall: ", recall)
# print("Accuracy: ", accuracy)



# ######################################
# # SECTION
# # LOAD LOSSES AND GUESSED TERMS
# ######################################

# # import pickle

# # with open("./cotraining_output/cnn_train_losses.pkl", "rb") as f:
# #   cnn_train_losses = pickle.load(cnn_train_losses, f)

# # with open("./cotraining_output/lstm_train_losses.pkl", "rb") as f:
# #   lstm_train_losses = pickle.load(lstm_train_losses, f)

# # TODO: Add reading of all_guessed_terms

