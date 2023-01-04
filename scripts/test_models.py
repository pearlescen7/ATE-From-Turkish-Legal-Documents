import torch
from models import ExtractorCNN, ExtractorLSTM

temp_labeled_embeddings = torch.load("./embeddings/labeled/labeled_embeddings.pt")
labels = torch.load("./embeddings/labeled/labels.pt")

label_words = []
with open("./embeddings/labeled/label_words.txt", "r") as fp:
  for line in fp.readlines():
    label_words.append(line.lstrip().rstrip())

labeled_embeddings = []
for i in range(len(temp_labeled_embeddings)):
  labeled_embeddings.append([temp_labeled_embeddings[i], labels[i], label_words[i]])

# test_data = labeled_embeddings[0:1500]
# test_data = labeled_embeddings[:-2000]
test_data = labeled_embeddings[1000:]

#####################################################################
# gt_terimler = []

# with open("./bert_dataset/terimler_filtreli.txt", encoding="utf-8") as gt_file:
#   gt_terimler = gt_file.readlines()

# with open("./bert_dataset/terimler_updated.txt", encoding="utf-8") as gt_file:
#   gt_terimler.extend(gt_file.readlines())

# gt_terimler = [terim.strip().lower() for terim in gt_terimler]
# gt_terimler_set = set(gt_terimler)

# temp_test_data = []
# for labeled in test_data:
#     if labeled[1][1]:
#         temp_test_data.append(labeled)

# test_data = temp_test_data
#####################################################################

# for data in test_data:
#     print("label: ", int(data[1][0]), " kelime: ", data[2])

model = "100_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_20_sched"

cnn_m = torch.load(f"./cotraining_output/{model}/cnn_model_final.pt")
lstm_m = torch.load(f"./cotraining_output/{model}/lstm_model_final.pt")
batch_size = 8

cnn_m.batch_size = batch_size
lstm_m.batch_size = batch_size

cnn_m.cuda()
lstm_m.cuda()

cnn_m.eval()
lstm_m.eval()

cnn_tp = 0
cnn_fp = 0
cnn_tn = 0
cnn_fn = 0

lstm_tp = 0
lstm_fp = 0
lstm_tn = 0
lstm_fn = 0

with torch.no_grad():
    for i in range(0, len(test_data), batch_size):
        if i+batch_size < len(test_data):
            input = torch.Tensor([]).cuda()
            for j in range(batch_size):
                input = torch.cat((input, test_data[i+j][0].unsqueeze(dim=0).cuda()), 0)

            cnn_output = cnn_m(input.unsqueeze(dim=1)).to('cpu')
            lstm_output = lstm_m(input).to('cpu')

            for j in range(batch_size):
                cnn_output[j] = torch.exp(cnn_output[j])
                lstm_output[j] = torch.exp(lstm_output[j])
                
                # print(cnn_output[j], " ", lstm_output[j])
                # print(int(test_data[i+j][1][0]), " ", test_data[i+j][2])

                if cnn_output[j][0] > cnn_output[j][1]:
                    if int(test_data[i+j][1][0]) == 1:
                        cnn_tp += 1
                    else:
                        cnn_fp += 1
                else:
                    if int(test_data[i+j][1][0]) == 1:
                        cnn_fn += 1
                    else:
                        cnn_tn += 1

                if lstm_output[j][0] > lstm_output[j][1]:
                    if int(test_data[i+j][1][0]) == 1:
                        lstm_tp += 1
                    else:
                        lstm_fp += 1
                else:
                    if int(test_data[i+j][1][0]) == 1:
                        lstm_fn += 1
                    else:
                        lstm_tn += 1

cnn_precision = cnn_tp/(cnn_tp+cnn_fp) * 100
cnn_recall    = cnn_tp/(cnn_tp+cnn_fn) * 100
cnn_accuracy  = (cnn_tp+cnn_tn)/(cnn_tp+cnn_tn+cnn_fp+cnn_fn) * 100

lstm_precision = lstm_tp/(lstm_tp+lstm_fp) * 100
lstm_recall    = lstm_tp/(lstm_tp+lstm_fn) * 100
lstm_accuracy  = (lstm_tp+lstm_tn)/(lstm_tp+lstm_tn+lstm_fp+lstm_fn) * 100

print("CNN True Positives: ", cnn_tp)
print("CNN True Negatives: ", cnn_tn)
print("CNN False Negatives: ", cnn_fn)
print("CNN False Positives: ", cnn_fp)
print("")
print("LSTM True Positives: ",  lstm_tp)
print("LSTM True Negatives: ",  lstm_tn)
print("LSTM False Negatives: ", lstm_fn)
print("LSTM False Positives: ", lstm_fp)
print("")
print("CNN Precision: ", cnn_precision)
print("CNN Recall: ", cnn_recall)
print("CNN Accuracy: ", cnn_accuracy)
print("")
print("LSTM Precision: ", lstm_precision)
print("LSTM Recall: ", lstm_recall)
print("LSTM Accuracy: ", lstm_accuracy)
print("")
print("CNN F1 Score: ", 2*(cnn_precision*cnn_recall)/(cnn_precision+cnn_recall))
print("LSTM F1 Score: ", 2*(lstm_precision*lstm_recall)/(lstm_precision+lstm_recall))
