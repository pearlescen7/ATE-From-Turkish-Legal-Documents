import torch
from transformers import BertTokenizer, AutoModel
from models import ExtractorCNN, ExtractorLSTM

bert_t = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
bert_m = AutoModel.from_pretrained("./bert-turkish-legal-pretrained")
bert_m.cuda()

cnn_m = torch.load("./cotraining_output/30_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_10_sched/cnn_model_final.pt")
lstm_m = torch.load("./cotraining_output/30_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_10_sched/lstm_model_final.pt")
batch_size = 8

cnn_m.cuda()
lstm_m.cuda()

cnn_m.eval()
lstm_m.eval()

ngram = "bir"

with torch.no_grad():
    ngram = ngram.strip().lower().split(" ")
    print(ngram)
    ngram_embedding = torch.Tensor([])
    for j in range(4-len(ngram)):
        ngram_embedding = torch.cat( (ngram_embedding, torch.zeros((1,768))) )

    for word in ngram:
        encoded_tokens = bert_t(word, return_tensors='pt', padding=True, truncation=True).to("cuda:0")
        output = bert_m(**encoded_tokens)
        cur_embedding = torch.mean(output.last_hidden_state.to("cpu"), dim=1)
        ngram_embedding = torch.cat( (cur_embedding, ngram_embedding) )

    # print(ngram_embedding.shape)

    ngram_embedding = ngram_embedding.unsqueeze(dim=0)
    ngram_embedding = torch.cat( (ngram_embedding, torch.zeros((7,4,768))) )

    print(ngram_embedding.shape)

    cnn_output = cnn_m(ngram_embedding.unsqueeze(dim=1).cuda())
    lstm_output = lstm_m(ngram_embedding.cuda())

    if cnn_output[0][0] > cnn_output[0][1]:
        print("CNN Prediction: Term")
    else:
        print("CNN Prediction: Not Term")

    if lstm_output[0][0] > lstm_output[0][1]:
        print("LSTM Prediction: Term")
    else:
        print("LSTM Prediction: Not Term")