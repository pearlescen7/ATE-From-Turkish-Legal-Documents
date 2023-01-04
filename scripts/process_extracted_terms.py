import heapq

cnn_g = [{}, {}, {}, {}]

lstm_g = [{}, {}, {}, {}]

cnn_model = "100_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_20_sched"
# lstm_model = "30_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_10_sched"
lstm_model = "100_iter_1000_labeled_1e-3_lr_25_g_1000_P_adamw_20_sched"

with open(f"./cotraining_output/{cnn_model}/cnn_extracted_terms_ngram.txt", "r", encoding="utf-8") as fp:
    for line in fp.readlines():
        line = line.split(" ")
        n_len = len(line[0:-1])
        ngram = " ".join(line[0:-1])
        cnn_g[n_len-1][ngram] = float(line[-1])

with open(f"./cotraining_output/{lstm_model}/lstm_extracted_terms_ngram.txt", "r", encoding="utf-8") as fp:
    for line in fp.readlines():
        line = line.split(" ")
        n_len = len(line[0:-1])
        ngram = " ".join(line[0:-1])
        lstm_g[n_len-1][ngram] = float(line[-1])

cnn_best1 = heapq.nlargest(500, cnn_g[0].items(), key=lambda i: i[1])
cnn_best2 = heapq.nlargest(500, cnn_g[1].items(), key=lambda i: i[1])
cnn_best3 = heapq.nlargest(500, cnn_g[2].items(), key=lambda i: i[1])
cnn_best4 = heapq.nlargest(500, cnn_g[3].items(), key=lambda i: i[1])

lstm_best1 = heapq.nlargest(500, lstm_g[0].items(), key=lambda i: i[1])
lstm_best2 = heapq.nlargest(500, lstm_g[1].items(), key=lambda i: i[1])
lstm_best3 = heapq.nlargest(500, lstm_g[2].items(), key=lambda i: i[1])
lstm_best4 = heapq.nlargest(500, lstm_g[3].items(), key=lambda i: i[1])

# print("CNN GUESSES\n###############################")
# for key, value in cnn_best:
#     print(key, " ", value)


# print("LSTM GUESSES\n###############################")
# for key, value in lstm_best:
#     print(key, " ", value)

with open(f"./cotraining_output/{cnn_model}/cnn_1_ngram.txt", "w", encoding="utf-8") as fp:
    for key, value in cnn_best1:
        fp.write(key + " " + str(value) + "\n")

# with open(f"./cotraining_output/{cnn_model}/cnn_2.txt", "w", encoding="utf-8") as fp:
#     for key, value in cnn_best2:
#         fp.write(key + " " + str(value) + "\n")

# with open(f"./cotraining_output/{cnn_model}/cnn_3.txt", "w", encoding="utf-8") as fp:
#     for key, value in cnn_best3:
#         fp.write(key + " " + str(value) + "\n")

# with open(f"./cotraining_output/{cnn_model}/cnn_4.txt", "w", encoding="utf-8") as fp:
#     for key, value in cnn_best4:
#         fp.write(key + " " + str(value) + "\n")
    
    
with open(f"./cotraining_output/{lstm_model}/lstm_1_ngram.txt", "w", encoding="utf-8") as fp:
    for key, value in lstm_best1:
        fp.write(key + " " + str(value) + "\n")

# with open(f"./cotraining_output/{lstm_model}/lstm_2.txt", "w", encoding="utf-8") as fp:
#     for key, value in lstm_best2:
#         fp.write(key + " " + str(value) + "\n")

# with open(f"./cotraining_output/{lstm_model}/lstm_3.txt", "w", encoding="utf-8") as fp:
#     for key, value in lstm_best3:
#         fp.write(key + " " + str(value) + "\n")

# with open(f"./cotraining_output/{lstm_model}/lstm_4.txt", "w", encoding="utf-8") as fp:
#     for key, value in lstm_best4:
#         fp.write(key + " " + str(value) + "\n")