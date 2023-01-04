import os

with open("./bert_dataset/all_documents.txt", "r", encoding="utf-8") as fp:
    lines = fp.readlines()

num_lines = len(lines)

batch_lines = num_lines//100

for i in range(0, num_lines, batch_lines):
    with open("./bert_dataset/splitted/all_documents_"+str(i)+".txt", "w", encoding="utf-8") as fp:
        for j in range(i, min(i+batch_lines, num_lines)):
            fp.write(lines[j])