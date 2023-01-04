import os

document = ""
for dirname, _, filenames in os.walk('./bert_dataset/analyzed_lower/'):
    for filename in filenames:
        doc_path = os.path.join(dirname, filename)
        doc_file = open(doc_path, 'r', encoding="utf-8")
        cur_doc = doc_file.readlines()
        cur_doc = ''.join(cur_doc)
        document += cur_doc

with open("./bert_dataset/all_documents_analyzed_lower.txt", "w", encoding="utf-8") as fp:
    fp.write(document)