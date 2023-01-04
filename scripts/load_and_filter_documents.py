import os, re, nltk

#Read files into a string
document = ""

count = 0

for dirname, _, filenames in os.walk('./yargitay'):
  for filename in filenames:
    doc_path = os.path.join(dirname, filename)
    doc_file = open(doc_path, encoding="cp1254")
    cur_doc = doc_file.readlines()
    cur_doc = ''.join(cur_doc[7:])
    document += cur_doc
    count += 1

    if count % 500 == 0:
      print(count, " documents read")

doc_length = len(document)
print(doc_length)

punctuation = list("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~’”“")

tokens = nltk.word_tokenize(document.lower())

length_filtered_tokens = []
for token in tokens:
  if len(token) > 2:
    length_filtered_tokens.append(token)
# #Filter stopwords
# stop_filtered_tokens = []

# for token in tokens:
#     if token not in tr_stopwords:
#         stop_filtered_tokens.append(token)

#Filter punctuation
punct_filtered_tokens = []

for token in length_filtered_tokens:
# for token in stop_filtered_tokens:
    found = False
    for punct in punctuation:
        if punct in token:
            found = True
    if not found:
        punct_filtered_tokens.append(token)

#Filter digits
digit_filtered_tokens = []

for token in punct_filtered_tokens:
    if not bool(re.search(r'\d+', token)):
        digit_filtered_tokens.append(token)


all_tokens = []
for token in digit_filtered_tokens:
  if len(token) > 2:
    all_tokens.append(token)

#Save the strings to files
with open("./bert_dataset/all_documents_filtered.txt", "w", encoding="utf-8") as f:
    for token in all_tokens:
        f.write(token + " ")