######################################
# SECTION
# LIBRARY IMPORTS
######################################

import nltk
import os, re, random

#Install NLTK packages
# nltk.download('punkt')

######################################
# SECTION
# READ DOCUMENTS
######################################

#Read files into a string
document = ""

count = 0
for dirname, _, filenames in os.walk('./yargitay'):
    for filename in filenames:
        count += 1
        doc_path = os.path.join(dirname, filename)
        try:
            doc_file = open(doc_path, encoding="cp1254")
            cur_doc = doc_file.readlines()
        except:
            try:
                doc_file = open(doc_path, encoding="utf-8")
                cur_doc = doc_file.readlines()
            except:
                print("Error: ", filename)
                cur_doc = None

        if cur_doc:
            cur_doc = ''.join(cur_doc[7:])
            document += cur_doc

        if count % 500 == 0:
            print(count, " documents read")


#Define punctuation characters
punctuation = list("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~’”“")


#Tokenize the document
tokens = nltk.word_tokenize(document.lower())

#Filter punctuation
punct_filtered_tokens = []

for token in tokens:
    found = False
    for punct in punctuation:
        if punct in token:
            found = True
    if not found:
        punct_filtered_tokens.append(token)

print("punct filtered token count: ", len(punct_filtered_tokens))

#Filter digits
digit_filtered_tokens = []

for token in punct_filtered_tokens:
    if not bool(re.search(r'\d+', token)):
        digit_filtered_tokens.append(token)

print("digit filtered token count: ", len(digit_filtered_tokens))


length_filtered_tokens = []
for token in digit_filtered_tokens:
  if len(token) > 2:
    length_filtered_tokens.append(token)

print("length filtered token count: ", len(length_filtered_tokens))

all_tokens = length_filtered_tokens
print("token count: ", len(all_tokens))


doc_length = len(all_tokens)
#Save the strings to files
with open("./bert_dataset/train_tokens_yargitay.txt", "w") as f:
    f.write(document[0:int(doc_length*0.7)] + " ")

with open("./bert_dataset/valid_tokens_yargitay.txt", "w") as f:
    f.write(document[int(doc_length*0.7):] + " ")