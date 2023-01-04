with open("./bert_dataset/all_documents_analyzed_lower.txt", "r", encoding="utf-8") as fp:
    lines = fp.readlines()

#(POS_TAG, lemmatized_word)
token_tuples = []
for line in lines:
    line_ = line.strip().split("\t")
    if len(line_)==2:
        if (len(line_[0])>=2 and line_[1]!="Punc") or (line_[1]=="Punc"):
            token_tuples.append((line_[1], line_[0]))

    elif len(line_)==3:
        if (len(line_[0])>=2 and line_[1]!="Punc") or (line_[1]=="Punc"):
            token_tuples.append((line_[1], line_[2]))
    else:
        continue
        # print(line_)

# print(token_tuples[:100])

# skip until adj or noun
# if adj
#   go back to first state if not adj or noun
#   if adj again go to if adj
#   if noun start looking for nouns

init_s = 0
adj_s  = 1
noun_s = 2

parser_state = init_s
start_idx = 0
end_idx = 0
adj_end_idx = 0

chunk_idxs = []
done = False

for idx, (tag, word) in enumerate(token_tuples):
    if done:
        break
    
    elif parser_state == init_s:

        if tag == "Adj":
            start_idx = idx
            parser_state = adj_s

        if tag == "Noun":
            start_idx = idx
            adj_end_idx = idx
            parser_state = noun_s

    elif parser_state == adj_s:
        if tag == "Adj":
            continue
        elif tag == "Noun":
            parser_state = noun_s
            adj_end_idx = idx
        else:
            parser_state = init_s


    elif parser_state == noun_s:
        if tag == "Noun":
            continue
        else:
            end_idx = idx
            parser_state = init_s
            # chunk_idxs.append((start_idx, end_idx))

            # adj1 adj2 adj3 adj4
            # => adj4, adj3 adj4, adj2 adj3 adj4 ...
            # does the same for the words but starts from the left, goes to right
            for i in range(adj_end_idx-start_idx+1):
                for j in range(end_idx-adj_end_idx):
                    chunk_idxs.append((adj_end_idx-i, end_idx-j))
                    # if len(chunk_idxs)==1e7:
                    #     done = True

            #also add ngrams of nouns since they are valid
            for i in range(4):
                for j in range(adj_end_idx, end_idx):
                    if j+i < end_idx:
                        chunk_idxs.append((j, j+i+1))


# noun_phrases = []
# for (start_idx, end_idx) in chunk_idxs:
#     noun_phrase = token_tuples[start_idx:end_idx]
#     noun_phrases.append(noun_phrase)
#     print(noun_phrase)

with open("./bert_dataset/pos_candidates_with_tags.txt", "w", encoding="utf-8") as fp:
    for (start_idx, end_idx) in chunk_idxs:
        noun_phrase = token_tuples[start_idx:end_idx]
        phrase_str = ""
        tag_str = ""
        for (tag, word) in noun_phrase:
            tag_str += tag + "_"
            phrase_str += word + "_"

        fp.write(tag_str + " " + phrase_str + "\n")

used = set()

with open("./bert_dataset/pos_candidates_without_tags.txt", "w", encoding="utf-8") as fp:
    for (start_idx, end_idx) in chunk_idxs:
        noun_phrase = token_tuples[start_idx:end_idx]
        phrase_str = ""
        tag_str = ""
        for (tag, word) in noun_phrase:
            phrase_str += word + " "

        if phrase_str not in used:
            fp.write(phrase_str + "\n")
            used.add(phrase_str)