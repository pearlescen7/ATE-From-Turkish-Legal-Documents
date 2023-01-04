word_dict = {}

with open("./cotraining_output/all_guessed_terms_fixed.txt", "r", encoding="utf-8") as fp:
    for line in fp.readlines():
        guess = line.rstrip().split(" ")
        word_dict[" ".join(guess[:-1])] = guess[-1]

print(len(word_dict.keys()))

n = []
p = []

for word in word_dict.keys():
    if word_dict[word] == 'n': 
        n.append(word)
    else:
        p.append(word)

p.sort()
n.sort()

with open("./cotraining_output/n.txt", 'w', encoding="utf-8") as fp:
    for word in n:
        fp.write(word + "\n")

with open("./cotraining_output/p.txt", 'w', encoding="utf-8") as fp:
    for word in p:
        fp.write(word + "\n")
        

