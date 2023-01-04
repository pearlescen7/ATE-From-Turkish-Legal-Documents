######################################
# SECTION
# LIBRARY IMPORTS
######################################

import nltk
import os, re, random

#Install NLTK packages
nltk.download('punkt')

######################################
# SECTION
# READ DOCUMENTS
######################################

#Read files into a string
document = ""

for dirname, _, filenames in os.walk('./documents'):
  for filename in filenames:
    doc_path = os.path.join(dirname, filename)
    doc_file = open(doc_path, encoding="cp1254")
    cur_doc = doc_file.readlines()
    cur_doc = ''.join(cur_doc[7:])
    document += cur_doc


doc_length = len(document)
#Save the strings to files
with open("./bert_dataset/train_tokens.txt", "w") as f:
    f.write(document[0:int(doc_length*0.7)])

with open("./bert_dataset/valid_tokens.txt", "w") as f:
    f.write(document[int(doc_length*0.7):])

######################################
# SECTION
# FILTER DATA
######################################

#Define Turkish stopwords
tr_stopwords = set(['a', 'acaba', 'altı', 'altmış', 'ama', 'ancak', 'arada', 'artık', 'asla', 'aslında', 'ayrıca', 'az', 'bana', 'bazen', 'bazı', 'bazıları', 'belki', 'ben', 'benden', 'beni', 'benim', 'beri', 'beş', 'bile', 'bilhassa', 'bin', 'bir', 'biraz', 'birçoğu', 'birçok', 'biri', 'birisi', 'birkaç', 'birşey', 'biz', 'bizden', 'bize', 'bizi', 'bizim', 'böyle', 'böylece', 'bu', 'buna', 'bunda', 'bundan', 'bunlar', 'bunları', 'bunların', 'bunu', 'bunun', 'burada', 'bütün', 'çoğu', 'çoğunu', 'çok', 'çünkü', 'da', 'daha', 'dahi', 'dan', 'de', 'defa', 'değil', 'diğer', 'diğeri', 'diğerleri', 'diye', 'doksan', 'dokuz', 'dolayı', 'dolayısıyla', 'dört', 'e', 'edecek', 'eden', 'ederek', 'edilecek', 'ediliyor', 'edilmesi', 'ediyor', 'eğer', 'elbette', 'elli', 'en', 'etmesi', 'etti', 'ettiği', 'ettiğini', 'fakat', 'falan', 'filan', 'gene', 'gereği', 'gerek', 'gibi', 'göre', 'hala', 'halde', 'halen', 'hangi', 'hangisi', 'hani', 'hatta', 'hem', 'henüz', 'hep', 'hepsi', 'her', 'herhangi', 'herkes', 'herkese', 'herkesi', 'herkesin', 'hiç', 'hiçbir', 'hiçbiri', 'i', 'ı', 'için', 'içinde', 'iki', 'ile', 'ilgili', 'ise', 'işte', 'itibaren', 'itibariyle', 'kaç', 'kadar', 'karşın', 'kendi', 'kendilerine', 'kendine', 'kendini', 'kendisi', 'kendisine', 'kendisini', 'kez', 'ki', 'kim', 'kime', 'kimi', 'kimin', 'kimisi', 'kimse', 'kırk', 'madem', 'mi', 'mı', 'milyar', 'milyon', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nedenle', 'nerde', 'nerede', 'nereye', 'neyse', 'niçin', 'nin', 'nın', 'niye', 'nun', 'nün', 'o', 'öbür', 'olan', 'olarak', 'oldu', 'olduğu', 'olduğunu', 'olduklarını', 'olmadı', 'olmadığı', 'olmak', 'olması', 'olmayan', 'olmaz', 'olsa', 'olsun', 'olup', 'olur', 'olursa', 'oluyor', 'on', 'ön', 'ona', 'önce', 'ondan', 'onlar', 'onlara', 'onlardan', 'onları', 'onların', 'onu', 'onun', 'orada', 'öte', 'ötürü', 'otuz', 'öyle', 'oysa', 'pek', 'rağmen', 'sana', 'sanki', 'şayet', 'şekilde', 'sekiz', 'seksen', 'sen', 'senden', 'seni', 'senin', 'şey', 'şeyden', 'şeye', 'şeyi', 'şeyler', 'şimdi', 'siz', 'sizden', 'size', 'sizi', 'sizin', 'sonra', 'şöyle', 'şu', 'şuna', 'şunları', 'şunu', 'ta', 'tabii', 'tam', 'tamam', 'tamamen', 'tarafından', 'trilyon', 'tüm', 'tümü', 'u', 'ü', 'üç', 'un', 'ün', 'üzere', 'var', 'vardı', 've', 'veya', 'ya', 'yani', 'yapacak', 'yapılan', 'yapılması', 'yapıyor', 'yapmak', 'yaptı', 'yaptığı', 'yaptığını', 'yaptıkları', 'ye', 'yedi', 'yerine', 'yetmiş', 'yi', 'yı', 'yine', 'yirmi', 'yoksa', 'yu', 'yüz', 'zaten', 'zira', 'amma', 'anca', 'emme', 'gah', 'hakeza', 'halbuki', 'hele', 'hoş', 'imdi', 'ister', 'kah', 'keşke', 'keza', 'kezalik', 'lakin', 'mademki', 'mamafih', 'meğer', 'meğerki', 'meğerse', 'netekim', 'nitekim', 'oysaki', 'velev', 'velhasıl', 'velhasılıkelam', 'veyahut', 'yahut', 'yalnız', 'yok', 'acep', 'açıkça', 'açıkçası', 'adamakıllı', 'adeta', 'bilcümle', 'binaen', 'binaenaleyh', 'birazdan', 'birden', 'birdenbire', 'birice', 'birlikte', 'bitevi', 'biteviye', 'bittabi', 'bizatihi', 'bizce', 'bizcileyin', 'bizzat', 'boşuna', 'böylecene', 'böylelikle', 'böylemesine', 'böylesine', 'buracıkta', 'buradan', 'büsbütün', 'çabuk', 'çabukça', 'çeşitli', 'çoğun', 'çoğunca', 'çoğunlukla', 'çokça', 'çokluk', 'çoklukla', 'cuk', 'dahil', 'dahilen', 'daima', 'demin', 'demincek', 'deminden', 'derakap', 'derhal', 'derken', 'elbet', 'enikonu', 'epey', 'epeyce', 'epeyi', 'esasen', 'esnasında', 'etraflı', 'etraflıca', 'evleviyetle', 'evvel', 'evvela', 'evvelce', 'evvelden', 'evvelemirde', 'evveli', 'gayet', 'gayetle', 'gayri', 'gayrı', 'geçende', 'geçenlerde', 'gerçi', 'gibilerden', 'gibisinden', 'gine', 'halihazırda', 'haliyle', 'handiyse', 'hasılı', 'hulasaten', 'iken', 'illa', 'illaki', 'itibarıyla', 'iyice', 'iyicene', 'kala', 'kısaca', 'külliyen', 'lütfen', 'nasılsa', 'nazaran', 'nedeniyle', 'nedense', 'nerden', 'nerdeyse', 'nereden', 'neredeyse', 'neye', 'neyi', 'nice', 'nihayet', 'nihayetinde', 'oldukça', 'onca', 'önceden', 'önceleri', 'öncelikle', 'onculayın', 'oracık', 'oracıkta', 'oradan', 'oranca', 'oranla', 'oraya', 'öylece', 'öylelikle', 'öylemesine', 'pekala', 'pekçe', 'peki', 'peyderpey', 'sadece', 'sahi', 'sahiden', 'sonradan', 'sonraları', 'sonunda', 'şuncacık', 'şuracıkta', 'tamamıyla', 'tek', 'vasıtasıyla', 'yakinen', 'yakında', 'yakından', 'yakınlarda', 'yalnızca', 'yeniden', 'yenilerde', 'yoluyla', 'yüzünden', 'zati', 'ait', 'bari', 'değin', 'dek', 'denli', 'doğru', 'gelgelelim', 'gırla', 'hasebiyle', 'ila', 'ilen', 'indinde', 'inen', 'kaffesi', 'kelli', 'Leh', 'maada', 'mebni', 'naşi', 'zarfında', 'başkası', 'beriki', 'birbiri', 'birileri', 'birkaçı', 'bizimki', 'burası', 'çokları', 'çoklarınca', 'cümlesi', 'filanca', 'iş', 'kaçı', 'kaynak', 'kimsecik', 'kimsecikler', 'nere', 'neresi', 'öbürkü', 'öbürü', 'onda', 'öteki', 'ötekisi', 'öz', 'şunda', 'şundan', 'şunlar', 'şunun', 'şura', 'şuracık', 'şurası'])

#Define punctuation characters
punctuation = list("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~’”“")


#Tokenize the document
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

all_tokens = digit_filtered_tokens
print("token count: ", len(all_tokens))
# print(all_tokens[0:100])


######################################
# SECTION
# GENERATE N-GRAMS
######################################

from nltk import ngrams

n_max = 4

def find_all_ngrams(words, n):
  ngram_list = []
  for n in range(n_max):
      all_ngrams = ngrams(words, n+1)

      for grams in all_ngrams:
          ngram_list.append(grams)

  return ngram_list

def find_stopped_ngrams(words, n_max):
  candidates = []

  last_idx = 0
  stopped = True
  for i, word in enumerate(words):
    if word in tr_stopwords:
      if not stopped:
        candidates.extend(find_all_ngrams(words[last_idx:i], n_max))
        stopped = True
      last_idx = i+1
    else:
      stopped = False

  if(last_idx != len(words)-1):
    candidates.extend(find_all_ngrams(words[last_idx:], n_max))

  return candidates

# ngram_list = find_all_ngram(sentence, n_max)

candidates = find_stopped_ngrams(all_tokens, n_max)

for i, ngram in enumerate(candidates):
  candidates[i] = " ".join(candidates[i])


######################################
# SECTION
# SAVE N-GRAMS
######################################

# with open("./drive/MyDrive/candidates_latest.txt", "w") as f:
#   for candidate in candidates:
#     f.write(candidate+"\n")

######################################
# SECTION
# LOAD N-GRAMS
######################################

# candidates = []

# with open("./drive/MyDrive/candidates_latest.txt", "r") as f:
#   for line in f.readlines():
#     candidates.append(line.rstrip().lstrip())

# print(candidates[0:100])
# print(len(candidates))


unlabeled_data = candidates
random.shuffle(unlabeled_data)
