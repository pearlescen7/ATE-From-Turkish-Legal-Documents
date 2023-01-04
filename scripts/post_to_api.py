# importing the requests library
# import requests
import pipeline_caller, os
  
# API_ENDPOINT = "http://tools.nlp.itu.edu.tr/SimpleApi"
API_KEY = "oqyHUmEvyHEHIyqSVuGkAW7mXJgbd8pn"
  
# with open("./bert_dataset/all_documents.txt", "r", encoding="utf-8") as fp:
#     docs = fp.readlines()
#     docs = "".join(docs)

count = 0

for dirname, _, filenames in os.walk('./bert_dataset/splitted/'):
  for filename in filenames:
    if filename > "all_documents_21004.txt":
      doc_path = os.path.join(dirname, filename)
      doc_file = open(doc_path, 'r', encoding="utf-8")
      cur_doc = doc_file.readlines()
      cur_doc = ''.join(cur_doc).lower()
      count += 1

      caller = pipeline_caller.PipelineCaller()
      analyzed_text = caller.call("morphanalyzer", cur_doc, API_KEY)

      with open(f"./bert_dataset/analyzed_lower/{filename}", "w", encoding="utf-8") as fp:
          fp.write(analyzed_text)

      # if count % 500 == 0:
      #     print(count, " documents read")

# data to be sent to api
# data = {'tool'  : 'morphanalyzer',
#         'input' : docs,
#         'token' : API_KEY}
  
# sending post request and saving response as response object
# r = requests.post(url=API_ENDPOINT, data=data)
  
# print(r)

# # extracting response text 
# analyzed_text = r.text

# print(docs)

# caller = pipeline_caller.PipelineCaller()
# analyzed_text = caller.call("pipelineSSMorph", docs, API_KEY)

# with open("./bert_dataset/morphanalyzed.txt", "w", encoding="utf-8") as fp:
#     fp.write(analyzed_text)