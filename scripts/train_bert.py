######################################
# SECTION
# SETUP BERT MODEL
######################################

#Import BERT
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
model = BertForMaskedLM.from_pretrained("dbmdz/bert-base-turkish-uncased")
# model = BertForMaskedLM.from_pretrained("./drive/MyDrive/bert-turkish-legal-pretrained")


# tokenizer = BertTokenizer.from_pretrained("./drive/MyDrive/bert-turkish-legal-pretrained-output/checkpoint-6500")
# model = BertForMaskedLM.from_pretrained("./drive/MyDrive/bert-turkish-legal-pretrained-output/checkpoint-6500")

model.cuda()
print(model.device)

######################################
# SECTION
# CREATE DATASET
######################################

from transformers import TextDataset, TrainingArguments, DataCollatorForLanguageModeling, Trainer

#Create train and validation splits
train_tokens = TextDataset(file_path="./bert_dataset/train_tokens_yargitay.txt", tokenizer=tokenizer, block_size=128)
valid_tokens = TextDataset(file_path="./bert_dataset/valid_tokens_yargitay.txt", tokenizer=tokenizer, block_size=128)


######################################
# SECTION
# SETUP TRAINING ARGUMENTS AND TRAINER
######################################

#Set training configuration
training_args = TrainingArguments(
    output_dir="./bert_output/",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_dir="./bert_log/",
    logging_steps=100,
    save_steps = 5000
)

#Create a data collator to do the masking
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

#Create trainer to train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokens,
    eval_dataset=valid_tokens,
    data_collator=data_collator
)

######################################
# SECTION
# TRAIN AND SAVE THE MODEL
######################################

#Train and save the model
# trainer.train("./drive/MyDrive/bert-turkish-legal-pretrained-output/checkpoint-2500")
trainer.train("./bert_output/checkpoint-13000")
trainer.save_model("./bert-turkish-legal-pretrained")