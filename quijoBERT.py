

from transformers import AutoTokenizer, AutoModelForMaskedLM,  RobertaConfig , RobertaTokenizer,RobertaForMaskedLM, DataCollatorForLanguageModeling, LineByLineTextDataset, Trainer, TrainingArguments


from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
from torchinfo import summary


import os

paths = [str(x) for x in Path(".").glob("**/el_*.txt")]
print(paths)
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2,
special_tokens=[
"<s>",
"<pad>",
"</s>",
"<unk>",
"<mask>",
])


dir_path = os.getcwd()
token_dir = os.path.join(dir_path, 'QuijoBERT')

if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model('QuijoBERT')

tokenizer = ByteLevelBPETokenizer(
"./QuijoBERT/vocab.json",
"./QuijoBERT/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
("</s>", tokenizer.token_to_id("</s>")),
("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)



config = RobertaConfig(
  vocab_size=52_000,
  max_position_embeddings=514,
  num_attention_heads=12,
  num_hidden_layers=6,
  type_vocab_size=1,
  )

"""# Step 8: Re-creating the Tokenizer in Transformers"""

tokenizer = RobertaTokenizer.from_pretrained("./QuijoBERT", max_length=512)

#Initializing a Model 

model = RobertaForMaskedLM(config=config)
#In case we want to recover the after a crash
#model = RobertaForMaskedLM.from_pretrained("./QuijoBERT/Checkpoint-xxxxx")


#Tensorflow
print(model)
#Pytorch
summary(model)


dataset = LineByLineTextDataset(
  tokenizer=tokenizer,
  file_path="./el_quijote.txt",
  block_size=128,
  )


#Defining a Data Collator

data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Initializing the Trainer Object
training_args = TrainingArguments(
  output_dir="./QuijoBERT",
  overwrite_output_dir=True,
  num_train_epochs=1,
  per_device_train_batch_size=64,
  save_steps=1000,
  save_total_limit=2,
  )
trainer = Trainer(
  model=model,
  args=training_args,
  data_collator=data_collator,
  train_dataset=dataset,
)


#Training the Model
print('aqui')
trainer.train()
trainer.save_model("./QuijoBERT")

#Saving the Final Model(+tokenizer + config) to disk
trainer.save_model("./QuijoBERT")

