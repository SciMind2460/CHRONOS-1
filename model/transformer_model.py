import preprocessor
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModelling,

class TextGenerationTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super(TextGenerationTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

dataset = load_dataset("wikitext", "wikitext-103-v1")

train_data = dataset['train']['text']  

tokenizer = AutoTokenizer.from_pretrained('gpt2')

test_model = transformer_model.TextGenerationTransformer(vocab_size, embed_size, hidden_size, num_heads, num_layers)


MAX_LENGTH = 128

def preprocess_function(examples):
    processed_text = [preprocess_text(text) for text in examples['text']]
    tokenized = tokenizer(
        processed_text, 
        padding='max_length', 
        truncation=True, 
        max_length=MAX_LENGTH
    )
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

def shift_tokens(examples):
    input_ids = examples['input_ids']
    labels = [ids[1:] + [tokenizer.pad_token_id] for ids in input_ids]
    examples['labels'] = labels
    return examples

tokenized_dataset = tokenized_dataset.map(shift_tokens, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir='./results',             
    num_train_epochs=10,                 
    per_device_train_batch_size=8,      
    per_device_eval_batch_size=16,       
    warmup_steps=500,                    
    weight_decay=0.01,                   
    logging_dir='./logs',                
    logging_steps=10,
    save_steps=1000,                    
    save_total_limit=2,                  
    evaluation_strategy="epoch",         
    learning_rate=0.001,                 
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  
)

trainer = Trainer(
    model=test_model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./model")
tokenizer.save_pretrained("./model")

if __name__ = "__main__":
    from transformers import pipeline
    
    generator = pipeline('text-generation', model='./model', tokenizer='./model')
    
    prompt = "We are all aware of"
    generated_text = generator(prompt, max_length=50, num_return_sequences=1())
    print(generated_text)
