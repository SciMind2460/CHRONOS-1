import transformer_model
import preprocessor
import datasets
import torch
import torch.nn as nn

dataset = datasets.load_dataset("wikitext", "wikitext-103-v1")

train_data = dataset['train'] 

tokenized_data = preprocess_text(train_data)

input_data = tokenized_data[:-1]
target_data = tokenized_data[1:]

input_data = torch.tensor(input_data, dtype=torch.float32)
target_data = torch.tensor(target_data, dtype=torch.float32)

vocab_size = 20000
embed_size = 256
hidden_size = 512
num_layers = 4
num_heads = 8

test_model = transformer_model.TextGenerationTransformer(vocab_size, embed_size, hidden_size, num_heads, num_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

sequence = sequence.unsqueeze(-1)

epochs = 100
for epoch in range(epochs):
  optimizer.zero_grad()
  output = test_model(input_data)
  loss = criterion(output, target_data.unsqueeze(-1))
  loss.backward()
  optimizer.step()
  
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

target_data = target.unsqueeze(-1)
model.eval()
with torch.no_grad():
  prediction = model(target_data)
  print(f'Predicted next value: {prediction.item():.4f}')
  
