import re 
import inflect 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Uses the Inflect library to convert text to numbers
p = inflect.engine()
def convert_number_to_text(text):
    temp_str = text.split() # String for holding processed text
    new_string = [] # List variable backer
    # The main converter
    for word in temp_str:
        # If the word is a digit
        if word.isdigit():
            try:
                temp = p.number_to_words(word)
                new_string.append(temp)
            # Exception handling for large numbers
            except Exception as e:
                new_string.append(word)
        # Normal course of words
        else:
            new_string.append(word)
    # Returns the value
    temp_str = ' '.join(new_string)
    return temp_str

# Main preprocessor 
def preprocess_text(dataset):
    dataset = dataset.lower()
    dataset = re.sub(r'[^\w\s\'\-]', '', dataset)
    dataset = convert_number_to_text(dataset)
    # Returns the list of words
    return tokenizer.encode(dataset)

def preprocess_function(examples, tokenizer):
    processed_text = [preprocessor.preprocess_text(text) for text in examples['text']]
    tokenized = tokenizer(
        processed_text, 
        padding='max_length', 
        truncation=True, 
        max_length=MAX_LENGTH
    )
    return tokenized

def shift_tokens(examples, tokenizer):
    input_ids = examples['input_ids']
    labels = [ids[1:] + [tokenizer.pad_token_id] for ids in input_ids]
    examples['labels'] = labels
    return examples

# Testing! Testing!
if __name__ == "__main__":
    preprocessor_sample = "Sometimes I wonder if physics will ever be complete. You shouldn't ever be sure. There are more than 1000 reasons why it won't, any time soon. Sorry, more than 100000000000000000000000000000000."
    process_sample = preprocess_text(preprocessor_sample)
    print(process_sample)
