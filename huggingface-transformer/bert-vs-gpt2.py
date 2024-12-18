from transformers import AutoModel, AutoTokenizer
import torch

# Define the input text
text = "What is Huggingface Transformers?"

# ---------------------
# BERT Model Processing
# ---------------------
# Load BERT model and tokenizer
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input text
bert_encoded_input = bert_tokenizer(text, return_tensors='pt')

# Get the model output
bert_output = bert_model(**bert_encoded_input)

# Extract the last hidden state
bert_last_hidden_state = bert_output.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

# Extract the [CLS] token embedding (first token)
bert_cls_embedding = bert_last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

# Convert to NumPy for easier viewing
bert_cls_embedding_np = bert_cls_embedding.detach().numpy()

print("BERT [CLS] Embedding:")
print(bert_cls_embedding_np)
print("Shape:", bert_cls_embedding_np.shape)  # Should be (1, 768) for bert-base-uncased

# ---------------------
# GPT-2 Model Processing
# ---------------------
# Load GPT-2 model and tokenizer
gpt_model = AutoModel.from_pretrained('gpt2')
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Tokenize the input text
gpt_encoded_input = gpt_tokenizer(text, return_tensors='pt')

# Get the model output
gpt_output = gpt_model(**gpt_encoded_input)

# Extract the last hidden state
gpt_last_hidden_state = gpt_output.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

# Extract the embedding of the last token
gpt_last_token_embedding = gpt_last_hidden_state[:, -1, :]  # Shape: (batch_size, hidden_size)

# Convert to NumPy for easier viewing
gpt_last_token_embedding_np = gpt_last_token_embedding.detach().numpy()

print("\nGPT-2 Last Token Embedding:")
print(gpt_last_token_embedding_np)
print("Shape:", gpt_last_token_embedding_np.shape)  # Should be (1, 768) for gpt2

# ---------------------
# Optional: Text Generation with GPT-2
# ---------------------
from transformers import AutoModelForCausalLM

# Load GPT-2 for text generation
gpt2_gen_model = AutoModelForCausalLM.from_pretrained('gpt2')
gpt2_gen_tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Encode the input text
input_ids = gpt2_gen_tokenizer.encode(text, return_tensors='pt')

# Generate text
generated_output = gpt2_gen_model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

# Decode the generated text
generated_text = gpt2_gen_tokenizer.decode(generated_output[0], skip_special_tokens=True)

print("\nGPT-2 Generated Text:")
print(generated_text)
