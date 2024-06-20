from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-2.7B"
model_save_path = "./gpt-neo-2.7B"

# Load and save the model and tokenizer locally
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_save_path)

model = GPTNeoForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_save_path)
print("Model Saved")
