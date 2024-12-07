from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

# formatted_chat = tokenizer.apply_chat_template(
#     chat, tokenize=False, return_dict=False, continue_final_message=True
# )

formatted_chat = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=False, continue_final_message=True
)
print(formatted_chat)
