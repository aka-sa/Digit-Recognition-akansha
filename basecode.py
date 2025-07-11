# pip install transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

# Define a custom LLM class
class LLM:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, user_input):
        messages = [{"role": "user", "content": user_input}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.92,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
device = "cpu"  # Use "cuda" for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Initialize the LLM
llm = llm_get(model, tokenizer, device)

# Test the LLM
test_input = "What is the capital of France?"
print(f"Test input: {test_input}")
print(f"Response: {llm.generate_response(test_input)}")


from transformers import AutoTokenizer
import transformers 
import torch
model = "HuggingFaceTB/SmolLM-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    repetition_penalty=1.5,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

llm = llm_get(model, tokenizer, device)


# import os, torch
# torch.cuda.empty_cache()

# os.system("clear");
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
# from langchain_huggingface.llms import HuggingFacePipeline
# from accelerate import Accelerator

# accelerator = Accelerator()

# model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
# #model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     offload_buffers=True,
#     use_cache=True,
# )
# model = accelerator.prepare(model)

# def get_llm(temperature=0.0, use_streamer=False, max_new_tokens=200):
#     if use_streamer:
#         text_pipeline = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             repetition_penalty=1.15,
#             streamer=streamer
#         )
#     else:
#         text_pipeline = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             repetition_penalty=1.15
#         )
    
#     return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": temperature})

# print("Model running on:", accelerator.device)

# llm = get_llm(temperature=0.7,max_new_tokens=400, use_streamer=True)

