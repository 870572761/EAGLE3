from eagle.model.ea_model import EaModel_lpf
from fastchat.model import get_conversation_template
import torch
base_model_path="meta-llama/Meta-Llama-3-8B-Instruct"
EAGLE_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B"
lpfrog_model_path ="/data/lei/lpfmodel"
# EAGLE_model_path = lpfrog_model_path
model = EaModel_lpf.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    lpfrog_model_path = lpfrog_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-2,
    depth = 3,
    top_k = 10,
)
model.generate_type = "topmatch"
model.ea_layer.method = "topmatch"
# model.generate_type = None
model.eval()
your_message="Hello"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=60)
output=model.tokenizer.decode(output_ids[0])
print(output)