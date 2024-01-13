import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.moe_lora import MistralMoLoraLayer, MoLoRAArgs, replace_layers


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    args = MoLoRAArgs(num_experts=8, num_experts_per_tok=2, lora_rank=8, lora_alpha=0.1)
    mistral_mlp_type = type(model.model.layers[0].mlp)

    replace_layers(model, mistral_mlp_type, MistralMoLoraLayer, args)

    model.cuda()
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    print(text_generator("The meaning of life is", max_length=50, do_sample=True, temperature=0.9))