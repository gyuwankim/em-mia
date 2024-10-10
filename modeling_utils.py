import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM
from huggingface_hub import list_repo_refs
from hf_olmo import OLMoForCausalLM


def load_model(name, use_float16=True, olmo_step=None):
    model_class = MambaForCausalLM if "mamba" in name else AutoModelForCausalLM
    revision = "main"

    if "OLMo" in name:
        model_class = OLMoForCausalLM
        branches = [b.name for b in list_repo_refs(name).branches]
        step2tokens = {int(b.split('-')[0][4:]): int(b.split('-')[1][6:-1]) for b in branches if b != "main"}        
        revision = f"step{olmo_step}-tokens{step2tokens[olmo_step]}B"

    model = model_class.from_pretrained(
        name, return_dict=True, trust_remote_code=True,
        torch_dtype=torch.float16 if use_float16 else torch.float32,
        device_map="auto",
        revision=revision,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def infer(model, tokenizer, target, prefix=None):
    def tokenize(texts, padding_side):
        tokenizer.padding_side = padding_side
        encodings = tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)
        return input_ids, attention_mask

    target_input_ids, _ = tokenize(target, "right")
    num_target, _ = target_input_ids.size()
    assert num_target == 1, "currently only support with a single target text"
    if prefix is not None:
        prefix_input_ids, prefix_attention_mask = tokenize(prefix, "left")
        num_prefix, prefix_len = prefix_input_ids.size()
        target_input_ids = target_input_ids.repeat(num_prefix, 1)
        input_ids = torch.cat((prefix_input_ids, target_input_ids), dim=1)
        target_attention_mask = torch.ones_like(target_input_ids)
        attention_mask = torch.cat((prefix_attention_mask, target_attention_mask), dim=1)
    else:
        num_prefix, prefix_len = 1, 0   # null prefix
        input_ids = target_input_ids
        attention_mask = None

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs[0]
    logits = logits[:, prefix_len:-1].contiguous().view(num_prefix, -1, logits.shape[-1]).contiguous()
    labels = target_input_ids[:, 1:].contiguous().view(num_prefix, -1).contiguous()
    ll = -F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none").view(num_prefix, -1).mean(-1)
    if prefix is None:
        return ll[0].item(), logits[0], labels[0]
    else:
        return ll.numpy(force=True), logits, labels
