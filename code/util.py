from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# SOURCE: https://huggingface.co/learn/llm-course/en/chapter7/3
def fill_mask(prompt, targets):
    inputs = tokenizer(prompt, return_tensors="pt")
    token_logits = model(**inputs).logits

    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1].item()
    mask_token_logits = token_logits[0, mask_token_index, :]

    target_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)[0]) for word in targets]
    log_probs = F.log_softmax(mask_token_logits, dim=-1)

    out = []
    for word, token_id in zip(targets, target_token_ids):
        prob = math.exp(log_probs[token_id].item())
        out.append((word, round(prob,4)))

    return out