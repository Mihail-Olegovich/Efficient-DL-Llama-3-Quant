import math
import time
from typing import Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def prepare_wikitext2_dataset(tokenizer, block_size: int = 2048):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    eval_dataset = dataset["validation"]

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=False)

    tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)
    lm_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    return lm_dataset


def create_dataloader(dataset, batch_size: int = 1):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True)


def evaluate_perplexity(model: torch.nn.Module, dataloader: DataLoader, device: torch.device | str) -> Dict[str, Any]:
    if isinstance(device, str):
        device = torch.device(device)

    start_time = time.time()
    num_tokens = 0
    loss_sum = 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            loss = outputs.loss
            num_tokens += batch["labels"].numel()
            loss_sum += loss.item()
            count += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    mean_loss = loss_sum / max(count, 1)
    ppl = math.exp(mean_loss)
    tps = num_tokens / elapsed if elapsed > 0 else float("nan")

    return {
        "perplexity": ppl,
        "eval_time_s": elapsed,
        "num_tokens": num_tokens,
        "tokens_per_sec": tps,
    }


