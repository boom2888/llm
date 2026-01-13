from datasets import load_dataset
import torch
block_size = 128      # context window length
batch_size = 32
EOT_TOKEN_ID=50256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
from gpt import GPT

enc = tiktoken.get_encoding("gpt2")

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(f'{ROOT_DIR}/train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap(f'{ROOT_DIR}/validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
def calc_loss_loader(split, model, device, num_batches=5):
    """
    Calculate average loss over num_batches using get_batch function

    Args:
        split: 'train' or 'validation'
        model: the model to evaluate
        device: device to run on
        num_batches: number of batches to average over
    """
    total_loss = 0.

    for i in range(num_batches):
        input_batch, target_batch = get_batch(split)
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches
def evaluate_model(model, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader('train', model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader('validation', model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
# def generate_and_print_sample(model, tokenizer, device, start_context):
#     model.eval()
#     context_size = model.pos_emb.weight.shape[0]
#     encoded = text_to_token_ids(start_context, tokenizer).to(device)
#     with torch.no_grad():
#         token_ids = generate_text_simple(
#             model=model, idx=encoded,
#             max_new_tokens=50, context_size=context_size
#         )
#     decoded_text = token_ids_to_text(token_ids, tokenizer)
#     print(decoded_text.replace("\n", " "))  # Compact print format
#     model.train()
def train_model_simple(model, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       steps_per_epoch=1000):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for step in range(steps_per_epoch):
            # Use the get_batch function instead of DataLoader
            input_batch, target_batch = get_batch('train')

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample__(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen
def generate_and_print_sample__(model, tokenizer, device, start_context):
    """Generate and print a sample text"""
    model.eval()
    context_size = 128
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = model.generate(
            idx=encoded,
            max_new_tokens=300,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
import time
start_time = time.time()

torch.manual_seed(123)
model = GPT()
model.to(device)
# Ensure all parameters require gradients
model.train()

# Verify gradients are enabled (optional but good for debugging)
print("Checking if model parameters require grad...")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"WARNING: {name} does not require grad!")
        param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model, optimizer, device,
    num_epochs=num_epochs, eval_freq=100, eval_iter=5,
    start_context="a boy got a ball", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")