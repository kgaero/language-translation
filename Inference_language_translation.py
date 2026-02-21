import os, sys

base = "/kaggle/input/datasets"
print("Level 1:", os.listdir(base))

found = None

for user_folder in os.listdir(base):
    user_path = f"{base}/{user_folder}"
    if not os.path.isdir(user_path):
        continue

    # List datasets under your username folder (kgaero)
    for ds in os.listdir(user_path):
        ds_path = f"{user_path}/{ds}"
        if not os.path.isdir(ds_path):
            continue

        # Look for the project folder
        if "language-translation" in os.listdir(ds_path):
            found = f"{ds_path}/language-translation"
            print("FOUND:", found)
            print("Files:", os.listdir(found))
            break

    if found:
        break

assert found, "Did not find language-translation folder anywhere under /kaggle/input/datasets"
sys.path.insert(0, found)

import config
print("Imported config from:", config.__file__)


PROJECT_DIR = found  # this is .../language-translation
WEIGHTS_DIR = f"{PROJECT_DIR}/opus_books_weights"
print("Weights dir:", WEIGHTS_DIR)
print("Weights files:", sorted([f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt")])[:10])


from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode
import altair as alt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# model_filename = get_weights_file_path(config, f"19")
# state= torch.load(model_filename)
# model.load_state_dict(state['model_state_dict'])


ckpts = sorted([f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt")])
print("All checkpoints:", ckpts)

model_filename = os.path.join(WEIGHTS_DIR, ckpts[-1])
print("Auto-loading:", model_filename)

state = torch.load(model_filename, map_location=device)
model.load_state_dict(state["model_state_dict"])


run_validation(
    model=model,
    validation_ds=val_dataloader,     # or val_dataloader depending on your code
    tokenizer_tgt=tokenizer_tgt,
    max_len=config["seq_len"],
    device=device,
    print_msg=lambda msg: print(msg),
    global_step=0,
    writer=None,
    num_examples=10,
)



def load_next_batch():
    batch = next(iter(val_dataloader))
    encoder_input = batch['encoder_input'].to(device)
    encoder_mask = batch['encoder_mask'].to(device)
    decoder_input = batch['decoder_input'].to(device)
    decoder_mask = batch['decoder_mask'].to(device)

    vocab_src = tokenizer_src
    vocab_tgt = tokenizer_tgt
    encoder_input_tokens = [vocab_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [vocab_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    model_out = greedy_decode(model, encoder_input, encoder_mask, vocab_tgt, config['seq_len'], device)
    return batch, encoder_input_tokens, decoder_input_tokens



def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def get_attn_map(attn_type: str, layer: int, head: int):
    if attn_type == "encoder":
        attn = model.encoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "decoder":
        attn = model.decoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "encoder-decoder":
        attn = model.decoder.layers[layer].cross_attention_block.attention_scores
    return attn[0, head].data

def attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(
        get_attn_map(attn_type, layer, head),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def get_all_attention_maps(attn_type: str, layers: list[int], heads: list[int], row_tokens: list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)




batch, encoder_input_tokens, decoder_input_tokens = load_next_batch()
print(f'Source: {batch["src_text"][0]}')
print(f'Target: {batch["tgt_text"][0]}')
sentence_len = encoder_input_tokens.index("[PAD]")



layers = [0, 1, 2]
heads = [0, 1, 2, 3, 4, 5, 6, 7]

# Encoder Self-Attention
get_all_attention_maps("encoder", layers, heads, encoder_input_tokens, encoder_input_tokens, min(20, sentence_len))


# Encoder Self-Attention
get_all_attention_maps("decoder", layers, heads, decoder_input_tokens, decoder_input_tokens, min(20, sentence_len))

# Encoder Self-Attention
get_all_attention_maps("encoder-decoder", layers, heads, encoder_input_tokens, decoder_input_tokens, min(20, sentence_len))
