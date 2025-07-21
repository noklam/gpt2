import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import transformers
    from transformers import GPT2LMHeadModel
    import torch
    import torch.nn as nn
    return GPT2LMHeadModel, nn


@app.cell
def _(GPT2LMHeadModel):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # Small 124 M model
    state_dict_hf = model_hf.state_dict()
    return (state_dict_hf,)


@app.cell
def _(state_dict_hf):
    for k, v in state_dict_hf.items():
        print(k, v.shape)
    return


@app.cell
def _():
    from gpt2.model_gpt2 import GPT2Config, GPT2Model, GPT2Block
    config = GPT2Config()
    return GPT2Block, GPT2Model, config


@app.cell
def _(GPT2Model, config):
    model = GPT2Model(config)
    return


@app.cell
def _():
    from transformers import pipeline, set_seed
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)
    generator("Hello, I am a language model GPT3", max_length=40, num_return_sequences=5)
    return


@app.cell
def _(config):
    config.n_embdd
    return


@app.cell
def _(GPT2Block, config, nn):
    layer = nn.ModuleDict(dict(
        wte=nn.Embedding(config.vocab_size, config.n_embdd),
        wpe = nn.Embedding(config.block_size, config.n_embdd),
            h = nn.ModuleList(GPT2Block(config) for _ in range(config.n_layer)),
              ln_f = nn.LayerNorm(config.n_embdd),
          
    ))
    return (layer,)


@app.cell
def _(layer):
    layer.state_dict().keys()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
