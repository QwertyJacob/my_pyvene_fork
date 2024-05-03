import sys
sys.path.append('/home/jovyan/pyreft/pyvene')

import torch
import pandas as pd
from pyvene.models.basic_utils import embed_to_distrib, top_vals, format_token
from pyvene import (
    IntervenableModel,
    AdditionIntervention,
    SubtractionIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene.models.gpt2.modelings_intervenable_gpt2 import create_gpt2

from plotnine import (
    ggplot,
    geom_tile,
    aes,
    facet_wrap,
    theme,
    element_text,
    geom_bar,
    geom_hline,
    scale_y_log10,
)

def activation_addition_position_config(
        model_type,
        intervention_type,
        n_layer):

    reprs = []
    for i in range(n_layer):
        reprs.append(
            RepresentationConfig(
                i,                  # layer
                intervention_type,  # component
                "pos",              # intervention unit
                1,                  # max number of unit
            ))

    config = IntervenableConfig(
        model_type=model_type,
        representations=reprs,
        intervention_types=AdditionIntervention,
    )
    return config


config, tokenizer, gpt = create_gpt2(
    cache_dir='home/jovyan/.cache/huggingface/hub')

config = activation_addition_position_config(
    type(gpt), "mlp_output", gpt.config.n_layer
)

intervenable = IntervenableModel(config, gpt)

base = "The capital of Spain is"
source = "The capital of Italy is"
inputs = [tokenizer(base, return_tensors="pt"), tokenizer(source, return_tensors="pt")]

print(base)
res = intervenable(inputs[0])[1]
distrib = embed_to_distrib(gpt, res.last_hidden_state, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)
print()

print(source)
res = intervenable(inputs[1])[1]
distrib = embed_to_distrib(gpt, res.last_hidden_state, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)

# we can patch mlp with the rome word embedding
rome_token_id = tokenizer(" Rome")["input_ids"][0]
rome_embedding = (
    gpt.wte(torch.tensor(rome_token_id)).clone().unsqueeze(0).unsqueeze(0)
)


base = "The capital of Spain is"

_, counterfactual_outputs = intervenable(
    base=tokenizer(base, return_tensors="pt"),
    unit_locations={
        "sources->base": 4
    },  # last position
    source_representations=rome_embedding,
)
distrib = embed_to_distrib(gpt,
                           counterfactual_outputs.last_hidden_state, 
                           logits=False)

top_vals(tokenizer, distrib[0][-1], n=10)


