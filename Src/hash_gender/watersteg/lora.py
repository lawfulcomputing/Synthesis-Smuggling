"""A small LoRA: low-rank additions to chosen linear layers, which can be switched off.

Switched on, the model is the student. Switched off, the same weights are the frozen teacher (base Qwen), so the
training step needs one copy of the model, not two. The base weights are never changed; only A and B train, and
they are kept in float32 whatever the base weights are (updates this small are lost in bfloat16).

`lm_head` is one of the wrapped layers on purpose. A fixed green list (k = 0) asks for "add delta to these
tokens' scores while in a draft", which is a low-rank change at the output layer: one direction in the hidden
state that says "draft, list M" and one column per list that says which tokens it lifts.
"""
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / rank, True
        device = base.weight.device
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)         # B starts at zero: the student starts as the teacher

    # what other code may ask a linear layer for
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        # A and B are cast to the activations' type for the product (they stay float32 as trained weights);
        # casting the activations up instead would copy every layer's input in float32.
        delta = nn.functional.linear(nn.functional.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return out + (delta * self.scale).to(out.dtype)


def add_lora(model, targets, rank, alpha):
    """Freeze the model and wrap every linear layer whose name is in `targets`. Returns the wrapped layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = []
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, rank, alpha)
                setattr(module, name, wrapped)
                layers.append(wrapped)
    if not layers:
        raise ValueError(f"no linear layer named any of {sorted(targets)} in the model")
    return layers


def lora_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def trainable(layers):
    return [p for layer in layers for p in (layer.A, layer.B)]


class adapters_off:
    """`with adapters_off(layers):` runs the model as the frozen teacher."""

    def __init__(self, layers):
        self.layers = layers

    def __enter__(self):
        for layer in self.layers:
            layer.enabled = False

    def __exit__(self, *exc):
        for layer in self.layers:
            layer.enabled = True


def lora_state(model):
    """The trained weights only: {parameter name: a copy on the CPU}. A copy, not a view: on the CPU `.cpu()`
    hands back the live tensor, and a snapshot of the best weights would then keep changing with training."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_lora(model, state):
    params = dict(model.named_parameters())
    missing = sorted(set(state) - set(params))
    if missing:
        raise ValueError(f"adapter weights for layers the model does not have: {missing[:3]} ...")
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor.to(params[name].device, params[name].dtype))


def save_adapter(path, model, meta):
    torch.save({"state": lora_state(model), "meta": meta}, path)


def attach_adapter(model, path):
    """Wrap `model` as its adapter file says and load the weights. Returns (layers, meta)."""
    saved = torch.load(path, map_location="cpu")
    lora = saved["meta"]["lora"]
    layers = add_lora(model, set(lora["targets"]), lora["rank"], lora["alpha"])
    load_lora(model, saved["state"])
    return layers, saved["meta"]
