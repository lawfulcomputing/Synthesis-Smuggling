"""Loading Qwen from disk first. The files are in the local Hugging Face cache; asking the hub about them on
every start is slow and fails without a network, so the hub is only tried when the cache does not have them."""


def local_first(loader, name, **kwargs):
    """`loader(name, ...)` from the local cache; from the hub only if the cache cannot serve it."""
    try:
        return loader(name, local_files_only=True, **kwargs)
    except Exception:  # noqa: BLE001  (not cached, or an older library without the argument)
        return loader(name, **kwargs)


def tokenizer(name):
    from transformers import AutoTokenizer
    return local_first(AutoTokenizer.from_pretrained, name)


def causal_lm(name, **kwargs):
    from transformers import AutoModelForCausalLM
    return local_first(AutoModelForCausalLM.from_pretrained, name, **kwargs)
