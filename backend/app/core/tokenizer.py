import tiktoken


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for a piece of text using tiktoken.

    Falls back to the cl100k_base encoding when the requested model
    is not directly supported by tiktoken.
    """
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def estimate_item_tokens(content: str, model: str = "gpt-4o") -> int:
    """Estimate token count for a context item content string."""
    return estimate_tokens(content, model=model)
