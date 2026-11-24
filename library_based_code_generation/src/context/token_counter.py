"""Cache delimiter prefixes, then independently verify every final prompt."""

from collections import OrderedDict


class EvidenceCounter:
    def __init__(self, encoding):
        self.encoding = encoding
        self.prefixes = OrderedDict()

    def exact(self, text):
        return len(self.encoding.encode(text, disallowed_special=()))

    def remember(self, text, tokens):
        self.prefixes[text] = tokens
        self.prefixes.move_to_end(text)
        while len(self.prefixes) > 128:
            self.prefixes.popitem(last=False)

    def __call__(self, text):
        # In o200k_base, comma followed by space ends a regex pre-token.
        # Keep the space on the suffix, where it can merge with the next name.
        # Other tokenizers use their full encoder, without this optimization.
        if self.encoding.name != "o200k_base":
            return self.exact(text)
        left, separator, right = text.rpartition(", ")
        if not separator:
            result = self.exact(text)
            self.remember(text + ",", self.exact(text + ","))
            return result
        prefix, suffix = left + ",", " " + right
        if prefix in self.prefixes:
            tokens = self.prefixes[prefix]
            self.prefixes.move_to_end(prefix)
        else:
            tokens = self.exact(prefix)
            self.remember(prefix, tokens)
        result = tokens + self.exact(suffix)
        self.remember(text + ",", tokens + self.exact(suffix + ","))
        return result

    def verify(self, packed, hint):
        if self.exact(packed["prompt"]) + 32 != packed["estimated_input_tokens"]:
            raise ValueError("Optimized count differs from exact final prompt; stop before inference")
        evidence = hint + ", ".join(packed["selected_apis"]) if packed["selected_apis"] else ""
        if self.exact(evidence) != packed["actual_context_tokens"]:
            raise ValueError("Optimized count differs from exact final evidence; stop before inference")
