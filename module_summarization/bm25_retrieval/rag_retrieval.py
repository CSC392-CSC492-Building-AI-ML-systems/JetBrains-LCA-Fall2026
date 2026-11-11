"""CPU-only BM25 over the supplied code pool; no reference text is accepted here."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'bm25-offset-windows-v1'
STOP = set('a an and are as at be by code documentation document file for from generate in is it module of on or return text that the this to using with'.split())


def digest(value):
    data = value if isinstance(value, bytes) else value.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def terms(text):
    found = []
    for word in re.findall(r'[A-Za-z_][A-Za-z_0-9]*', text):
        parts = re.findall(r'[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+', word)
        for term in dict.fromkeys([word.lower(), *(x.lower() for x in parts)]):
            if len(term) > 1 and term not in STOP:
                found.append(term)
    return found


class CodeTokenizer:
    """Load a local GPT-2 tokenizer JSON, without transformers/torch or network."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.fingerprint = digest(self.path.read_bytes())
        self.backend = Tokenizer.from_file(str(path))
        self.backend.no_truncation()
        self.backend.no_padding()

    def encode_offsets(self, text):
        encoded = self.backend.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def count(self, text):
        return len(self.backend.encode(text, add_special_tokens=False).ids)

    def truncate(self, text, budget):
        if budget <= 0:
            return ''
        end = min(len(text), max(1024, budget * 8))
        while True:
            ids, offsets = self.encode_offsets(text[:end])
            if len(ids) >= budget + 8 or end == len(text):
                break
            end = min(len(text), end * 2)
        if len(ids) <= budget:
            return text
        # Slice source characters rather than decode a partial UTF-8 token.
        end = offsets[budget - 1][1]
        while end and self.count(text[:end]) > budget:
            end -= 1
        return text[:end]


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    counts: dict


def chunk_code(code, tokenizer, size=400, overlap=50):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require size > 0 and 0 <= overlap < size')
    chunks, seen = [], set()
    # Bounded encoding prevents tokenizer max-length truncation and huge token arrays.
    block_chars = 16000
    for block_start in range(0, len(code), block_chars - 400):
        block = code[block_start:block_start + block_chars]
        ids, offsets = tokenizer.encode_offsets(block)
        for pos in range(0, len(ids), size - overlap):
            stop = min(pos + size, len(ids))
            start = block_start + offsets[pos][0]
            end = block_start + offsets[stop - 1][1]
            if (start, end) not in seen and code[start:end].strip():
                chunks.append(Chunk(start, end, dict(Counter(terms(code[start:end])))))
                seen.add((start, end))
            if stop == len(ids):
                break
        if block_start + block_chars >= len(code):
            break
    return chunks


def subtract_intervals(start, end, selected):
    spans = [(start, end)]
    for a, b in selected:
        parts = []
        for left, right in spans:
            if b <= left or a >= right:
                parts.append((left, right))
            else:
                if left < a:
                    parts.append((left, a))
                if right > b:
                    parts.append((b, right))
        spans = parts
    return spans


class BM25Index:
    def __init__(self, code, chunks, k1=1.5, b=.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Invalid BM25 k1/b')
        self.code, self.chunks, self.k1, self.b = code, chunks, k1, b
        self.postings = defaultdict(list)
        self.lengths = [sum(c.counts.values()) for c in chunks]
        self.average = sum(self.lengths) / max(len(chunks), 1) or 1
        for i, chunk in enumerate(chunks):
            for term, count in chunk.counts.items():
                self.postings[term].append((i, count))

    @classmethod
    def load_or_build(cls, code, tokenizer, cache_dir=None, size=400, overlap=50, k1=1.5, b=.75):
        key = digest(json.dumps([VERSION, digest(code), tokenizer.fingerprint, size, overlap]))
        path = Path(cache_dir) / (key + '.json.gz') if cache_dir else None
        if path and path.exists():
            with gzip.open(path, 'rt') as f:
                data = json.load(f)
            if data['key'] != key:
                raise ValueError('Index cache fingerprint mismatch')
            chunks = [Chunk(**c) for c in data['chunks']]
        else:
            chunks = chunk_code(code, tokenizer, size, overlap)
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp')
                with gzip.open(tmp, 'wt') as f:
                    json.dump({'key': key, 'chunks': [c.__dict__ for c in chunks]}, f)
                tmp.replace(path)
        return cls(code, chunks, k1, b)

    def rank(self, query):
        scores = defaultdict(float)
        n = len(self.chunks)
        for term in set(terms(query)):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (n - len(postings) + .5) / (len(postings) + .5))
            for i, tf in postings:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    def retrieve(self, query, tokenizer, budget=2000):
        if budget <= 0:
            raise ValueError('Context budget must be positive')
        ranked = self.rank(query)
        fragments, selected, traces = [], [], []
        fallback = None
        if not ranked:
            fallback = 'empty_query' if not terms(query) else 'no_term_match'
        for i, score in ranked:
            chunk = self.chunks[i]
            for start, end in subtract_intervals(chunk.start, chunk.end, selected):
                text = self.code[start:end]
                candidate = '\n\n'.join(fragments + [text])
                if tokenizer.count(candidate) > budget:
                    if fragments:
                        continue  # A shorter lower-ranked candidate may fit.
                    text = tokenizer.truncate(text, budget)
                    end = start + len(text)
                if not text.strip():
                    continue
                fragments.append(text)
                selected.append((start, end))
                traces.append({'chunk_id': i, 'start_char': start, 'end_char': end, 'score': score})
            if tokenizer.count('\n\n'.join(fragments)) >= budget:
                break
        if not fragments:
            fallback = fallback or 'no_chunk_fits'
            context = tokenizer.truncate(self.code, budget)
            if context:
                traces = [{'chunk_id': None, 'start_char': 0, 'end_char': len(context), 'score': None}]
        else:
            context = '\n\n'.join(fragments)
        assert tokenizer.count(context) <= budget
        return context, {'query': query, 'query_terms': sorted(set(terms(query))),
                         'fallback': fallback, 'candidate_count': len(ranked),
                         'index_chunks': len(self.chunks), 'selected': traces,
                         'context_tokens': tokenizer.count(context), 'context_sha256': digest(context)}
