"""Read-only search and excerpts over one benchmark row's supplied source."""
from bm25_retrieval.rag_retrieval import subtract_intervals


def function_tool(name, description, argument, value_type):
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': description,
            'strict': True,
            'parameters': {
                'type': 'object',
                'properties': {argument: {'type': value_type}},
                'required': [argument],
                'additionalProperties': False,
            },
        },
    }


TOOLS = [
    function_tool('search_code', 'Search supplied code by identifier or keywords. Returns previews and chunk IDs.',
                  'query', 'string'),
    function_tool('read_code', 'Read the supplied code around a chunk ID returned by search_code.',
                  'chunk_id', 'integer'),
]


class SourceTools:
    def __init__(self, index, tokenizer, total_budget, response_budget):
        self.index = index
        self.tokenizer = tokenizer
        self.total_budget = total_budget
        self.response_budget = response_budget
        self.calls = set()
        self.visible = set()
        self.excerpts = []
        self.code_tokens = 0

    def excerpt(self, chunk_id, start, end, limit):
        code = self.tokenizer.truncate(self.index.code[start:end], limit)
        return {
            'chunk_id': chunk_id, 'start_char': start, 'end_char': start + len(code),
            'code': code, 'code_tokens': self.tokenizer.count(code),
        }

    def execute(self, name, args):
        if name == 'search_code' and set(args) == {'query'} and isinstance(args['query'], str):
            query = ' '.join(args['query'].split())
            if not query or len(query) > 400:
                return {'status': 'invalid_query', 'excerpts': []}
            key = (name, query.casefold())
        elif name == 'read_code' and set(args) == {'chunk_id'} and type(args['chunk_id']) is int:
            key = (name, args['chunk_id'])
            if key[1] not in self.visible:
                return {'status': 'unknown_chunk', 'excerpts': []}
        else:
            return {'status': 'invalid_tool_call', 'excerpts': []}
        if key in self.calls:
            return {'status': 'repeated_call', 'excerpts': []}
        self.calls.add(key)
        remaining = min(self.response_budget, self.total_budget - self.code_tokens)
        if remaining <= 0:
            return {'status': 'evidence_limit', 'excerpts': []}

        excerpts = []
        if name == 'search_code':
            ranked = sorted(self.index.rank(query), key=lambda hit: (-round(hit[1], 12), hit[0]))
            for chunk_id, score in ranked[:3]:
                chunk = self.index.chunks[chunk_id]
                item = self.excerpt(chunk_id, chunk.start, chunk.end, min(300, remaining))
                if not item['code']:
                    continue
                item['score'] = round(score, 12)
                self.visible.add(chunk_id)
                excerpts.append(item)
                remaining -= item['code_tokens']
                if remaining <= 0:
                    break
        else:
            chunk_id = args['chunk_id']
            chunk = self.index.chunks[chunk_id]
            # Start at the hit and include following source within the response budget.
            item = self.excerpt(chunk_id, chunk.start, chunk.end + remaining * 8, remaining)
            if item['code']:
                excerpts.append(item)

        count = sum(item['code_tokens'] for item in excerpts)
        self.code_tokens += count
        self.excerpts.extend(excerpts)
        return {'status': 'ok' if excerpts else 'no_results', 'excerpts': excerpts, 'code_tokens': count}

    def final_context(self):
        """Use only seen source. Preserve its order and remove duplicate spans."""
        spans, fragments, selected = [], [], []
        for item in sorted(self.excerpts, key=lambda hit: hit['start_char']):
            for start, end in subtract_intervals(item['start_char'], item['end_char'], spans):
                text = self.index.code[start:end]
                if self.tokenizer.count('\n\n'.join([*fragments, text])) > self.total_budget:
                    continue
                fragments.append(text)
                spans.append((start, end))
                selected.append({'start_char': start, 'end_char': end})
        return '\n\n'.join(fragments), selected
