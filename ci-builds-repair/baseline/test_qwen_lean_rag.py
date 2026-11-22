import unittest

import run_qwen_lean_rag as lean


class LeanRagTests(unittest.TestCase):
    def test_token_count_uses_ids_not_encoding_dictionary_keys(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {'input_ids': [1, 2, 3, 4, 5], 'attention_mask': [1] * 5}
        self.assertEqual(lean.encode_prompt(Tokenizer(), 'test'), [1, 2, 3, 4, 5])

    def test_retrieval_evidence_cannot_overlap_target_or_duplicate_text(self):
        snippet = {'path': 'a.py', 'start': 1, 'end': 3, 'text': 'same code'}
        with self.assertRaises(ValueError):
            lean.validate_evidence([snippet], [{'path': 'a.py', 'start': 3, 'count': 1}])
        with self.assertRaises(ValueError):
            lean.validate_evidence([snippet, {**snippet, 'path': 'b.py'}], [])
    def test_query_uses_failed_log_cues(self):
        logs = [{"step_name": "secret_label", "log":
                 "\x1b[31m2024-01-09T01:02:03.123Z TypeError in pkg/core.py and test_render\x1b[0m\n"
                 "##[error]Process completed with exit code 1"}]
        query, cues = lean.make_lean_query(logs)
        self.assertEqual(cues, ["TypeError", "pkg/core.py", "test_render"])
        for term in ("typeerror", "core", "test_render"):
            self.assertIn(term, query.split())
        for term in ("secret_label", "error", "process"):
            self.assertNotIn(term, query.split())

    def test_pack_enforces_deduplication_overlap_and_budget(self):
        def chunk(path, start, end, text):
            return {"path": path, "start": start, "end": end, "text": text}

        ranked = [
            chunk("target.py", 1, 10, "overlaps"),
            chunk("long.py", 1, 20, "x" * 1000),
            chunk("first.py", 1, 20, "a b"),
            chunk("first.py", 21, 40, "repeat path"),
            chunk("other.py", 1, 20, "a  b"),
            chunk("second.py", 1, 20, "c d"),
            chunk("third.py", 1, 20, "e f"),
            chunk("fourth.py", 1, 20, "g h"),
        ]
        blocks = [{"path": "target.py", "start": 10, "count": 2}]
        self.assertEqual([c["path"] for c in lean.pack("baseline", ranked, blocks, len)],
                         ["first.py", "second.py", "third.py"])
        self.assertEqual(lean.assemble("baseline", []), "baseline")
        self.assertTrue(lean.assemble("baseline", ranked[2:3]).endswith("baseline"))

    def test_unmatched_rows_do_not_count_as_pairs(self):
        row = {"id": 1, "seed": 0, "arm": "plain", "error": None, "nonempty_patch": True,
               "syntax_preserved": True, "patch_exact_match": False, "edit_token_f1": 0.2, "snippets": 0}
        self.assertFalse(lean.summarize([row], [1], [0])["complete"])
        comparison = lean.summarize([row, {**row, "arm": "lean_rag", "snippets": 1, "edit_token_f1": 0.3}], [1], [0])
        self.assertTrue(comparison["complete"])
        self.assertEqual(comparison["paired_f1_wins_ties_losses"], [1, 0, 0])

    def test_pack_respects_explicit_generation_budget(self):
        ranked = [{'path': 'other.py', 'start': 1, 'end': 1, 'text': 'supporting code'}]
        # The base already consumes the entire available input budget.
        self.assertEqual(lean.pack('baseline', ranked, [], len, context=300, cap=40), [])

    def test_improved_scoring_matches_composition_policy(self):
        files = {'a.py': 'x = 1\n\ny = 2\n'}
        gold = [{'path': 'a.py', 'start': 2, 'count': 2, 'old': '\ny = 2\n', 'new': '\ny = 3\n'}]
        blocks = lean.pilot.prediction_blocks(gold)
        answer = {'skip': None, 'finish_reason': 'stop',
                  'response': '[start of a.py#L1#B0]\ny = 3\n[end of a.py#L9#E0]'}
        original = lean.score_answer(answer, blocks, files, gold, 'original')
        improved = lean.score_answer(answer, blocks, files, gold, 'recover-markers')
        self.assertIsNotNone(original['error'])
        self.assertIsNone(improved['error'])
        self.assertTrue(improved['patch_exact_match'])
        capped = lean.score_answer({**answer, 'finish_reason': 'length'}, blocks, files, gold, 'recover-markers')
        self.assertEqual(capped['error'], 'output_token_limit')


if __name__ == "__main__":
    unittest.main()
