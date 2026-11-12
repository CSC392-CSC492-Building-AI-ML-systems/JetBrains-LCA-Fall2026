import unittest

from run_edit_format_pilot import LEGACY, SPARSE, parse_sparse, sparse_prompt, summarize


class EditFormatTests(unittest.TestCase):
    blocks = [{"old": "first\n"}, {"old": "second\r\n"}]

    def test_omitted_blocks_and_newline(self):
        self.assertEqual(parse_sparse('{"edits":{"B1":"replacement"}}', self.blocks),
                         ["first\n", "replacement\r\n"])
        self.assertEqual(parse_sparse('{"edits":{}}', self.blocks), ["first\n", "second\r\n"])

    def test_empty_replacement_and_single_json_fence(self):
        response = '```json\n{"edits":{"B0":""}}\n```'
        self.assertEqual(parse_sparse(response, self.blocks), ["", "second\r\n"])

    def test_reject_bad_or_ambiguous_edits(self):
        examples = [
            ('{"edits":{"B0":"one","B0":"two"}}', "duplicate_json_key"),
            ('{"edits":{"B2":"one"}}', "unknown_block_id"),
            ('{"edits":{"B0":null}}', "invalid_replacement"),
            ('{"edits":{"B0":"\\ud800"}}', "invalid_replacement"),
            ('{"edits":{},"extra":1}', "invalid_schema"),
            ('{"edits":{}}\ncommentary', "invalid_json"),
        ]
        for answer, error in examples:
            with self.subTest(answer=answer), self.assertRaisesRegex(ValueError, error):
                parse_sparse(answer, self.blocks)

    def test_prompt_keeps_exact_input_blocks(self):
        code = "[start of one.py#L2#B0]\n    abc\n[end of one.py#L2#B0]"
        self.assertEqual(sparse_prompt("log" + LEGACY + code), "log" + SPARSE + code)

    def test_no_unmatched_contrasts(self):
        result = summarize([], 1)
        self.assertFalse(result["complete"])
        self.assertEqual(result["matched_contrasts"]["edit_token_f1"]["paired_cases"], 0)


if __name__ == "__main__":
    unittest.main()
