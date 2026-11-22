import unittest

from repair_composition import compose_files, compose_replacements, preserve_boundaries, recover_markers


class CompositionTests(unittest.TestCase):
    def test_missing_context_blank_is_restored(self):
        self.assertEqual(preserve_boundaries('\n    old()\n', '    fixed()\n'), '\n    fixed()\n')

    def test_extra_edges_removed_and_inner_layout_unchanged(self):
        text = '\n\ndef f():\n    x = 1\n\n    return x\n\n\n'
        self.assertEqual(preserve_boundaries('def f():\n    return 0\n', text),
                         'def f():\n    x = 1\n\n    return x\n')

    def test_trailing_context_blanks_preserved(self):
        self.assertEqual(preserve_boundaries('old()\n\n\n', 'fixed()\n'), 'fixed()\n\n\n')

    def test_deletions_and_blank_only_blocks_not_invented(self):
        for text in ['', '\n', '  \n']:
            self.assertEqual(preserve_boundaries('\nold()\n', text), text)
        self.assertEqual(preserve_boundaries('\n\n', 'new()\n'), 'new()\n')

    def test_final_newline_conventions(self):
        self.assertEqual(preserve_boundaries('old()', 'new()\n'), 'new()')
        self.assertEqual(preserve_boundaries('\r\nold()\r\n', 'new()\n'), '\r\nnew()\r\n')

    def test_incorrect_code_or_indentation_is_not_repaired(self):
        self.assertEqual(preserve_boundaries('\n    old()\n', 'broken(\n'), '\nbroken(\n')

    def test_reference_text_is_rejected(self):
        block = {'path': 'a.py', 'start': 1, 'count': 1, 'old': 'old()\n', 'new': 'gold()\n'}
        with self.assertRaises(ValueError):
            compose_replacements([block], ['new()\n'])

    def test_policy_is_idempotent(self):
        old, text = '\n\n    old()\n\n', '    new()\n'
        result = preserve_boundaries(old, text)
        self.assertEqual(preserve_boundaries(old, result), result)

    def test_blank_inside_multiline_string_cannot_change(self):
        files = {'a.py': 'text = """\n\nold\n"""\n'}
        blocks = [{'path': 'a.py', 'start': 2, 'count': 2, 'old': '\nold\n'}]
        result, notes = compose_files(files, blocks, ['fixed\n'])
        self.assertEqual(result['a.py'], 'text = """\nfixed\n"""\n')
        self.assertEqual(notes['a.py'], 'ast_change_rejected')

    def test_valid_python_boundary_change_is_accepted(self):
        files = {'a.py': 'x = 1\n\ny = 2\n'}
        blocks = [{'path': 'a.py', 'start': 2, 'count': 2, 'old': '\ny = 2\n'}]
        result, notes = compose_files(files, blocks, ['y = 3\n'])
        self.assertEqual(result['a.py'], 'x = 1\n\ny = 3\n')
        self.assertEqual(notes['a.py'], 'boundary_whitespace_preserved')


class MarkerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [{'path': 'a.py', 'start': 10, 'count': 1, 'old': 'old()\n'}]

    def test_line_aliases_cannot_change_the_target(self):
        answer = '[start of a.py#L9#B0]\nfixed()\n[end of a.py#L12#E0]'
        self.assertEqual(recover_markers(answer, self.blocks), ['fixed()\n'])

    def test_unknown_paths_ids_duplicates_and_prose_rejected(self):
        good = '[start of a.py#L9#B0]\nfixed()\n[end of a.py#L12#E0]'
        for answer in [good.replace('a.py', 'other.py'), good.replace('B0', 'B1'), good + good,
                       'Here is the fix\n' + good, good + '\nAdditional fix', good.rsplit('\n', 1)[0]]:
            with self.assertRaises(ValueError):
                recover_markers(answer, self.blocks)

    def test_all_blocks_required(self):
        blocks = self.blocks + [{'path': 'b.py', 'start': 1, 'count': 1, 'old': 'old()\n'}]
        with self.assertRaises(ValueError):
            recover_markers('[start of a.py#L1#B0]\nfixed()\n[end of a.py#L1#B0]', blocks)


if __name__ == '__main__':
    unittest.main()
