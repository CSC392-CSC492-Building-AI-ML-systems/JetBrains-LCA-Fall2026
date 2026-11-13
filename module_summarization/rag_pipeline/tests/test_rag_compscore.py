import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_pipeline.rag_compscore import prepare, execute
from rag_pipeline.rag_replication import Dataset, digest


class ScoringAdapterTests(unittest.TestCase):
    def fixture(self, root):
        path=root/'dataset.jsonl'
        path.write_text('\n'.join(json.dumps({'intent':f'intent{i}',
            'relevant_code_context':f'code{i}', 'target_text':f'gold{i}'}) for i in range(3)))
        dataset=Dataset(path)
        manifest={'dataset_sha256':dataset.sha256, 'original_ids':[2,0], 'strategies':['prefix','bm25']}
        (root/'manifest.json').write_text(json.dumps(manifest))
        for strategy in manifest['strategies']:
            (root/strategy).mkdir()
            for idx in manifest['original_ids']:
                pred=f'{strategy}{idx}'
                (root/strategy/f'{idx}.txt').write_text(pred)
                (root/strategy/f'{idx}.generation.json').write_text(json.dumps({'prediction_sha256':digest(pred)}))
        return dataset

    def test_readiness_missing_predictions_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=self.fixture(root)
            _,plan=prepare(root,data)
            self.assertTrue(plan['ready']); self.assertEqual(plan['judge_evaluations'],8)
            (root/'prefix/2.txt').unlink()
            _,plan=prepare(root,data)
            self.assertFalse(plan['ready'])
            self.assertEqual(plan['missing_predictions'],[{'strategy':'prefix','original_id':2}])
            (root/'prefix/2.txt').write_text('changed')
            with self.assertRaises(ValueError): prepare(root,data)

    def test_invalid_dataset_or_original_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=self.fixture(root)
            manifest=json.loads((root/'manifest.json').read_text())
            for ids in [[3],[2,2],[-1]]:
                manifest['original_ids']=ids
                (root/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): prepare(root,data)
            manifest.update(original_ids=[2,0],dataset_sha256='wrong')
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): prepare(root,data)

    def test_scoring_uses_original_rows_and_same_judge_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=self.fixture(root)
            _,plan=prepare(root,data,judge_revision='judge-sha',judge_tokenizer_revision='tokenizer-sha')
            scorer_factory = Mock(return_value=object())
            tokenizer_factory = Mock(return_value=object())
            calls=[]
            def metric(scorer,intent,code,gold,pred):
                calls.append((intent,code,gold,pred)); return .25
            modules={'metrics':SimpleNamespace(get_metric=metric),
                'utils.scorer':SimpleNamespace(OptionsScoringModel=scorer_factory),
                'utils.context_utils':SimpleNamespace(trim_context=lambda code,tok,n:code+'-trimmed'),
                'transformers':SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_factory))}
            with patch.dict(sys.modules, modules):
                output=execute(root,data,plan,'cuda:0')
            scorer_factory.assert_called_once_with(plan['judge'], 'cuda:0', revision='judge-sha')
            tokenizer_factory.assert_called_once_with(plan['judge_tokenizer'], revision='tokenizer-sha')
            self.assertEqual(calls,[('intent2','code2-trimmed','gold2','prefix2'),
                ('intent2','code2-trimmed','gold2','bm252'),
                ('intent0','code0-trimmed','gold0','prefix0'),
                ('intent0','code0-trimmed','gold0','bm250')])
            self.assertEqual(output['means'],{'prefix':.25,'bm25':.25})
            self.assertEqual(output['plan']['status'],'complete')


if __name__=='__main__': unittest.main()
