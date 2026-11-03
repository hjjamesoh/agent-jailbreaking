import copy
import importlib.util
from pathlib import Path
import unittest


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path('scripts') / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PilotTests(unittest.TestCase):
    def setUp(self):
        runner = script('run_authored_pilot')
        self.rows = runner.load_rows(Path('datasets/authored_pilot_v1/pairs.jsonl'))
        self.audit = runner.audit

    def test_counts_and_shared_counterfactual_rendering(self):
        self.assertTrue(self.audit(self.rows)['passed_structural'])
        for row in self.rows:
            template = row['task_template']
            before, after = template.split('{{risk_context}}')
            for side in ('benign', 'harmful'):
                prompt = template.replace('{{risk_context}}', row[f'{side}_context'])
                self.assertEqual(prompt, before + row[f'{side}_context'] + after)

    def test_unresolved_placeholder_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['task_template'] += ' {{app_name}}'
        with self.assertRaises(AssertionError):
            self.audit(rows)

    def test_family_leak_rejected_even_when_counts_preserved(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['split'], rows[16]['split'] = rows[16]['split'], rows[0]['split']
        with self.assertRaises(AssertionError):
            self.audit(rows)


if __name__ == '__main__':
    unittest.main()
