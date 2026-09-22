#!/usr/bin/env python3
"""Check frozen Engram source identity and published result scope without ML deps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'code/engram'
RESULTS = ROOT / 'results/engram'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(RESULTS / 'source_manifest.json')
    assert digest({k: manifest[k] for k in ('files', 'versions')}) == manifest['sha256']
    assert len(manifest['files']) == 22
    actual_runtime = {'run_engram.py', *[str(p.relative_to(CODE)) for p in (CODE / 'bnhgq2').glob('*.py')]}
    assert actual_runtime == set(manifest['files'])
    supplemental = read(RESULTS / 'supplementary_files.json')['files']
    for name, expected in {**manifest['files'], **supplemental}.items():
        assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == expected, name

    configs = {}
    for i in range(8):
        name = f'engram-e{i:02d}-s1'
        cfg = read(CODE / 'configs/engram' / (name + '.json'))
        assert cfg['name'] == name and cfg['experiment']['seed'] == 1
        assert cfg['arch']['n_part'] == 16 and cfg['train']['epochs'] == 1000
        assert cfg['engram_study']['cost_convention'] == 'native_hgq2_plus_custom_estimate'
        configs[name] = cfg

    status = read(RESULTS / 'status-20260921.json')
    assert status['metric_split'] == 'internal_validation' and status['n_validation'] == 124000
    assert not status['hardware_validated'] and not status['held_out_evaluation_updated']
    rows = status['runs']
    assert {r['run'] for r in rows} == {f'engram-e{i:02d}-s1' for i in range(4)}
    assert len(rows) == 4
    for row in rows:
        name = row['run']
        assert row['config_sha256'] == digest(configs[name]), name
        assert row['source_manifest_sha256'] == manifest['sha256']
        assert 0 < row['completed_epochs'] <= row['target_epochs'] == 1000
        assert row['data_identity']['n_train'] == 496000
        assert row['data_identity']['n_val'] == 124000
        assert row['best_feasible'] is None and not row['verified_final_result_available']
        assert row['lowest_cost']['ebops'] > row['target_selection_cost'] == 350000
        metrics = row['latest_history_metrics']
        assert metrics['epoch'] + 1 == row['completed_epochs']
        assert all(0 <= metrics[k] <= 1 for k in ('val_macro_auc', 'val_categorical_accuracy'))
        costs = row['latest_cost_breakdown']
        assert costs['native_hgq2_backbone_ebops'] + costs['custom_estimated_bitops'] == costs['selection_cost'] == metrics['ebops']
        module = row['cost_contract']['module']
        assert costs['custom_estimated_bitops'] == (module['estimated_bitops'] if module else 0)
        if name != 'engram-e00-s1':
            assert row['completed_epochs'] == 1000 and row['execution_status'] == 'failed_final_validation'
        old = row['artifact_status']['engram_result.json']
        if old['exists']:
            assert old['completed_epochs'] == 100 < row['completed_epochs']
            assert 'historical_100_epoch_screen' in old['scope']

    failures = status['finalization_failures']
    assert {r['run'] for r in failures} == {f'engram-e{i:02d}-s1' for i in (1, 2, 3)}
    for failure in failures:
        assert failure['exception'] == 'AssertionError'
        assert failure['mismatched_metrics']
        for metric in failure['mismatched_metrics']:
            assert abs(metric['reloaded'] - metric['recorded']) > 1e-7

    for path in [CODE / 'README.md', ROOT / 'docs/current-work/ENGRAM_STUDY.md']:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent / target.split('#')[0]).exists(), (path, target)
    print('Validated Engram source hashes, eight configs, four run records, three finalization failures and publication links')


if __name__ == '__main__':
    main()
