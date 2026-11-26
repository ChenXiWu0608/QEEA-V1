"""Reproduce the current Step1--5 results offline using saved judge records.

Run from any directory: python <path>/run_pipeline.py
Steps 2.1/2.2 are input judgements, not new model inference. Their deterministic
rules and question alignment are applied by Steps 3/4 when reading those inputs.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'step5_eqe_output'
SCRIPTS = [
    'step1_Test calibration/vs/calculate_judge_weights.py',
    'step1_Test calibration/Penalty test/calculate_weights.py',
    'step3_ability_output/step3_ability.py',
    'step4_q_total_output/step4_q_total.py',
    'step5_eqe_output/step5_eqe.py',
    'step1_Test calibration/prepare_chart_data.py',
]


def verify():
    """Check coverage and compare to the saved manuscript snapshot, if present."""
    cells = pd.read_csv(OUT/'eqee_cells.csv')
    keys = ['model', 'report', 'substep']
    cells['model'] = cells.merge_key
    if len(cells) != 220 or cells.duplicated(keys).any() or cells.model.nunique() != 22:
        raise ValueError('Invalid final cell coverage')
    if not np.isfinite(cells[['EQ','EQEE','total_energy_joule','gpu_energy_joule']]).all().all():
        raise ValueError('Non-finite final value')
    if (cells.total_energy_joule <= 0).any() or (cells.gpu_energy_joule <= 0).any():
        raise ValueError('Non-positive energy')
    weights = pd.read_csv(ROOT/'step1_Test calibration/vs/weight_calculation_results.csv')
    if len(weights) != 5 or not weights.n.eq(100).all() or abs(weights.omega.sum()-1) > 1e-12:
        raise ValueError('Invalid win calibration coverage/normalization')
    penalty_weights = pd.read_csv(ROOT/'step1_Test calibration/Penalty test/weight_calculation_results.csv')
    if len(penalty_weights) != 5 or not penalty_weights.n.eq(100).all() or abs(penalty_weights.omega.sum()-1) > 1e-12:
        raise ValueError('Invalid penalty calibration coverage/normalization')
    items = pd.read_csv(ROOT/'step3_ability_output/item_merged.csv', dtype={'item_id': str})
    workflow = items[items.test_step.eq('Step3')]
    selected = pd.read_csv(ROOT/'step3_ability_output/workflow_selected_items.csv', dtype={'item_id':str})
    counts = selected.groupby(['test_model','baseline_anchor','judge_model']).size()
    if len(counts) != 880 or not counts.eq(8).all():
        raise ValueError('Workflow must select 880 groups of 8 identical-question comparisons')
    invalid_workflow = selected[selected.vs_1_status.ne('valid') & selected.vs_2_status.ne('valid')]
    if len(workflow) + len(invalid_workflow) != len(selected):
        raise ValueError('Workflow omissions must exactly match explicitly unusable saved judgements')
    windows = pd.read_csv(OUT/'energy_task_windows.csv')
    checks = pd.read_csv(OUT/'energy_source_checks.csv')
    report = {'verified_at': datetime.now().astimezone().isoformat(),
              'win_calibration_items': 100, 'penalty_calibration_items': 100,
              'cells': len(cells), 'models': cells.model.nunique(),
              'workflow_selected_items': len(selected), 'workflow_scored_items': len(workflow),
              'workflow_unusable_judgements': invalid_workflow[['test_model','baseline_anchor','judge_model','item_id']].to_dict('records'),
              'energy_mj': float(cells.total_energy_joule.sum()/1e6),
              'legacy_missing_energy_rows': int(windows.legacy_j.isna().sum()),
              'manual_duration_rows': int(windows.manual_duration.notna().sum()),
              'boundary_rows': int(windows.boundary_ms.gt(0).sum()),
              'gpu_channel_legacy_check_max_delta_j': float(checks.legacy_gpu_nearest_max_delta_j.max())}
    protocol_path = ROOT/'step2.1_Penalty/length_protocol.json'
    if protocol_path.exists():
        from pipeline_sources import word_helpers
        helpers = word_helpers()
        if helpers['extract_word_limit']('请写一篇500字的文章。') != 500:
            raise ValueError('Chinese target-length extraction regressed')
        expected = {499: 'hard', 500: 'soft', 749: 'soft', 750: None,
                    1250: None, 1251: 'soft', 1500: 'soft', 1501: 'hard'}
        if any(helpers['calculate_word_penalty'](actual, 1000) != label
               for actual, label in expected.items()):
            raise ValueError('Evaluation length boundaries differ from the current 25%/50% protocol')
        report['length_evaluation_protocol'] = json.loads(protocol_path.read_text(encoding='utf-8'))
    snapshot = ROOT/'manuscript_revision_20261019/cells_weights100.csv'
    if snapshot.exists():
        prior = pd.read_csv(snapshot)
        same = cells.merge(prior, on=keys, suffixes=('_pipeline','_snapshot'), validate='one_to_one')
        if len(same) != 220:
            raise ValueError('Manuscript snapshot coverage mismatch')
        errors = {}
        for field in ['WA','WP','Q_raw','Q_total']:
            errors[field] = float(abs(same[field+'_pipeline']-same[field+'_snapshot']).max())
        errors['energy_j'] = float(abs(same.total_energy_joule-same.completion_j).max())
        errors['EQEE'] = float(abs(same.EQEE_pipeline-same.EQEE_snapshot).max())
        if max(errors.values()) > 1e-5 or max(errors[k] for k in ['WA','WP','Q_raw','Q_total','EQEE']) > 1e-8:
            raise ValueError(f'Pipeline differs from saved manuscript: {errors}')
        report['manuscript_snapshot_max_abs_differences'] = errors
    hash_paths = [ROOT/p for p in SCRIPTS] + [ROOT/'pipeline_sources.py', ROOT/'run_pipeline.py',
        ROOT/'step2.1_Penalty/penalty_evaluator_batch.py', ROOT/'step2.2_vs_baseline/dataset_vs_baseline_comparison.py',
        ROOT/'step1_Test calibration/human_comparison_prompts.json']
    hash_paths += list((ROOT/'step1_Test calibration/vs').glob('Test_vs_*.csv'))
    hash_paths += list((ROOT/'step1_Test calibration/Penalty test').glob('Test_result_*.csv'))
    hash_paths += [ROOT/'step1_Test calibration/vs/weight_calculation_results.csv',
                   ROOT/'step1_Test calibration/Penalty test/weight_calculation_results.csv']
    if protocol_path.exists():
        hash_paths.append(protocol_path)
        for report_name in ('reports1', 'reports2'):
            hash_paths += sorted((ROOT/'step2.1_Penalty/Step2'/report_name).glob('Penalty_*.csv'))
        hash_paths += [ROOT/'length_rejudge_20261026/manifest.json',
                       ROOT/'length_rejudge_20261026/judgements.jsonl']
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in hash_paths}
    (OUT/'pipeline_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_sha256'},ensure_ascii=False,indent=2))
    return report


def main():
    for index, name in enumerate(SCRIPTS,1):
        print(f'[{index}/{len(SCRIPTS)}] {name}',flush=True)
        result = subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/name)],cwd=ROOT,
                                capture_output=True,text=True,encoding='utf-8')
        log = OUT/f'pipeline_{index}.log'
        log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}\n{result.stderr[-3000:]}')
    verify()


if __name__ == '__main__':
    main()
