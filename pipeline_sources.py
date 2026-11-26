"""Source alignment and deterministic corrections shared by Steps 2--5.

No audit/recheck/manuscript output is an input. Saved model judgements are
retained as evidence; their applicable rows and deterministic rules are checked
against Dataset, Baseline and RAW on each run.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
JUDGES = ['gemma4_e4b', 'llama3.1_8b', 'ministral3_8b', 'olmo3_7b', 'qwen3.5_9b']
ALIASES = {'qwen3_5_4b': 'qwen3.5-4b', 'qwen3_5_9b': 'qwen3.5-9b',
           'qwen3_5_35b': 'qwen3.5-35b', 'glm4_7_flash': 'glm4.7'}


def norm(value):
    return str(value).strip().lower().replace('-', '_')


def clean(value):
    if value is None or pd.isna(value):
        return ''
    value = str(value)
    return '' if value.strip().lower() in ('nan', 'none') else value


def failed(value):
    return clean(value).strip() in ('', '(请求失败)', '（请求失败）')


def strings(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def task_key(value):
    value = clean(value).strip()
    return str(int(float(value))) if re.fullmatch(r'\d+(?:\.0+)?', value) else value


def match_questions(ds, bl, step):
    """Match main task AND full prompt; retain Dataset's source-row identity."""
    idcol = 'task_id' if step == 'Step3' else 'test_id'
    left, right = ds.copy(), bl.copy()
    for frame in (left, right):
        frame[idcol] = frame[idcol].map(task_key)
        frame['_prompt_key'] = frame.prompt.map(clean)
        frame.drop(frame[frame[idcol].eq('') | frame._prompt_key.str.strip().eq('')].index, inplace=True)
        if step == 'Step3':
            for col in ('category', 'subcategory'):
                if col in frame:
                    frame.drop(frame[frame[col].str.strip().str.lower().eq('code_generation')].index, inplace=True)
    left['_source_row'] = left.index + 1
    keys = [idcol, '_prompt_key']
    merged = left.merge(right, on=keys, suffixes=('_dataset', '_baseline'),
                        how='inner', validate='one_to_one', sort=False)
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f'{step}: unmatched questions: dataset={len(left)}, baseline={len(right)}, matched={len(merged)}')
    return merged


@lru_cache(maxsize=1)
def dataset_index():
    result = {}
    for step in ('Step1', 'Step2', 'Step3'):
        for rn in (1, 2):
            report = f'reports{rn}'
            for analysis in sorted((ROOT/'Dataset'/step/f'analysis{rn}').glob('analysis_*.csv')):
                if '.sync-conflict' in analysis.name:
                    continue
                stamp = re.search(r'_(\d{8}_\d{6})$', analysis.stem)
                model = norm(analysis.stem[:stamp.start()].removeprefix('analysis_'))
                path, = [p for p in (ROOT/'Dataset'/step/report).glob('*.csv') if stamp.group(1) in p.name]
                key = (step, model, report)
                if key in result:
                    raise ValueError(f'Duplicate dataset: {key}')
                result[key] = path
    return result


@lru_cache(maxsize=128)
def workflow_sources(test_model, baseline):
    model, report = norm(test_model).rsplit('_', 1)
    ds = strings(dataset_index()[('Step3', model, report)])
    candidates = [p for p in (ROOT/'Baseline/Step3').glob('*_annotated.csv') if ALIASES[baseline] in p.name]
    if len(candidates) != 1:
        raise ValueError(f'Expected one baseline: {baseline}, {candidates}')
    bl = strings(candidates[0])
    return ds, bl, match_questions(ds, bl, 'Step3')


def select_workflow_rows(frame, test_model, baseline):
    """Select matching answers from either historical 22-row or new 8-row files."""
    ds, bl, matched = workflow_sources(test_model, baseline)
    # Reconstruct the historical row index only to resolve identical-text ties.
    left, right = ds.copy(), bl.copy()
    left.task_id = left.task_id.map(task_key)
    right.task_id = right.task_id.map(task_key)
    cartesian = left.merge(right, on='task_id', suffixes=('_dataset', '_baseline'))
    cartesian = cartesian[~cartesian.category_dataset.str.lower().eq('code_generation')
                          & ~cartesian.subcategory_dataset.str.lower().eq('code_generation')].reset_index(drop=True)
    selected = []
    for _, row in matched.iterrows():
        prompt = row['_prompt_key']
        response, reference = clean(row.response_dataset), clean(row.response_baseline)
        hit = frame[frame.prompt.map(clean).eq(prompt)
                    & frame.dataset_response.map(clean).eq(response)
                    & frame.baseline_response.map(clean).eq(reference)]
        if hit.empty:
            raise ValueError(f'No saved judgement for {test_model}/{baseline}, source row {row._source_row}')
        historical = cartesian.index[cartesian.prompt_dataset.eq(prompt) & cartesian.prompt_baseline.eq(prompt)]
        chosen = historical[0] if len(frame) == len(cartesian) and len(historical) == 1 and historical[0] in hit.index else hit.index[0]
        auto = failed(response) or failed(reference)
        if len(hit[['vs_1', 'vs_2']].drop_duplicates()) > 1 and not auto:
            raise ValueError(f'Ambiguous saved judgement for {test_model}/{baseline}/{row._source_row}')
        item = frame.loc[chosen].copy()
        item['_item_id'] = str(int(row['_source_row']))
        if auto:
            item['vs_1'], item['vs_2'] = (('A=B', 'A=B') if failed(response) and failed(reference)
                                         else ('A<<B', 'A>>B') if failed(response) else ('A>>B', 'A<<B'))
        selected.append(item)
    result = pd.DataFrame(selected).reset_index(drop=True)
    if len(result) != 8:
        raise ValueError(f'Expected 8 retained workflow questions, found {len(result)}')
    return result


@lru_cache(maxsize=1)
def word_helpers():
    path = ROOT/'step2.1_Penalty/penalty_evaluator_batch.py'
    names = {'extract_word_limit', 'detect_language_type', 'count_words', 'calculate_word_penalty'}
    functions = [n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'re': re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def effective_penalties(cfg, normalize_model_name):
    records, changes = [], []
    helpers = word_helpers()
    for (step, model, report), path in dataset_index().items():
        if 'deepseek_moe' in model:
            continue
        ds = strings(path)
        candidates = [p for p in (ROOT/'step2.1_Penalty'/step/report).glob('Penalty_*.csv')
                      if norm(p.stem.removeprefix('Penalty_')) == model]
        if len(candidates) != 1:
            raise ValueError(f'Penalty source missing/duplicated: {path}')
        pen = strings(candidates[0])
        # Blank trailing rows have no Dataset identity and are not observations.
        ds = ds[ds[('task_id' if step == 'Step3' else 'test_id')].str.strip().ne('')].reset_index(drop=True)
        if not ds.prompt.eq(pen.iloc[:len(ds)].prompt).all() or not ds.response.eq(pen.iloc[:len(ds)].response).all():
            raise ValueError(f'Penalty/source answer mismatch: {path}')
        key = normalize_model_name(model + '_' + report)
        for i, row in ds.iterrows():
            if row.category.strip().lower() == 'code_generation':
                continue
            ss = ('ArenaHard' if i < 100 else 'LiveCodeBench' if i < 150 else 'GPQA') if step == 'Step1' else step
            word_hard = False
            if step == 'Step2':
                expected = helpers['extract_word_limit'](row.prompt)
                if expected:
                    actual = helpers['count_words'](row.response, helpers['detect_language_type'](row.language))
                    word_hard = helpers['calculate_word_penalty'](actual, expected) == 'hard'
            for judge in JUDGES:
                raw = pen.iloc[i][judge.replace('.', '_') + '_verdict'].strip().lower()
                value = cfg['verdict_map'].get(raw, np.nan)
                fixed = 3. if failed(row.response) or word_hard else value
                if not np.isfinite(fixed):
                    raise ValueError(f'Unresolved penalty: {path}, row {i+1}, {judge}')
                records.append({'merge_key': key, 'test_step': step, 'substep': ss,
                                'judge_model': judge, 'penalty_val': fixed})
                if not np.isfinite(value) or fixed != value:
                    changes.append({'model': model, 'report': report, 'step': step, 'source_row': i+1,
                                    'judge': judge, 'stored_verdict': raw, 'effective_verdict': 'hard',
                                    'rule': 'failed_response' if failed(row.response) else 'word_count_hard'})
    out = Path(cfg['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(changes).to_csv(out/'deterministic_penalty_corrections.csv', index=False, encoding='utf-8-sig')
    frame = pd.DataFrame(records)
    result = frame.groupby(['merge_key', 'test_step', 'substep', 'judge_model']).penalty_val.agg(['mean', 'count']).reset_index()
    result.columns = ['merge_key', 'test_step', 'substep', 'judge_model', 'Penalty', 'n_pen_items']
    return result


def union(windows):
    merged = []
    for start, end in sorted(windows):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(f'Invalid task interval: {start}, {end}')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def integrate_energy(cfg):
    """Integrate retained task windows, including failed requests, from RAW power."""
    cells, task_records, log_records = [], [], []
    for (step, model, report), path in dataset_index().items():
        if 'deepseek_moe' in model:
            continue
        rn = int(report[-1])
        stamp = re.search(r'\d{8}_\d{6}', path.stem).group()
        if step == 'Step3':
            rawdir = ROOT/'RAW/Step3_Workflow_Data/Original_Data'/report
            rawpath, = rawdir.glob(stamp+'*.csv')
            powerpath, = (rawdir.parent.parent/f'PMD{rn}').glob('*'+stamp+'.csv')
            manual = strings(rawdir.parent/f'timeout_tasks_summary{rn}.csv')
            manual = manual[manual.file_name.eq(rawpath.name)]
        else:
            rawbase, = [p for p in (ROOT/'RAW').iterdir() if ('Benchmark' if step == 'Step1' else 'Long-text') in p.name]
            rawdir = rawbase/report
            rawpath, = rawdir.glob('test_*'+stamp+'.csv')
            def power_model(p):
                base = re.sub(r'^\d{8}_\d{6}_', '', p.stem.removeprefix('PMD2_'))
                return norm(re.sub(r'_\d{8}_\d{6}$', '', base))
            powerpath, = [p for p in rawdir.glob('PMD2_*.csv') if power_model(p) == model]
            manual = None
        raw, ds = strings(rawpath), strings(path)
        ds = ds[ds['task_id' if step == 'Step3' else 'test_id'].str.strip().ne('')].reset_index(drop=True)
        if len(raw) != len(ds) or not raw.prompt.eq(ds.prompt).all():
            raise ValueError(f'Raw task/source mismatch: {path}')
        analysis_path, = (ROOT/'Dataset'/step/f'analysis{rn}').glob('analysis_*'+stamp+'.csv')
        an = pd.read_csv(analysis_path, dtype={'task_id': str})
        tasks = []
        for i, row in ds.iterrows():
            if row.category.strip().lower() == 'code_generation':
                continue
            source = raw.iloc[i]
            start, end = float(source.start_time_ms), float(source.end_time_ms)
            duration = np.nan
            if manual is not None:
                hit = manual[(manual.workflow_name == source.workflow_name) & (manual.question_index == source.question_index)
                             & (pd.to_numeric(manual.start_time_ms) == start)]
                if len(hit) > 1:
                    raise ValueError(f'Duplicate saved duration: {rawpath}, {i+1}')
                if len(hit):
                    duration = pd.to_numeric(hit.actual_duration_sec.iloc[0], errors='coerce')
            completion = start + duration*1000 if np.isfinite(duration) and duration > 0 else end
            ss = ('ArenaHard' if i < 100 else 'LiveCodeBench' if i < 150 else 'GPQA') if step == 'Step1' else step
            aid = str(i+1) if step == 'Step3' else row.test_id
            a, = an[an.task_id.eq(aid)].to_dict('records')
            tasks.append({'model': model, 'report': report, 'step': step, 'substep': ss,
                          'source_row': i+1, 'start': start, 'client': end, 'completion': completion,
                          'manual_duration': duration, 'legacy_j': a['total_energy_joule'],
                          'legacy_gpu_j': a['gpu_energy_joule'], 'quality_failed': failed(row.response)})
        power = pd.read_csv(powerpath, usecols=lambda c: c == 'timestamp_ms' or c.endswith('_P(W)')).sort_values('timestamp_ms')
        times = power.timestamp_ms.to_numpy(dtype=float)
        channels = power.drop(columns='timestamp_ms')
        if not np.isfinite(channels.to_numpy()).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
            raise ValueError(f'Invalid power samples: {powerpath}')
        gpu_cols = ['HPWR1_P(W)', 'PCIE3_P(W)']
        powers = np.stack([channels.sum(axis=1).to_numpy(), channels[gpu_cols].sum(axis=1).to_numpy()], axis=1)
        dt = np.diff(times)/1000
        accum = np.vstack([np.zeros(2), np.cumsum((powers[:-1]+powers[1:])*dt[:,None]/2, axis=0)])
        tolerance = np.median(np.diff(times))*1.5
        def primitive(x):
            if x < times[0]:
                if times[0]-x > tolerance: raise ValueError(f'Power start does not cover {powerpath}: {x}')
                return powers[0]*(x-times[0])/1000
            if x > times[-1]:
                if x-times[-1] > tolerance: raise ValueError(f'Power end does not cover {powerpath}: {x}')
                return accum[-1]+powers[-1]*(x-times[-1])/1000
            k = min(np.searchsorted(times,x,side='right')-1,len(times)-2)
            dx = (x-times[k])/1000
            return accum[k]+powers[k]*dx+(powers[k+1]-powers[k])/dt[k]*dx*dx/2
        t = pd.DataFrame(tasks)
        for ss,g in t.groupby('substep'):
            windows = union(zip(g.start,g.completion))
            value = sum((primitive(b)-primitive(a) for a,b in windows), start=np.zeros(2))
            cells.append({'merge_key': model, 'test_step': step, 'substep': ss, 'report': report,
                          'total_energy_joule': value[0], 'gpu_energy_joule': value[1],
                          'n_tasks': len(g), 'missing_legacy_rows': int(g.legacy_j.isna().sum()),
                          'manual_duration_rows': int(g.manual_duration.notna().sum()), 'union_windows': len(windows)})
        gpu_deltas = []
        for row in t.itertuples(index=False):
            nearest = np.abs(times-row.start).argmin(), np.abs(times-row.client).argmin()
            delta = float(accum[nearest[1],1]-accum[nearest[0],1]-row.legacy_gpu_j)
            if step != 'Step3' and np.isfinite(delta): gpu_deltas.append(abs(delta))
            task_records.append(row._asdict() | {'boundary_ms': max(times[0]-row.start,0)+max(row.completion-times[-1],0)})
        log_records.append({'model': model, 'report': report, 'step': step,
                            'power_path': str(powerpath.relative_to(ROOT)), 'samples': len(times),
                            'gpu_channels': '+'.join(gpu_cols), 'median_period_ms': float(np.median(np.diff(times))),
                            'legacy_gpu_nearest_max_delta_j': max(gpu_deltas, default=np.nan)})
        if len(log_records)%12 == 0: print(f'Integrated {len(log_records)}/132 RAW power logs', flush=True)
    out = Path(cfg['out_dir'])
    pd.DataFrame(task_records).to_csv(out/'energy_task_windows.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(log_records).to_csv(out/'energy_source_checks.csv',index=False,encoding='utf-8-sig')
    result = pd.DataFrame(cells)
    if len(result) != 220 or result.merge_key.nunique() != 22:
        raise ValueError('Energy coverage must be 22 configurations x 2 reports x 5 substeps')
    result.to_csv(out/'energy_cells.csv',index=False,encoding='utf-8-sig')
    return result
