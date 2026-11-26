"""Rejudge exactly the 81 Chinese Step2 answers missing the length-soft note.

All artifacts stay beside this file. Original source, judgements and official
scores are read-only inputs. No third-party Python packages are required.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SOURCE = ROOT / "step2.1_Penalty/penalty_evaluator_batch.py"
PROMPTS = ROOT / "step2.1_Penalty/Penalty.json"
PEN_WEIGHTS = ROOT / "step1_Test calibration/Penalty test/weight_calculation_results.csv"
VALUES = {"nice": 0.0, "soft": 1.0, "hard": 3.0}
WRITE_LOCK = threading.Lock()


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def digest(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".pending")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def append_json(path, data):
    with WRITE_LOCK:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
            stream.flush()


def load_source_helpers():
    """Compile pure helpers only; importing the original script is unnecessary."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "OLLAMA_CONFIG", "JUDGE_MODELS", "MODEL_SHORT_NAME"
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    names = {
        "extract_word_limit", "detect_language_type", "count_words",
        "calculate_word_penalty", "modify_step2_system_prompt", "build_user_prompt",
        "clean_json_string", "parse_llm_json_response",
    }
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    scope = {"re": re, "json": json, "datetime": datetime,
             "print": lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), scope)
    return constants, scope


def norm(value):
    return value.lower().replace("-", "_")


def load_inputs(constants, helpers):
    all_rows, files = [], {SOURCE, PROMPTS, PEN_WEIGHTS}
    for report in ("reports1", "reports2"):
        for path in sorted((ROOT / "Dataset/Step2" / report).glob("*.csv")):
            if "deepseek-moe" in path.name:
                continue
            model = norm(re.sub(r"_\d{8}_\d{6}_annotated$", "",
                                path.stem.removeprefix("test_")))
            candidates = [p for p in (ROOT / "step2.1_Penalty/Step2" / report).glob("Penalty_*.csv")
                          if norm(p.stem.removeprefix("Penalty_")) == model]
            if len(candidates) != 1:
                raise ValueError(f"Expected one saved penalty file for {path}")
            penalty_path = candidates[0]
            saved = read_csv(penalty_path)
            files.update((path, penalty_path))
            for i, row in enumerate(read_csv(path)):
                if not row["test_id"].strip():
                    continue
                if (row["prompt"], row["response"]) != (saved[i]["prompt"], saved[i]["response"]):
                    raise ValueError(f"Source/saved answer mismatch: {path}, row {i+1}")
                old_expected = helpers["extract_word_limit"](row["prompt"])
                expected = old_expected
                if row["language"] == "Chinese":
                    match = re.search(r"(\d+)\s*字", row["prompt"])
                    if not match:
                        raise ValueError(f"Cannot extract Chinese target: {path}, row {i+1}")
                    expected = int(match.group(1))
                if not expected:
                    raise ValueError(f"Missing target: {path}, row {i+1}")
                actual = helpers["count_words"](
                    row["response"], helpers["detect_language_type"](row["language"]))
                failed = row["response"].strip() in ("", "(请求失败)", "（请求失败）")
                item_key = f"{model}/{report}/{row['test_id']}"
                item = {
                    "item_key": item_key, "item_id": digest(item_key)[:20],
                    "model": model, "report": report, "test_id": row["test_id"],
                    "source_row": i + 1, "penalty_id": saved[i]["penalty_id"],
                    "source_file": str(path.relative_to(ROOT)),
                    "penalty_file": str(penalty_path.relative_to(ROOT)),
                    "source_step": "Step2", "category": row["category"],
                    "subcategory": row["subcategory"], "language": row["language"],
                    "prompt": row["prompt"], "response": row["response"],
                    "prompt_sha256": digest(row["prompt"]),
                    "response_sha256": digest(row["response"]),
                    "old_expected": old_expected, "expected": expected, "actual": actual,
                    "ratio": actual / expected, "failed": failed,
                    "response_truncated": len(row["response"]) > 2000,
                    "original_verdicts": {}, "original_reasons": {},
                }
                for full_name in constants["JUDGE_MODELS"]:
                    short = constants["MODEL_SHORT_NAME"][full_name]
                    label = saved[i][short + "_verdict"].strip().lower()
                    if label not in VALUES:
                        raise ValueError(f"Invalid saved label: {item_key}/{short}: {label}")
                    item["original_verdicts"][short] = label
                    item["original_reasons"][short] = saved[i][short + "_reason"]
                all_rows.append(item)
    assert len(all_rows) == 1056
    assert len({row["model"] for row in all_rows}) == 22
    assert len({row["item_key"] for row in all_rows}) == 1056
    for path in (ROOT / "step5_eqe_output/eqee_cells.csv",
                 ROOT / "step5_eqe_output/final_rankings.csv"):
        files.add(path)
    return all_rows, files


def selected(row):
    r = row["ratio"]
    return (row["language"] == "Chinese" and not row["failed"]
            and (0.5 <= r < 0.75 or 1.25 < r <= 1.5))


def payload_for(row, full_name, constants, helpers, prompts):
    template = prompts["Step2"][row["category"] + "_" + row["subcategory"]]
    return {
        "model": full_name,
        "prompt": helpers["build_user_prompt"](
            template["user_prompt_template"], row, row["category"], row["subcategory"]),
        "system": helpers["modify_step2_system_prompt"](template["system_prompt"], "soft"),
        "think": False, "stream": False,
        "options": constants["OLLAMA_CONFIG"]["default_options"],
    }


def prepare():
    constants, helpers = load_source_helpers()
    all_rows, files = load_inputs(constants, helpers)
    items = [row for row in all_rows if selected(row)]
    assert len(items) == 81
    assert Counter(row["report"] for row in items) == {"reports1": 39, "reports2": 42}
    assert sum(row["response_truncated"] for row in items) == 13
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8-sig"))
    hashes = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted(files)}
    manifest = {
        "schema": 1,
        "scope": "81 Chinese Step2 answers; hard bounds .5/1.5, soft bounds .75/1.25",
        "original_response_truncation_chars": 2000,
        "rejudge_rows": 81, "expected_judgements": 405,
        "judge_models": constants["JUDGE_MODELS"],
        "model_short_names": constants["MODEL_SHORT_NAME"],
        "options": constants["OLLAMA_CONFIG"]["default_options"],
        "think": False, "stream": False, "timeout_seconds": 120,
        "source_sha256": hashes, "items": items,
    }
    fingerprint = digest(json_bytes(manifest))
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if digest(json_bytes(old)) != fingerprint:
            raise ValueError("Prepared inputs changed; use a separate run directory")
    else:
        write_json(manifest_path, manifest)
        fields = ["item_key", "item_id", "model", "report", "test_id", "source_row",
                  "penalty_id", "source_file", "penalty_file", "category", "subcategory",
                  "expected", "actual", "ratio", "response_truncated",
                  "prompt_sha256", "response_sha256"]
        write_csv(OUT / "manifest.csv", [{k: row[k] for k in fields} for row in items])
        with (OUT / "request_payloads.jsonl").open("w", encoding="utf-8") as stream:
            for name in constants["JUDGE_MODELS"]:
                short = constants["MODEL_SHORT_NAME"][name]
                for row in items:
                    payload = payload_for(row, name, constants, helpers, prompts)
                    stream.write(json.dumps({"item_key": row["item_key"], "judge_short": short,
                                             "payload_sha256": digest(json_bytes(payload)),
                                             "payload": payload}, ensure_ascii=False) + "\n")
    return constants, helpers, all_rows, items, prompts, fingerprint


def http_json(url, payload=None, timeout=20):
    data = json_bytes(payload) if payload is not None else None
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    # This private endpoint is reached directly, consistently with the curl probe.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def preflight(base_url, constants, fingerprint):
    status = {"checked_at": now(), "base_url": base_url,
              "manifest_sha256": fingerprint, "ok": False}
    try:
        tags = http_json(base_url + "/api/tags")
        available = {m["name"] for m in tags.get("models", [])}
        missing = [m for m in constants["JUDGE_MODELS"] if m not in available]
        status.update(tags=tags, missing_models=missing, ok=not missing)
        if missing:
            raise RuntimeError("Missing exact judge tags: " + ", ".join(missing))
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(OUT / "preflight.json", status)
        raise
    write_json(OUT / "preflight.json", status)
    return status


def completed(fingerprint):
    path = OUT / "judgements.jsonl"
    results = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        if result["manifest_sha256"] != fingerprint:
            raise ValueError("Checkpoint uses different source inputs")
        if result["new_verdict"] not in VALUES:
            raise ValueError("Checkpoint contains an invalid label")
        key = result["item_key"], result["judge_short"]
        if key in results:
            raise ValueError(f"Duplicate completed judgement: {key}")
        results[key] = result
    return results


def run_one(row, name, base_url, constants, helpers, prompts, fingerprint, attempts):
    short = constants["MODEL_SHORT_NAME"][name]
    payload = payload_for(row, name, constants, helpers, prompts)
    payload_hash = digest(json_bytes(payload))
    rawdir = OUT / "raw_responses"
    rawdir.mkdir(exist_ok=True)
    error = None
    for attempt in range(1, attempts + 1):
        started = now()
        tick = time.monotonic()
        record = {"item_key": row["item_key"], "judge_model": name, "judge_short": short,
                  "started_at": started, "attempt": attempt, "base_url": base_url,
                  "payload_sha256": payload_hash, "manifest_sha256": fingerprint}
        try:
            response = http_json(base_url + "/api/generate", payload, timeout=120)
            record["api_response"] = response
            label, reason, parsed = helpers["parse_llm_json_response"](
                response.get("response", ""), debug_mode=False)
            record.update(parsed_verdict=label, parsed_reason=reason, parse_result=parsed)
            if label not in VALUES:
                raise ValueError(f"Invalid/unknown judge verdict: {label!r}")
            record["ok"] = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            record.update(ok=False, error=error)
            if isinstance(exc, urllib.error.HTTPError):
                record["http_error_body"] = exc.read().decode("utf-8", errors="replace")
        record.update(finished_at=now(), elapsed_seconds=time.monotonic() - tick)
        path = rawdir / f"{row['item_id']}__{short}__{time.time_ns()}.json"
        write_json(path, record)
        append_json(OUT / "attempts.jsonl", {k: v for k, v in record.items() if k != "api_response"})
        if record["ok"]:
            return {
                "item_key": row["item_key"], "item_id": row["item_id"],
                "model": row["model"], "report": row["report"], "test_id": row["test_id"],
                "source_row": row["source_row"], "judge_model": name, "judge_short": short,
                "old_verdict": row["original_verdicts"][short], "new_verdict": label,
                "new_reason": reason, "raw_file": str(path.relative_to(OUT)),
                "payload_sha256": payload_hash, "manifest_sha256": fingerprint,
                "finished_at": record["finished_at"],
                "elapsed_seconds": record["elapsed_seconds"],
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "api_total_duration_ns": response.get("total_duration"),
            }
        if attempt < attempts:
            time.sleep(2)
    raise RuntimeError(f"{row['item_key']}/{short}: {error}")


def summarize(results, fingerprint, error=None):
    transitions = Counter((r["old_verdict"], r["new_verdict"]) for r in results.values())
    summary = {"updated_at": now(), "manifest_sha256": fingerprint,
               "expected_judgements": 405, "completed_judgements": len(results),
               "remaining_judgements": 405 - len(results), "complete": len(results) == 405,
               "per_judge": dict(Counter(r["judge_short"] for r in results.values())),
               "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())}}
    if error:
        summary["last_error"] = str(error)
    write_json(OUT / "summary.json", summary)
    write_csv(OUT / "judgement_comparison.csv", list(results.values()))
    return summary


def run(args, prepared):
    constants, helpers, all_rows, items, prompts, fingerprint = prepared
    results = completed(fingerprint)
    summarize(results, fingerprint)
    preflight(args.base_url, constants, fingerprint)
    for name in constants["JUDGE_MODELS"]:
        short = constants["MODEL_SHORT_NAME"][name]
        todo = [row for row in items if (row["item_key"], short) not in results]
        print(f"MODEL {name}: pending={len(todo)}", flush=True)
        # The first real judgement checks generation before scheduling the batch;
        # it is persisted and counts toward the 405, never a disposable probe.
        if todo:
            first = run_one(todo.pop(0), name, args.base_url, constants, helpers,
                            prompts, fingerprint, args.attempts)
            append_json(OUT / "judgements.jsonl", first)
            results[first["item_key"], short] = first
            summarize(results, fingerprint)
            print(f"PROGRESS {len(results)}/405 {short} {first['old_verdict']}->{first['new_verdict']}", flush=True)
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, row, name, args.base_url, constants,
                                   helpers, prompts, fingerprint, args.attempts): row for row in todo}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    append_json(OUT / "judgements.jsonl", result)
                    results[result["item_key"], short] = result
                    summarize(results, fingerprint)
                    print(f"PROGRESS {len(results)}/405 {short} {result['old_verdict']}->{result['new_verdict']}", flush=True)
                except Exception as exc:
                    failed.append(str(exc))
                    print(f"FAILED {exc}", flush=True)
        if failed:
            summarize(results, fingerprint, "\n".join(failed))
            raise RuntimeError(f"{short}: {len(failed)} failed requests; resume to retry only missing results")
    # Check the original inputs again before producing comparative scores.
    prepare()
    analyze(prepared, results)
    print(json.dumps(summarize(results, fingerprint), ensure_ascii=False, indent=2), flush=True)


def aggregate(cells):
    grouped = defaultdict(list)
    for cell in cells:
        grouped[cell["merge_key"]].append(cell)
    final = {}
    for model, rows in grouped.items():
        repeats = defaultdict(list)
        for row in rows:
            repeats[row["report"]].append(row)
        assert len(rows) == 10 and all(len(rr) == 5 for rr in repeats.values())
        repeat_values = [{metric: statistics.mean(float(c[metric]) for c in rr)
                          for metric in ("EQ", "EQEE")} for rr in repeats.values()]
        final[model] = {"model": model}
        for metric in ("EQ", "EQEE"):
            vals = [r[metric] for r in repeat_values]
            final[model][metric] = statistics.mean(vals)
            final[model][metric + "_std"] = statistics.stdev(vals)
    for row in final.values():
        for metric in ("EQ", "EQEE"):
            row[metric + "_rank"] = 1 + sum(r[metric] > row[metric] for r in final.values())
    return final


def analyze(prepared, results):
    constants, helpers, rows, items, prompts, fingerprint = prepared
    if len(results) != 405:
        raise ValueError(f"Refusing partial score analysis: {len(results)}/405 judgements")
    required = {(row["item_key"], constants["MODEL_SHORT_NAME"][name])
                for row in items for name in constants["JUDGE_MODELS"]}
    assert set(results) == required
    weight_rows = read_csv(PEN_WEIGHTS)
    weights = {r["model"].replace(".", "_"): float(r["omega"]) for r in weight_rows}
    assert abs(sum(weights.values()) - 1) < 1e-12
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["report"]].append(row)
    penalties = {scenario: {} for scenario in ("baseline", "hard_only", "hard_and_soft")}
    for key, rs in groups.items():
        assert len(rs) == 24
        totals = defaultdict(float)
        for row in rs:
            old_n = row["old_expected"]
            oldhard = row["failed"] or (old_n and (row["actual"] / old_n < .5 or row["actual"] / old_n > 2))
            newhard = row["failed"] or row["ratio"] < .5 or row["ratio"] > 1.5
            for short, weight in weights.items():
                old_label = row["original_verdicts"][short]
                totals["baseline"] += weight * (3. if oldhard else VALUES[old_label])
                totals["hard_only"] += weight * (3. if newhard else VALUES[old_label])
                new_label = results[row["item_key"], short]["new_verdict"] if selected(row) else old_label
                totals["hard_and_soft"] += weight * (3. if newhard else VALUES[new_label])
        for scenario in penalties:
            penalties[scenario][key] = totals[scenario] / len(rs)
    original_cells = read_csv(ROOT / "step5_eqe_output/eqee_cells.csv")
    assert len(original_cells) == 220
    baseline_error = max(abs(float(c["WP"]) - penalties["baseline"][c["merge_key"], c["report"]])
                         for c in original_cells if c["substep"] == "Step2")
    if baseline_error > 1e-12:
        raise ValueError(f"Baseline WP replay mismatch: {baseline_error}")
    original_final = aggregate(original_cells)
    saved_final = {r["model"]: r for r in read_csv(ROOT / "step5_eqe_output/final_rankings.csv")}
    final_error = max(abs(r[metric] - float(saved_final[m][metric]))
                      for m, r in original_final.items() for metric in ("EQ", "EQEE"))
    if final_error > 1e-10:
        raise ValueError(f"Baseline final-score replay mismatch: {final_error}")
    finals = {"baseline": original_final}
    for scenario in ("hard_only", "hard_and_soft"):
        cells = [dict(c) for c in original_cells]
        for c in cells:
            if c["substep"] == "Step2":
                c["WP"] = penalties[scenario][c["merge_key"], c["report"]]
                c["Q_raw"] = float(c["WA"]) - c["WP"]
        step2 = [c for c in cells if c["substep"] == "Step2"]
        minimum, maximum = min(c["Q_raw"] for c in step2), max(c["Q_raw"] for c in step2)
        for c in step2:
            c["EQ"] = c["Q_total"] = (c["Q_raw"] - minimum) / (maximum - minimum) * 100
            c["EQE_total"] = c["EQ"] / float(c["total_energy_joule"])
            c["EQE_gpu"] = c["EQ"] / float(c["gpu_energy_joule"])
            c["EQEE"] = c["EQE_total"] * 1e6
        finals[scenario] = aggregate(cells)
        write_csv(OUT / f"quality_cells_{scenario}.csv", cells)
        write_csv(OUT / f"final_rankings_{scenario}.csv",
                  sorted(finals[scenario].values(), key=lambda r: r["EQEE_rank"]))
    comparison = []
    for model in sorted(original_final, key=lambda m: original_final[m]["EQ_rank"]):
        row = {"model": model}
        for scenario, scores in finals.items():
            for metric in ("EQ", "EQEE", "EQ_rank", "EQEE_rank"):
                row[scenario + "_" + metric] = scores[model][metric]
        for metric in ("EQ", "EQEE", "EQ_rank", "EQEE_rank"):
            row["change_" + metric] = finals["hard_and_soft"][model][metric] - original_final[model][metric]
        comparison.append(row)
    write_csv(OUT / "score_comparison.csv", comparison)
    write_json(OUT / "score_verification.json", {
        "checked_at": now(), "manifest_sha256": fingerprint,
        "baseline_wp_max_abs_error": baseline_error,
        "baseline_final_score_max_abs_error": final_error,
        "judgements": 405, "cells_per_scenario": 220, "models_per_scenario": 22,
        "fixed_inputs": ["judge weights", "ability scores", "energy", "nonselected LLM judgements"],
        "official_outputs_modified": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "run", "status", "analyze"))
    parser.add_argument("--base-url", default="http://100.116.124.0:11434")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if not 1 <= args.workers <= 4 or not 1 <= args.attempts <= 3:
        parser.error("workers must be 1..4 and attempts 1..3")
    prepared = prepare()
    constants, helpers, rows, items, prompts, fingerprint = prepared
    print(f"Prepared {len(items)} answers x {len(constants['JUDGE_MODELS'])} judges = 405 requests", flush=True)
    try:
        if args.action == "check":
            print(json.dumps(preflight(args.base_url, constants, fingerprint), ensure_ascii=False, indent=2))
        elif args.action == "run":
            run(args, prepared)
        elif args.action == "analyze":
            analyze(prepared, completed(fingerprint))
        elif args.action == "status":
            print(json.dumps(summarize(completed(fingerprint), fingerprint), ensure_ascii=False, indent=2))
        else:
            print(f"Manifest SHA256: {fingerprint}")
    except Exception as exc:
        summarize(completed(fingerprint), fingerprint, f"{type(exc).__name__}: {exc}")
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
