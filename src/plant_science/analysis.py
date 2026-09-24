"""按预注册协议执行确定性统计分析。"""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Iterable, Mapping

from .contracts import Metric, Observation, Protocol
from .numeric import summarize, wilson_interval


ALGORITHM_VERSION = "robot-trials-analysis/2"

# 连续/计数指标至少需要两个有效样本才能诚实地给出方差与置信区间；
# 单样本的 bootstrap 区间宽度为零，会造成“看似精确”的假象。
MIN_INTERVAL_SAMPLES = 2


def _quantile(values: list[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("分位数输入不能为空")
    position = probability * Decimal(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - Decimal(lower)
    return ordered[lower] * (Decimal(1) - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(
    values: Iterable[Decimal], *, seed: int, samples: int
) -> tuple[Decimal, Decimal]:
    data = tuple(values)
    if not data:
        raise ValueError("bootstrap 至少需要一个样本")
    generator = random.Random(seed)
    means: list[Decimal] = []
    for _ in range(samples):
        total = sum((data[generator.randrange(len(data))] for _ in data), Decimal(0))
        means.append(total / Decimal(len(data)))
    return _quantile(means, Decimal("0.025")), _quantile(means, Decimal("0.975"))


def _missing_metric_result(metric: Metric, seed: int, samples: int) -> dict[str, object]:
    """分层内没有任何有效测点时的显式占位，缺失不等于零值。"""

    result: dict[str, object] = {
        "status": "missing",
        "count": 0,
        "zero_count": 0,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "median": None,
        "sample_variance": None,
        "seed": seed,
        "ci_method": "wilson" if metric.kind == "binary" else "bootstrap",
        "ci_lower": None,
        "ci_upper": None,
    }
    if metric.kind == "binary":
        result.update({"successes": 0, "proportion": None, "wilson_lower": None, "wilson_upper": None})
    else:
        result.update({"bootstrap_samples": samples, "bootstrap_lower": None, "bootstrap_upper": None})
    return result


def _metric_result(metric: Metric, values: list[Decimal], seed: int, samples: int) -> dict[str, object]:
    if not values:
        return _missing_metric_result(metric, seed, samples)
    summary = summarize(values)
    result: dict[str, object] = summary.as_dict()
    result["zero_count"] = sum(1 for value in values if value == 0)
    result["seed"] = seed
    if metric.kind == "binary":
        successes = sum(int(value) for value in values)
        interval = wilson_interval(successes, len(values))
        result.update({
            "status": "ok",
            "successes": successes,
            "proportion": successes / len(values),
            "ci_method": "wilson",
            "ci_lower": interval.lower,
            "ci_upper": interval.upper,
            "wilson_lower": interval.lower,
            "wilson_upper": interval.upper,
        })
        return result
    result["ci_method"] = "bootstrap"
    result["bootstrap_samples"] = samples
    if len(values) < MIN_INTERVAL_SAMPLES:
        result.update({
            "status": "degenerate",
            "ci_lower": None,
            "ci_upper": None,
            "bootstrap_lower": None,
            "bootstrap_upper": None,
        })
        return result
    lower, upper = bootstrap_mean_interval(values, seed=seed, samples=samples)
    result.update({
        "status": "ok",
        "ci_lower": format(lower, "f"),
        "ci_upper": format(upper, "f"),
        "bootstrap_lower": format(lower, "f"),
        "bootstrap_upper": format(upper, "f"),
    })
    return result


def analyze(protocol: Protocol, observations: Iterable[Observation]) -> dict[str, object]:
    all_observations = tuple(observations)
    included = tuple(item for item in all_observations if item.excluded_reason is None)
    strata: dict[str, dict[str, object]] = {}
    insufficient: list[dict[str, object]] = []
    for stratum_index, stratum in enumerate(protocol.strata):
        stratum_all = [item for item in all_observations if item.stratum_key == stratum.key]
        rows = [item for item in included if item.stratum_key == stratum.key]
        coverage = {"actual": len(rows), "required": stratum.required_trials, "complete": len(rows) >= stratum.required_trials}
        if not rows:
            stratum_status = "missing"
            insufficient.append({
                "stratum": stratum.key,
                "condition": "stratum_missing",
                "actual": 0,
                "required": stratum.required_trials,
                "detail": "分层没有任何有效测点，缺失不等于零值",
            })
        elif not coverage["complete"]:
            stratum_status = "insufficient"
            insufficient.append({
                "stratum": stratum.key,
                "condition": "coverage_below_required",
                "actual": len(rows),
                "required": stratum.required_trials,
                "detail": "有效测点数低于协议要求",
            })
        else:
            stratum_status = "ok"
        metrics: dict[str, object] = {}
        for metric_index, metric in enumerate(protocol.metrics):
            values = [item.metrics[metric.key] for item in rows]
            seed = protocol.seed + stratum_index * 1009 + metric_index
            metric_result = _metric_result(metric, values, seed, protocol.bootstrap_samples)
            if metric.kind != "binary" and metric_result["status"] == "degenerate":
                insufficient.append({
                    "stratum": stratum.key,
                    "metric": metric.key,
                    "condition": "metric_precision_insufficient",
                    "actual": len(values),
                    "required": MIN_INTERVAL_SAMPLES,
                    "detail": "样本量不足以计算方差与置信区间",
                })
            metrics[metric.key] = metric_result
        strata[stratum.key] = {
            "label": stratum.label,
            "required_trials": stratum.required_trials,
            "status": stratum_status,
            "imported": len(stratum_all),
            "included": len(rows),
            "excluded": len(stratum_all) - len(rows),
            "coverage": coverage,
            "metrics": metrics,
        }

    aggregate: dict[str, object] = {}
    for metric in protocol.metrics:
        available = [
            (stratum.key, strata[stratum.key]["metrics"][metric.key])
            for stratum in protocol.strata
        ]
        blocked = [key for key, value in available if value["status"] != "ok"]
        if blocked:
            aggregate[metric.key] = {
                "available": False,
                "reason": f"分层 {blocked} 缺失或样本精度不足，无法汇总",
            }
            continue
        if metric.kind == "binary":
            weighted = sum(
                protocol.stratum_weights[key] * Decimal(str(value["proportion"]))
                for key, value in available
            )
            lower = sum(
                protocol.stratum_weights[key] * Decimal(str(value["wilson_lower"]))
                for key, value in available
            )
            aggregate[metric.key] = {
                "available": True,
                "weighted_mean": format(weighted, "f"),
                "wilson_lower": format(lower, "f"),
            }
        else:
            weighted = sum(
                protocol.stratum_weights[key] * Decimal(str(value["mean"]))
                for key, value in available
            )
            aggregate[metric.key] = {"available": True, "weighted_mean": format(weighted, "f")}

    rule_results: list[dict[str, object]] = []
    for raw_rule in protocol.admission_rules:
        rule = dict(raw_rule)
        metric_key = str(rule["metric"])
        statistic = str(rule.get("statistic", "weighted_mean"))
        metric_result = aggregate.get(metric_key, {})
        raw_value = metric_result.get(statistic) if isinstance(metric_result, Mapping) else None
        threshold = Decimal(str(rule["threshold"]))
        passed = False
        if raw_value is not None:
            value = Decimal(str(raw_value))
            passed = value >= threshold if rule["operator"] == "gte" else value <= threshold
        rule_results.append({
            "metric": metric_key,
            "statistic": statistic,
            "operator": rule["operator"],
            "threshold": format(threshold, "f"),
            "actual": None if raw_value is None else str(raw_value),
            "passed": passed,
        })
    conclusion = "insufficient" if insufficient else ("pass" if all(item["passed"] for item in rule_results) else "fail")
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": protocol.seed,
        "bootstrap_samples": protocol.bootstrap_samples,
        "imported_count": len(all_observations),
        "included_count": len(included),
        "excluded_count": len(all_observations) - len(included),
        "strata": strata,
        "aggregate": aggregate,
        "rules": rule_results,
        "insufficient": insufficient,
        "conclusion": conclusion,
    }
