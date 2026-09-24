"""按预注册协议执行确定性统计分析。

每个协议声明的工况层都会输出样本量、均值、方差和置信区间，
并明确区分缺失（从未观测或读数为空）、剔除（复核后排除）和真实零值。
连续与计数指标的置信区间来自固定随机种子的可重放抽样，
每次抽样的均值序列摘要随结果保存，便于审计重放核对。
"""

from __future__ import annotations

import hashlib
import random
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from .contracts import Metric, Observation, Protocol
from .numeric import summarize, wilson_interval


ALGORITHM_VERSION = "robot-trials-analysis/2"

_LOWER_QUANTILE = Decimal("0.025")
_UPPER_QUANTILE = Decimal("0.975")


def _quantile(values: Sequence[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("分位数输入不能为空")
    position = probability * Decimal(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - Decimal(lower)
    return ordered[lower] * (Decimal(1) - fraction) + ordered[upper] * fraction


def bootstrap_means(values: Iterable[Decimal], *, seed: int, samples: int) -> tuple[Decimal, ...]:
    """按固定种子重放有放回抽样，返回每次重抽样的均值序列。"""

    data = tuple(values)
    if not data:
        raise ValueError("bootstrap 至少需要一个样本")
    generator = random.Random(seed)
    means: list[Decimal] = []
    for _ in range(samples):
        total = sum((data[generator.randrange(len(data))] for _ in data), Decimal(0))
        means.append(total / Decimal(len(data)))
    return tuple(means)


def bootstrap_mean_interval(
    values: Iterable[Decimal], *, seed: int, samples: int
) -> tuple[Decimal, Decimal]:
    means = bootstrap_means(values, seed=seed, samples=samples)
    return _quantile(means, _LOWER_QUANTILE), _quantile(means, _UPPER_QUANTILE)


def _resample_digest(means: Iterable[Decimal]) -> str:
    digest = hashlib.sha256()
    for value in means:
        digest.update(format(value, "f").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resample_digest(values: Iterable[Decimal], *, seed: int, samples: int) -> str:
    """重放抽样并返回均值序列摘要，用于审计核对分析结果。"""

    return _resample_digest(bootstrap_means(values, seed=seed, samples=samples))


def _metric_result(
    metric: Metric,
    values: list[Decimal],
    *,
    missing: int,
    excluded: int,
    required: int,
    seed: int,
    samples: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": metric.kind,
        "count": len(values),
        "missing": missing,
        "excluded": excluded,
        "zero_count": sum(1 for value in values if value == 0),
        "required": required,
        "sufficient": len(values) >= required,
    }
    if not values:
        result.update({
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "sample_variance": None,
        })
        if metric.kind == "binary":
            result.update({
                "successes": 0,
                "proportion": None,
                "wilson_lower": None,
                "wilson_upper": None,
            })
        else:
            result.update({
                "bootstrap_lower": None,
                "bootstrap_upper": None,
                "seed": seed,
                "bootstrap_samples": samples,
                "resample_sha256": None,
            })
        return result
    result.update(summarize(values).as_dict())
    if metric.kind == "binary":
        successes = sum(int(value) for value in values)
        interval = wilson_interval(successes, len(values))
        result.update({
            "successes": successes,
            "proportion": successes / len(values),
            "wilson_lower": interval.lower,
            "wilson_upper": interval.upper,
        })
    else:
        means = bootstrap_means(values, seed=seed, samples=samples)
        result.update({
            "bootstrap_lower": format(_quantile(means, _LOWER_QUANTILE), "f"),
            "bootstrap_upper": format(_quantile(means, _UPPER_QUANTILE), "f"),
            "seed": seed,
            "bootstrap_samples": samples,
            "resample_sha256": _resample_digest(means),
        })
    return result


def _stratum_coverage(stratum_required: int, observed: int, excluded: int) -> dict[str, object]:
    included = observed - excluded
    if observed == 0:
        status = "missing"
    elif included == 0:
        status = "all_excluded"
    elif included < stratum_required:
        status = "shortfall"
    else:
        status = "ok"
    return {
        "required": stratum_required,
        "observed": observed,
        "excluded": excluded,
        "included": included,
        "missing": max(0, stratum_required - included),
        "complete": included >= stratum_required,
        "status": status,
    }


def analyze(protocol: Protocol, observations: Iterable[Observation]) -> dict[str, object]:
    all_observations = tuple(observations)
    strata: dict[str, dict[str, object]] = {}
    insufficient: list[dict[str, object]] = []
    included_total = 0
    excluded_total = 0
    missing_value_total = 0
    for stratum_index, stratum in enumerate(protocol.strata):
        observed_rows = [item for item in all_observations if item.stratum_key == stratum.key]
        included_rows = [item for item in observed_rows if item.excluded_reason is None]
        observed = len(observed_rows)
        excluded = observed - len(included_rows)
        included = len(included_rows)
        included_total += included
        excluded_total += excluded
        coverage = _stratum_coverage(stratum.required_trials, observed, excluded)
        status = coverage["status"]
        if status == "missing":
            insufficient.append({
                "condition": "stratum_missing",
                "stratum": stratum.key,
                "required": stratum.required_trials,
                "observed": 0,
                "included": 0,
            })
        elif status == "all_excluded":
            insufficient.append({
                "condition": "stratum_all_excluded",
                "stratum": stratum.key,
                "required": stratum.required_trials,
                "observed": observed,
                "excluded": excluded,
            })
        elif status == "shortfall":
            insufficient.append({
                "condition": "coverage_shortfall",
                "stratum": stratum.key,
                "required": stratum.required_trials,
                "included": included,
                "missing": coverage["missing"],
            })
        metrics: dict[str, object] = {}
        for metric_index, metric in enumerate(protocol.metrics):
            valid_values: list[Decimal] = []
            missing_values = 0
            for item in included_rows:
                value = item.metrics[metric.key]
                if value is None:
                    missing_values += 1
                else:
                    valid_values.append(value)
            missing_value_total += missing_values
            metrics[metric.key] = _metric_result(
                metric,
                valid_values,
                missing=missing_values,
                excluded=excluded,
                required=stratum.required_trials,
                seed=protocol.seed + stratum_index * 1009 + metric_index,
                samples=protocol.bootstrap_samples,
            )
            valid_count = len(valid_values)
            if included > 0 and valid_count == 0:
                insufficient.append({
                    "condition": "metric_values_missing",
                    "stratum": stratum.key,
                    "metric": metric.key,
                    "included": included,
                    "missing_values": missing_values,
                })
            elif 0 < valid_count < stratum.required_trials:
                insufficient.append({
                    "condition": "metric_coverage_shortfall",
                    "stratum": stratum.key,
                    "metric": metric.key,
                    "required": stratum.required_trials,
                    "valid": valid_count,
                })
        strata[stratum.key] = {"label": stratum.label, "coverage": coverage, "metrics": metrics}

    aggregate: dict[str, object] = {}
    for metric in protocol.metrics:
        available = [
            (stratum.key, strata[stratum.key]["metrics"][metric.key])
            for stratum in protocol.strata
        ]
        if any(value["count"] == 0 for _, value in available):
            aggregate[metric.key] = {"available": False, "reason": "至少一个预注册分层无有效样本"}
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
        "observed_count": len(all_observations),
        "included_count": included_total,
        "excluded_count": excluded_total,
        "missing_value_count": missing_value_total,
        "strata": strata,
        "aggregate": aggregate,
        "rules": rule_results,
        "insufficient": insufficient,
        "conclusion": conclusion,
    }
