"""Paired statistical helpers; pairing is always performed by question_id."""
from __future__ import annotations

import math
from statistics import mean, stdev

try:
    from scipy import stats
except ImportError:  # pragma: no cover - reported by CLI when inference is requested
    stats = None


def align_pairs(rows, left, right, metric):
    """Inner-align non-null observations by question ID, rejecting duplicates."""
    indexes = {}
    for configuration in (left, right):
        index = {}
        for row in rows:
            if row.get("configuration_id") != configuration or row.get(metric) is None:
                continue
            qid = row["question_id"]
            if qid in index: raise ValueError(f"Duplicate pair key: {configuration}/{qid}")
            index[qid] = float(row[metric])
        indexes[configuration] = index
    ids = sorted(set(indexes[left]) & set(indexes[right]))
    return ids, [indexes[left][qid] for qid in ids], [indexes[right][qid] for qid in ids]


def holm_adjust(p_values):
    """Holm step-down adjustment, preserving input positions and monotonicity."""
    adjusted = [None] * len(p_values); running = 0.0
    for rank, (position, value) in enumerate(sorted(enumerate(p_values), key=lambda x: x[1])):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[position] = running
    return adjusted


def continuous(rows, left, right, metric):
    ids, a, b = align_pairs(rows, left, right, metric)
    if not ids: return None
    differences = [x-y for x, y in zip(a, b)]; n = len(ids)
    sd = stdev(differences) if n > 1 else 0.0; se = sd / math.sqrt(n)
    critical = stats.t.ppf(.975, n-1) if stats is not None and n > 1 else 1.96
    t_p = float(stats.ttest_rel(a, b).pvalue) if stats is not None and n > 1 and sd else 1.0
    try: wilcoxon_p = float(stats.wilcoxon(a, b).pvalue) if stats is not None and any(differences) else 1.0
    except ValueError: wilcoxon_p = 1.0
    difference = mean(differences)
    return {"metric": metric, "n": n, "question_ids": ids, "left_mean": mean(a), "right_mean": mean(b),
            "mean_difference": difference, "standard_deviation": sd, "standard_error": se,
            "ci95_low": difference-critical*se, "ci95_high": difference+critical*se,
            "paired_t_p": t_p, "wilcoxon_p": wilcoxon_p,
            "cohens_d_paired": difference/sd if sd else (0.0 if difference == 0 else None)}


def binary(rows, left, right, metric):
    ids, a, b = align_pairs(rows, left, right, metric)
    if not ids: return None
    n = len(ids); left_success = sum(a); right_success = sum(b); discordant_left = sum(x == 1 and y == 0 for x,y in zip(a,b)); discordant_right = sum(x == 0 and y == 1 for x,y in zip(a,b))
    def wilson(success):
        z=1.96; p=success/n; center=(p+z*z/(2*n))/(1+z*z/n); half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return [center-half, center+half]
    mcnemar_p = float(stats.binomtest(min(discordant_left, discordant_right), discordant_left+discordant_right, .5).pvalue) if stats is not None and discordant_left+discordant_right else 1.0
    odds = (discordant_left+.5)/(discordant_right+.5)
    return {"metric":metric,"n":n,"question_ids":ids,"left_rate":left_success/n,"right_rate":right_success/n,
            "left_wilson95":wilson(left_success),"right_wilson95":wilson(right_success),"absolute_rate_difference":(left_success-right_success)/n,
            "mcnemar_p":mcnemar_p,"matched_pairs_odds_ratio":odds}
