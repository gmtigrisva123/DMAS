"""Agreement and classification metrics.

Raw accuracy is the wrong tool with 28 mostly rare labels (always predicting
the majority class looks good), so agreement is reported chance corrected
(Cohen's kappa, Krippendorff's alpha) and per class (macro averages).
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


@dataclass
class ClassificationReport:
    """Multi label agreement between the system and the reference labels."""

    support: int = 0
    exact_match: float = 0.0
    hamming_accuracy: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    macro_f1: float = 0.0
    top1_accuracy: float = 0.0
    kappa: float = 0.0
    per_label: Dict[str, Tuple[float, float, float, int]] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"n={self.support} top1={self.top1_accuracy:.3f} micro-F1={self.micro_f1:.3f} "
            f"macro-F1={self.macro_f1:.3f} kappa={self.kappa:.3f}"
        )


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two single label annotators.
    kappa = (p_o - p_e) / (1 - p_e), p_e = agreement expected from the
    marginals. Returns 1.0 when both raters are constant and identical (the
    usual degenerate case).
    """
    if len(a) != len(b):
        raise ValueError("annotations must have equal length")
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b[label] for label in set(counts_a) | set(counts_b)) / (n * n)
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if observed >= 1.0 - 1e-12 else 0.0
    return (observed - expected) / (1.0 - expected)


def krippendorff_alpha(annotations: Sequence[Sequence[str]]) -> float:
    """Krippendorff's alpha, nominal data, any number of raters.
    alpha = 1 - D_o / D_e from the coincidence matrix. Unlike kappa it handles
    missing values (None) and more than two raters.
    """
    units = [[label for label in unit if label is not None] for unit in annotations]
    units = [unit for unit in units if len(unit) >= 2]
    if not units:
        return 0.0

    coincidence: Counter = Counter()
    total = 0.0
    for unit in units:
        m = len(unit)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coincidence[(unit[i], unit[j])] += 1.0 / (m - 1)
                total += 1.0 / (m - 1)
    if total <= 0:
        return 0.0

    labels = sorted({label for unit in units for label in unit})
    marginals = {label: sum(coincidence[(label, other)] for other in labels) for label in labels}

    observed_disagreement = sum(
        coincidence[(x, y)] for x in labels for y in labels if x != y
    )
    expected_disagreement = sum(
        marginals[x] * marginals[y] for x in labels for y in labels if x != y
    ) / max(total - 1.0, 1e-12)
    if expected_disagreement <= 0:
        return 1.0 if observed_disagreement <= 0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def multilabel_report(
    predictions: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    *,
    top_choice: Sequence[str] = (),
) -> ClassificationReport:
    """Precision / recall / F1 plus chance corrected agreement.
    top_choice optionally gives each item's single best prediction, scored
    against "any gold label" (a tutor acts on one diagnosis at a time).
    """
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold must align")
    n = len(gold)
    report = ClassificationReport(support=n)
    if n == 0:
        return report

    labels: Set[str] = set()
    for row in list(predictions) + list(gold):
        labels.update(row)

    tp = fp = fn = 0
    exact = 0
    hamming = 0.0
    per_label_counts: Dict[str, List[int]] = {label: [0, 0, 0] for label in labels}
    for predicted, actual in zip(predictions, gold):
        p_set, g_set = set(predicted), set(actual)
        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
        exact += 1 if p_set == g_set else 0
        if labels:
            agreeing = sum(1 for label in labels if (label in p_set) == (label in g_set))
            hamming += agreeing / len(labels)
        for label in labels:
            if label in p_set and label in g_set:
                per_label_counts[label][0] += 1
            elif label in p_set:
                per_label_counts[label][1] += 1
            elif label in g_set:
                per_label_counts[label][2] += 1

    report.exact_match = exact / n
    report.hamming_accuracy = hamming / n
    report.micro_precision = tp / (tp + fp) if tp + fp else 0.0
    report.micro_recall = tp / (tp + fn) if tp + fn else 0.0
    report.micro_f1 = _f1(report.micro_precision, report.micro_recall)

    f1s: List[float] = []
    for label, (label_tp, label_fp, label_fn) in per_label_counts.items():
        support = label_tp + label_fn
        precision = label_tp / (label_tp + label_fp) if label_tp + label_fp else 0.0
        recall = label_tp / support if support else 0.0
        f1 = _f1(precision, recall)
        report.per_label[label] = (precision, recall, f1, support)
        if support:
            f1s.append(f1)
    report.macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    if top_choice:
        report.top1_accuracy = sum(
            1 for choice, actual in zip(top_choice, gold) if choice and choice in set(actual)
        ) / n
        # kappa on the single label view: top choice vs the gold label it matches
        # (or the first gold label if none)
        system_labels = [choice or "(none)" for choice in top_choice]
        human_labels = [
            (choice if choice in set(actual) else (sorted(actual)[0] if actual else "(none)"))
            for choice, actual in zip(top_choice, gold)
        ]
        report.kappa = cohen_kappa(system_labels, human_labels)
    return report


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
