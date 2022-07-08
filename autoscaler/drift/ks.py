import bisect
import logging
import math


def has_drift(expected, actual, alpha_level = 0.05) -> bool:
    scores = _calculate_ks_scores(expected, actual, alpha_level)
    
    return scores["D"] >= scores["p_value"]


def _calculate_ks_scores(expected, actual, alpha_level = 0.05):
    D = 0

    alpha_values = {
        0.01: 1.63,
        0.05: 1.36,
        0.1: 1.22,
        0.15: 1.14,
        0.2: 1.07,
    }

    values = expected + actual
    values = sorted(list(set(values)))

    expected_sorted = sorted(expected)
    actual_sorted = sorted(actual)

    for value in values:
        expected_index = bisect.bisect(expected_sorted, value)
        actual_index = bisect.bisect(actual_sorted, value)

        d = abs(float(expected_index - actual_index) / len(expected_sorted))
        D = max(d, D)
        logging.debug(f"Kolmogorov-Smirnov: expected idx: {expected_index}, actual idx: {actual_index}, d: {d}")

    p_value = alpha_values[alpha_level] / math.sqrt(len(expected_sorted))

    # if D < p_value we are all good
    return {"D": D, "p_value": p_value}
