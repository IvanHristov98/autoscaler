import logging
import numpy as np


def has_drift(expected, actual, buckets, bucket_type = 'bins'):
    psi_score = _calculate_psi_score(expected, actual, buckets, bucket_type)
    logging.info(f"detected psi score: {psi_score}")
    
    if psi_score >= 0.1:
        return True

    return False


def _calculate_psi_score(expected, actual, buckets, bucket_type = 'bins'):
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if bucket_type == 'bins':
        breakpoints = _scale_range(breakpoints, np.min(expected), np.max(expected))
    elif bucket_type == 'quantiles':
        breakpoints = np.stack([np.percentile(expected, b) for b in breakpoints])

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    psi_value = np.sum(_sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

    # psi_value < 0.1 - all good
    # psi_value < 0.2 - slight drift
    # psi_value > 0.2 - giant drift
    return psi_value


def _sub_psi(expected_percents, actual_percents):
    if actual_percents == 0:
        actual_percents = 0.0001
    if expected_percents == 0:
        expected_percents = 0.0001

    value = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    return(value)


def _scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input
