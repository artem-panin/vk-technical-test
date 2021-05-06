import numpy as np


def apk(actual, predicted, k=10):
    """
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted
    predicted : list
                A list of predicted elements
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
    predicted : list
                A list of lists of predicted elements
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    # return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
    res = []
    for a, p in zip(actual, predicted):
        if len(a) <= len(p):
            res.append(apk(a, p, k))
        else:
            for i in range(k, len(a), k):
                res.append(apk(a[i:i+k], p, k))
    return np.mean(res)

