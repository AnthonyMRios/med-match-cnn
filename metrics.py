import numpy as np

def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    try:
        pos_label = unique_y[1]
    except:
        return 0.
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    #return float(n_relevant) / min(n_pos, k)
    return float(n_relevant) / float(k)

def ranking_recall_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    try:
        pos_label = unique_y[1]
    except:
        return 0.
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    total = np.sum(y_true) 
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / float(n_pos)

def pak(y_true, y_score, k=10):
    paks = []
    for t, p in zip(y_true, y_score):
        paks.append(ranking_precision_score(t, p, k))
    return np.mean(paks)

def rak(y_true, y_score, k=10):
    raks = []
    for t, p in zip(y_true, y_score):
        raks.append(ranking_recall_score(t, p, k))
    return np.mean(raks)
