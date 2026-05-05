def _score_value(passage, names, default=0.0):
    for name in names:
        if name in passage and passage[name] is not None:
            return float(passage[name])
    return default


def normalize_min_max(values):
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    if span <= 1e-12:
        return [1.0 for _ in values]
    return [(value - min_value) / span for value in values]


def select_standard_rag_passages(passages, top_k):
    passages = [dict(passage) for passage in passages]
    passages.sort(
        key=lambda p: _score_value(p, ["score_topic", "topic_score", "retrieval_score", "score_topic_orig"]),
        reverse=True,
    )
    return passages[:top_k]


def select_frag_passages(passages, top_k, alpha=0.6, normalize_topic=True):
    passages = [dict(passage) for passage in passages]
    topic_scores = [
        _score_value(p, ["score_topic", "topic_score", "retrieval_score", "score_topic_orig"])
        for p in passages
    ]
    factuality_scores = [
        _score_value(p, ["score_factuality", "factuality_score", "fact_score"])
        for p in passages
    ]
    if normalize_topic:
        topic_scores = normalize_min_max(topic_scores)

    for passage, topic_score, factuality_score in zip(passages, topic_scores, factuality_scores):
        passage["score_topic_normalized"] = topic_score
        passage["score_factuality_used"] = factuality_score
        passage["score_frag"] = alpha * topic_score + (1.0 - alpha) * factuality_score

    passages.sort(key=lambda p: p["score_frag"], reverse=True)
    return passages[:top_k]
