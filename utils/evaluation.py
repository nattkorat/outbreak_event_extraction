from collections import defaultdict

def evaluate_bleurt_seqeval(predictions, golds, semantic_fn, threshold=0.9):
    """
    predictions and golds are lists of (label, span_text) tuples
    e.g., [("DISEASE", "dengue outbreak"), ("LOCATION", "Phnom Penh")]
    """
    pred_by_type = defaultdict(list)
    gold_by_type = defaultdict(list)
    
    for label, span in predictions:
        pred_by_type[label].append(span)
    
    for label, span in golds:
        gold_by_type[label].append(span)

    results = {}
    
    for label in set(pred_by_type.keys()).union(gold_by_type.keys()):
        preds = pred_by_type[label]
        refs = gold_by_type[label]
        
        matched = [False] * len(refs)
        tp = 0
        fp = 0
        
        for pred in preds:
            found_match = False
            for i, ref in enumerate(refs):
                if not matched[i] and semantic_fn(pred, ref) >= threshold:
                    matched[i] = True
                    found_match = True
                    tp += 1
                    break
            if not found_match:
                fp += 1

        fn = matched.count(False)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        
    return results

def evaluate_event_types(gold, pred):
    gold_types = set(gold.keys())
    pred_types = set(pred.keys())
    
    tp = len(gold_types & pred_types)
    fp = len(pred_types - gold_types)
    fn = len(gold_types - pred_types)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def evaluate_event_triggers(gold, pred, semantic_fn, threshold=0.9):
    gold_triggers = []
    pred_triggers = []

    for event_type, events in gold.items():
        for ev in events:
            gold_triggers.append((event_type, ev["trigger"]))
    
    for event_type, events in pred.items():
        for ev in events:
            pred_triggers.append((event_type, ev["trigger"]))
    
    return evaluate_bleurt_seqeval(pred_triggers, gold_triggers, semantic_fn, threshold)

def evaluate_event_arguments(gold, pred, semantic_fn, threshold=0.9):
    all_scores = defaultdict(list)

    for event_type in gold:
        if event_type not in pred:
            continue
        
        for g_event in gold[event_type]:
            g_trigger = g_event["trigger"]
            g_args = g_event["arguments"]

            for p_event in pred[event_type]:
                p_trigger = p_event["trigger"]
                if semantic_fn(g_trigger, p_trigger) >= threshold:
                    pred_args = p_event["arguments"]
                    pred_arg_pairs = list(pred_args.items())
                    gold_arg_pairs = list(g_args.items())

                    arg_scores = evaluate_bleurt_seqeval(
                        predictions=pred_arg_pairs,
                        golds=gold_arg_pairs,
                        semantic_fn=semantic_fn,
                        threshold=threshold
                    )
                    all_scores[event_type].append(arg_scores)
                    break

    return dict(all_scores)

if __name__ == "__main__":
    
    # Example usage of the evaluation function with BLEURT
    from utils.semantic_similarity import calculate_bleurt

    predictions = [("DISEASE", "dengue outbreak"), ("LOCATION", "Phnom Penh")]
    golds = [("DISEASE", "dengue outbreak"), ("LOCATION", "Phnom Penh"), ("LOCATION", "Siem Reap")]

    results = evaluate_bleurt_seqeval(predictions, golds, calculate_bleurt)
    
    print(f"Evaluation Results:\n {results}")