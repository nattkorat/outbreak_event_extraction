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

if __name__ == "__main__":
    
    # Example usage of the evaluation function with BLEURT
    from utils.semantic_similarity import calculate_bleurt

    predictions = [("DISEASE", "dengue outbreak"), ("LOCATION", "Phnom Penh")]
    golds = [("DISEASE", "dengue outbreak"), ("LOCATION", "Phnom Penh"), ("LOCATION", "Siem Reap")]

    results = evaluate_bleurt_seqeval(predictions, golds, calculate_bleurt)
    
    print(f"Evaluation Results:\n {results}")