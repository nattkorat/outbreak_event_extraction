from utils import evaluation, semantic_similarity
from tabulate import tabulate
from collections import defaultdict


def run(gold_data, pred_data, threshold=0.5, matching="strict"):
    # Build ID-based lookup
    gold_dict = {item["id"]: item["events"] for item in gold_data}
    pred_dict = {item["id"]: item["events"] for item in pred_data}

    common_ids = sorted(set(gold_dict.keys()) & set(pred_dict.keys()))
    if not common_ids:
        raise ValueError("No overlapping IDs between gold and prediction.")

    rows = []
    sum_type_p = sum_type_r = sum_type_f1 = 0
    sum_trigger_p = sum_trigger_r = sum_trigger_f1 = 0
    sum_argument_p = sum_argument_r = sum_argument_f1 = 0

    semantic_fn = semantic_similarity.stricty_compare 
    if matching == "bleu":
        semantic_fn = semantic_similarity.calculate_bleu
    elif matching == "rouge":
        semantic_fn = semantic_similarity.calculate_rouge
    elif matching == "bleurt":
        semantic_fn = semantic_similarity.calculate_bleurt

    for idx, sample_id in enumerate(common_ids):
        print(f"\n[INFO] Evaluating sample {idx+1}/{len(common_ids)} with ID: {sample_id}")

        gold_events = gold_dict[sample_id]
        pred_events = pred_dict[sample_id]

        # Type of event evaluation
        type_score = evaluation.evaluate_event_types(gold_events, pred_events)

        # Trigger evaluation
        trigger_scores = evaluation.evaluate_event_triggers(
            gold_events, pred_events,
            semantic_fn=semantic_fn,
            threshold=threshold
        )

        # Argument evaluation
        argument_scores = evaluation.evaluate_event_arguments(
            gold_events, pred_events,
            semantic_fn=semantic_fn,
            threshold=threshold
        )

        # Aggregate trigger scores
        trigger_tp = sum(s['tp'] for s in trigger_scores.values())
        trigger_fp = sum(s['fp'] for s in trigger_scores.values())
        trigger_fn = sum(s['fn'] for s in trigger_scores.values())
        trigger_precision = trigger_tp / (trigger_tp + trigger_fp + 1e-8)
        trigger_recall = trigger_tp / (trigger_tp + trigger_fn + 1e-8)
        trigger_f1 = 2 * trigger_precision * trigger_recall / (trigger_precision + trigger_recall + 1e-8)

        # Aggregate argument scores
        argument_tp = argument_fp = argument_fn = 0
        for roles_list in argument_scores.values():
            for role_scores in roles_list:
                for s in role_scores.values():
                    argument_tp += s['tp']
                    argument_fp += s['fp']
                    argument_fn += s['fn']

        argument_precision = argument_tp / (argument_tp + argument_fp + 1e-8)
        argument_recall = argument_tp / (argument_tp + argument_fn + 1e-8)
        argument_f1 = 2 * argument_precision * argument_recall / (argument_precision + argument_recall + 1e-8)

        row = [
            sample_id,
            type_score['precision'], type_score['recall'], type_score['f1'],
            trigger_precision, trigger_recall, trigger_f1,
            argument_precision, argument_recall, argument_f1
        ]
        rows.append(row)

        sum_type_p += type_score['precision']
        sum_type_r += type_score['recall']
        sum_type_f1 += type_score['f1']
        sum_trigger_p += trigger_precision
        sum_trigger_r += trigger_recall
        sum_trigger_f1 += trigger_f1
        sum_argument_p += argument_precision
        sum_argument_r += argument_recall
        sum_argument_f1 += argument_f1

        print("-- Trigger Evaluation by Type --")
        for t, s in trigger_scores.items():
            print(f"[{t}] P={s['precision']:.2f}, R={s['recall']:.2f}, F1={s['f1']:.2f}, TP={s['tp']}, FP={s['fp']}, FN={s['fn']}")

        print("-- Argument Evaluation by Type and Role --")
        for t, roles_list in argument_scores.items():
            for role_scores in roles_list:
                for role, s in role_scores.items():
                    print(f"[{t}::{role}] P={s['precision']:.2f}, R={s['recall']:.2f}, F1={s['f1']:.2f}, TP={s['tp']}, FP={s['fp']}, FN={s['fn']}")

    # Print overall table
    headers = [
        "ID",
        "Type_P", "Type_R", "Type_F1",
        "Trig_P", "Trig_R", "Trig_F1",
        "Arg_P", "Arg_R", "Arg_F1"
    ]
    print("\n===== Evaluation Results =====")
    print(tabulate(rows, headers=headers, floatfmt=".2f"))

    n = len(common_ids)
    print("\n===== Average Results =====")
    print(tabulate([[
        "Avg",
        sum_type_p / n, sum_type_r / n, sum_type_f1 / n,
        sum_trigger_p / n, sum_trigger_r / n, sum_trigger_f1 / n,
        sum_argument_p / n, sum_argument_r / n, sum_argument_f1 / n
    ]], headers=headers, floatfmt=".2f"))

if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate event extraction results.")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold data JSON file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction data JSON file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Semantic similarity threshold for matching.")
    parser.add_argument("--matching", type=str, choices=["strict", "bleu", "rouge", "bleurt"], default="strict",
                        help="Method for semantic matching.")

    args = parser.parse_args()

    with open(args.gold, 'r') as f:
        gold_data = json.load(f)
    
    with open(args.pred, 'r') as f:
        pred_data = json.load(f)

    run(gold_data, pred_data, threshold=args.threshold, matching=args.matching)