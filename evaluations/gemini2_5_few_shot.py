import argparse
import json
import time
from utils import prompt, response_parser, evaluation, semantic_similarity
from llms import gemini2_5
from tabulate import tabulate
from collections import defaultdict


def run(test_data, few_shot_samples, threshold=0.5, sleep_time=5):
    rows = []
    
    sum_type_p = sum_type_r = sum_type_f1 = 0
    sum_trigger_p = sum_trigger_r = sum_trigger_f1 = 0
    sum_argument_p = sum_argument_r = sum_argument_f1 = 0
    all_type_scores = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for i, example in enumerate(test_data):
        print(f"\n[INFO] Processing article {i+1}/{len(test_data)}")
        print(f"[INFO] Article : {example['text']}")
        # Build prompt
        prompt_text = prompt.event_extraction_prompt_few_shot(
            article=example['text'],
            samples=few_shot_samples
        )

        # Request LLM (with sleep)
        llm_raw_response = gemini2_5.chat_with_gemini2_5(prompt_text)
        time.sleep(sleep_time)

        # Parse response to structured format
        llm_response = response_parser.json_string_response_parser(llm_raw_response)

        # Evaluate
        gold = example['events']
        pred = llm_response

        type_score = evaluation.evaluate_event_types(gold, pred)
        trigger_scores = evaluation.evaluate_event_triggers(
            gold, pred,
            semantic_fn=semantic_similarity.calculate_bleu,
            threshold=threshold
        )
        argument_scores = evaluation.evaluate_event_arguments(
            gold, pred,
            semantic_fn=semantic_similarity.calculate_bleu,
            threshold=threshold
        )

        # Average trigger scores across all event types
        trigger_tp = sum(s['tp'] for s in trigger_scores.values())
        trigger_fp = sum(s['fp'] for s in trigger_scores.values())
        trigger_fn = sum(s['fn'] for s in trigger_scores.values())

        trigger_precision = trigger_tp / (trigger_tp + trigger_fp + 1e-8)
        trigger_recall = trigger_tp / (trigger_tp + trigger_fn + 1e-8)
        trigger_f1 = 2 * trigger_precision * trigger_recall / (trigger_precision + trigger_recall + 1e-8)

        # Average argument scores across all event types and roles
        argument_tp = argument_fp = argument_fn = 0
        for event_type, roles_list in argument_scores.items():
            for role_scores in roles_list:
                for role, s in role_scores.items():
                    argument_tp += s['tp']
                    argument_fp += s['fp']
                    argument_fn += s['fn']

        argument_precision = argument_tp / (argument_tp + argument_fp + 1e-8)
        argument_recall = argument_tp / (argument_tp + argument_fn + 1e-8)
        argument_f1 = 2 * argument_precision * argument_recall / (argument_precision + argument_recall + 1e-8)

        row = [
            i + 1,
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

        # Print per-type trigger scores
        print("-- Trigger Evaluation by Type --")
        for t, s in trigger_scores.items():
            print(f"[{t}] P={s['precision']:.2f}, R={s['recall']:.2f}, F1={s['f1']:.2f}, TP={s['tp']}, FP={s['fp']}, FN={s['fn']}")

        # Print per-type and per-role argument scores
        print("-- Argument Evaluation by Type and Role --")
        for t, roles_list in argument_scores.items():
            for role_scores in roles_list:
                for role, s in role_scores.items():
                    print(f"[{t}::{role}] P={s['precision']:.2f}, R={s['recall']:.2f}, F1={s['f1']:.2f}, TP={s['tp']}, FP={s['fp']}, FN={s['fn']}")

    # Display results as a table
    headers = [
        "ID",
        "Type_P", "Type_R", "Type_F1",
        "Trig_P", "Trig_R", "Trig_F1",
        "Arg_P", "Arg_R", "Arg_F1"
    ]
    print("\n===== Evaluation Results =====")
    print(tabulate(rows, headers=headers, floatfmt=".2f"))
    
    n = len(test_data)
    print("\n===== Average Results =====")
    print(tabulate([
        [
            "Avg",
            sum_type_p / n, sum_type_r / n, sum_type_f1 / n,
            sum_trigger_p / n, sum_trigger_r / n, sum_trigger_f1 / n,
            sum_argument_p / n, sum_argument_r / n, sum_argument_f1 / n
        ]
    ], headers=headers, floatfmt=".2f"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate event extraction from LLM output.")
    parser.add_argument("--input", type=str, required=True, help="Path to test file (JSON)")
    parser.add_argument("--sample", type=str, required=True, help="Path to few-shot template file (JSON)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
    parser.add_argument("--sleep", type=int, default=5, help="Sleep time between LLM requests (in seconds)")

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        test_data = json.load(f)

    with open(args.sample, 'r') as f:
        few_shot_samples = json.load(f)

    run(test_data, few_shot_samples, threshold=args.threshold, sleep_time=args.sleep)
