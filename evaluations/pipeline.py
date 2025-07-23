import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from evaluations.e2e import run
from datetime import datetime
from tabulate import tabulate
from collections import Counter
from collections import defaultdict

from utils import semantic_similarity
from utils.evaluation import evaluate_bleurt_seqeval


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def trigger_identification(model, tokenizer, data, label_map):
    model.eval()
    all_trigger_spans = []
    for sample in data:
        tokens = sample["tokens"]
        enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        word_ids = enc.word_ids()
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = torch.argmax(logits, dim=2).squeeze().tolist()
        pred_labels = [label_map[pid] for pid in pred_ids]

        # Align labels to original words
        aligned_labels = []
        seen_word_ids = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen_word_ids:
                continue
            aligned_labels.append(pred_labels[i])
            seen_word_ids.add(wid)

        # Extract trigger spans
        spans = []
        i = 0
        while i < len(aligned_labels):
            if aligned_labels[i] == "B-TRIGGER":
                start = i
                end = i + 1
                while end < len(aligned_labels) and aligned_labels[end] == "I-TRIGGER":
                    end += 1
                spans.append((start, end))
                i = end
            else:
                i += 1
        all_trigger_spans.append(spans)
    return all_trigger_spans


def trigger_classification(model, tokenizer, data, trigger_spans, id2label):
    model.eval()
    all_event_types = []
    for sample, spans in zip(data, trigger_spans):
        types = []
        tokens = sample["tokens"]
        for start, end in spans:
            marked = tokens[:start] + ["<TRIGGER>"] + tokens[start:end] + ["</TRIGGER>"] + tokens[end:]
            enc = tokenizer(marked, is_split_into_words=True, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**enc).logits
            pred = torch.argmax(logits, dim=-1).item()
            types.append(id2label[pred])
        all_event_types.append(types)
    return all_event_types


def argument_extraction(model, tokenizer, data, trigger_spans, event_types, id2label):
    model.eval()
    all_events = []
    for sample, spans, types in zip(data, trigger_spans, event_types):
        tokens = sample["tokens"]
        sample_events = []

        for (start, end), evt_type in zip(spans, types):
            marked = tokens[:start] + ["<TRIGGER>"] + tokens[start:end] + ["</TRIGGER>"] + tokens[end:]
            enc = tokenizer(marked, is_split_into_words=True, return_tensors="pt", truncation=True)
            word_ids = enc.word_ids()

            with torch.no_grad():
                logits = model(**enc).logits
            pred_ids = torch.argmax(logits, dim=2).squeeze().tolist()
            pred_labels = [id2label[pid] for pid in pred_ids]

            # Align to original token indices (exclude marker)
            argument_spans = []
            current_role = None
            current_span = []

            seen_word_ids = set()
            for i, wid in enumerate(word_ids):
                if wid is None or wid in seen_word_ids or wid >= len(tokens):
                    continue

                seen_word_ids.add(wid)
                label = pred_labels[i]

                if label.startswith("B-"):
                    if current_span:
                        argument_spans.append((current_role, current_span))
                    current_role = label[2:]
                    current_span = [wid]
                elif label.startswith("I-") and current_role == label[2:]:
                    current_span.append(wid)
                else:
                    if current_span:
                        argument_spans.append((current_role, current_span))
                        current_span = []
                        current_role = None
            if current_span:
                argument_spans.append((current_role, current_span))

            arguments = []
            for role, span in argument_spans:
                s, e = span[0], span[-1] + 1
                arguments.append({
                    "role": role,
                    "start": s,
                    "end": e,
                    "text": "".join(tokens[s:e])
                })

            sample_events.append({
                "event_type": evt_type,
                "trigger": {
                    "start": start,
                    "end": end,
                    "text": "".join(tokens[start:end])
                },
                "arguments": arguments
            })
        all_events.append({"id": sample["id"], "events": sample_events})
    return all_events


def event_type_eval(golds, preds):
            gold_event_types = [e['event_type'] for e in golds]
            pred_event_types = [e['event_type'] for e in preds]

            gold_counts = Counter(gold_event_types)
            pred_counts = Counter(pred_event_types)

            # True Positives: predicted types that correctly match gold types (can appear multiple times)
            tp = sum(min(pred_counts[etype], gold_counts[etype]) for etype in pred_counts)

            fp = sum(pred_counts[etype] for etype in pred_counts) - tp
            fn = sum(gold_counts[etype] for etype in gold_counts) - tp

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

def evaluate_event_triggers(gold_events, pred_events, semantic_fn, threshold=0.9):
    gold_triggers = []
    pred_triggers = []

    for ev in gold_events:
        if "trigger" in ev and ev["trigger"]:
            gold_triggers.append((ev["event_type"], ev["trigger"]['text']))

    for ev in pred_events:
        if "trigger" in ev and ev["trigger"]:
            pred_triggers.append((ev["event_type"], ev["trigger"]['text']))

    return evaluate_bleurt_seqeval(pred_triggers, gold_triggers, semantic_fn, threshold)


def evaluate_event_arguments(gold_events, pred_events, semantic_fn, threshold=0.9):
    all_scores = defaultdict(list)

    used_pred_indices = set()

    for g_event in gold_events:
        g_type = g_event["event_type"]
        g_trigger = g_event["trigger"]['text']
        g_args = g_event["arguments"]

        for idx, p_event in enumerate(pred_events):
            if idx in used_pred_indices:
                continue

            if p_event["event_type"] != g_type:
                continue

            p_trigger = p_event["trigger"]['text']
            if semantic_fn(g_trigger, p_trigger) >= threshold:
                # Consider arguments a list of dicts with 'role', 'start', 'end', 'text'
                pred_arg_pairs = [(arg["role"], arg['text']) for arg in p_event["arguments"]]
                gold_arg_pairs = [(arg["role"], arg['text']) for arg in g_args]

                arg_scores = evaluate_bleurt_seqeval(
                    predictions=pred_arg_pairs,
                    golds=gold_arg_pairs,
                    semantic_fn=semantic_fn,
                    threshold=threshold
                )
                all_scores[g_type].append(arg_scores)
                used_pred_indices.add(idx)
                break  # Assume 1-to-1 match for event by type and trigger

    return dict(all_scores)


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
        type_score = event_type_eval(gold_events, pred_events)

        # Trigger evaluation
        trigger_scores = evaluate_event_triggers(
            gold_events, pred_events,
            semantic_fn=semantic_fn,
            threshold=threshold
        )

        # Argument evaluation
        argument_scores = evaluate_event_arguments(
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

def main(config_path):
    config = load_config(config_path)
    raw_data = load_jsonl(config["test_dataset_path"])

    print("[1] Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained(config["trigger_model"])
    trigger_model = AutoModelForTokenClassification.from_pretrained(config["trigger_model"])
    trigger_cls_model = AutoModelForSequenceClassification.from_pretrained(config["trigger_classifier"])
    argument_model = AutoModelForTokenClassification.from_pretrained(config["argument_model"])

    trigger_id2label = trigger_model.config.id2label
    trigger_cls_id2label = trigger_cls_model.config.id2label
    argument_id2label = argument_model.config.id2label

    print("[2] Predicting Triggers...")
    trigger_spans = trigger_identification(trigger_model, tokenizer, raw_data, trigger_id2label)

    print("[3] Classifying Event Types...")
    event_types = trigger_classification(trigger_cls_model, tokenizer, raw_data, trigger_spans, trigger_cls_id2label)

    print("[4] Extracting Arguments...")
    predictions = argument_extraction(argument_model, tokenizer, raw_data, trigger_spans, event_types, argument_id2label)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], "evaluation_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Predictions saved to {output_file}")

    print("[6] Evaluating...")

    log_file = os.path.join(config["output_dir"], f"eval_log_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        # Redirect print to log file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        run(gold_data=raw_data, pred_data=predictions, threshold=config.get("threshold", 0.5), matching=config.get("matching", "strict"))
        sys.stdout = original_stdout

    print(f"[INFO] Evaluation log saved to {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
