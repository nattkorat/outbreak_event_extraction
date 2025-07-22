import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification, DataCollatorForTokenClassification,
    TrainingArguments, Trainer
)
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_argument_extraction(config, tokenizer):
    with open(config["dataset_path"], "r") as f:
        data = [json.loads(line) for line in f]

    def create_argument_dataset(data):
        samples = []
        for item in data:
            tokens = item["tokens"]
            for event in item["events"]:
                start, end = event["trigger"]["start"], event["trigger"]["end"]
                if start != -1 and end != -1:
                    marked = tokens.copy()
                    marked.insert(end, "</TRIGGER>")
                    marked.insert(start, "<TRIGGER>")
                    labels = ["O"] * len(tokens)
                    for arg in event["arguments"]:
                        role = arg["role"].upper()
                        a_start, a_end = arg["start"], arg["end"]
                        if a_start != -1 and a_end != -1:
                            labels[a_start] = f"B-{role}"
                            for i in range(a_start + 1, a_end):
                                labels[i] = f"I-{role}"
                    labels.insert(end, "O")
                    labels.insert(start, "O")
                    samples.append({
                        "tokens": marked,
                        "labels": labels,
                        "event_type": event["event_type"]
                    })
        return samples

    samples = create_argument_dataset(data)

    def get_label_vocab(samples):
        label_set = set()
        for s in samples:
            label_set.update(s["labels"])
        label_list = sorted(label_set)
        return {l: i for i, l in enumerate(label_list)}, {i: l for i, l in enumerate(label_list)}

    label2id, id2label = get_label_vocab(samples)

    def tokenize_and_align(ex):
        encoding = tokenizer(ex["tokens"], is_split_into_words=True, truncation=True)
        word_ids = encoding.word_ids()
        labels = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word:
                labels.append(label2id[ex["labels"][word_id]])
            else:
                label = ex["labels"][word_id]
                labels.append(label2id[label] if label.startswith("I-") else -100)
            prev_word = word_id
        encoding["labels"] = labels
        return encoding

    ds = Dataset.from_list(samples).train_test_split(test_size=0.2, seed=42)
    tokenized = ds.map(tokenize_and_align)

    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name_or_path"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=f"{config['output_dir']}/arg_extraction",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        logging_dir=f"{config['output_dir']}/logs"
    )

    def compute_arg_metrics(eval_preds):
        predictions, labels = eval_preds
        preds = torch.argmax(torch.tensor(predictions), axis=2)
        true_labels, true_preds = [], []
        for pred, label in zip(preds, labels):
            cur_labels, cur_preds = [], []
            for p, l in zip(pred, label):
                if l != -100:
                    cur_labels.append(id2label[l.item()])
                    cur_preds.append(id2label[p.item()])
            true_labels.append(cur_labels)
            true_preds.append(cur_preds)
        return {
            "accuracy": accuracy_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_arg_metrics
    )

    trainer.train()
    trainer.save_model()

    return trainer.evaluate()
