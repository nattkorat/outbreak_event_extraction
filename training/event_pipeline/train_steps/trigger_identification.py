import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification, DataCollatorForTokenClassification,
    TrainingArguments, Trainer
)
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_trigger_identification(config, tokenizer):
    with open(config["dataset_path"], "r") as f:
        data = [json.loads(line) for line in f]

    def create_labels(ex):
        labels = ['O'] * len(ex["tokens"])
        for e in ex["events"]:
            start, end = e["trigger"]["start"], e["trigger"]["end"]
            if start != -1 and end != -1:
                labels[start] = "B-TRIGGER"
                for i in range(start + 1, end):
                    labels[i] = "I-TRIGGER"
        return labels

    examples = [{"tokens": d["tokens"], "labels": create_labels(d)} for d in data]

    label_list = ["O", "B-TRIGGER", "I-TRIGGER"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    def tokenize_and_align(example):
        encoding = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
        word_ids = encoding.word_ids()
        labels = []
        prev = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev:
                labels.append(label2id[example["labels"][word_id]])
            else:
                l = example["labels"][word_id]
                labels.append(label2id[l] if l.startswith("I-") else -100)
            prev = word_id
        encoding["labels"] = labels
        return encoding

    ds = Dataset.from_list(examples).train_test_split(test_size=0.2, seed=42)
    tokenized = ds.map(tokenize_and_align)

    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name_or_path"], num_labels=len(label_list),
        id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=f"{config['output_dir']}/trigger_id",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        logging_dir=f"{config['output_dir']}/logs"
    )

    def compute_metrics(eval_preds):
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
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()

    return trainer.evaluate()
