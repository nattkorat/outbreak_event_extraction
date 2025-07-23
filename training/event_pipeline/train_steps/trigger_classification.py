import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer
)
import evaluate
import numpy as np

def run_trigger_classification(config, tokenizer):
    accuracy = evaluate.load("accuracy")

    # Load and process data
    with open(config["dataset_path"], "r") as f:
        data = [json.loads(line) for line in f]

    def make_trigger_cls_dataset(data):
        samples = []
        for item in data:
            tokens = item["tokens"]
            for event in item["events"]:
                start, end = event["trigger"]["start"], event["trigger"]["end"]
                if start != -1 and end != -1:
                    marked = tokens.copy()
                    marked.insert(end, "</TRIGGER>")
                    marked.insert(start, "<TRIGGER>")
                    samples.append({
                        "tokens": marked,
                        "event_type": event["event_type"]
                    })
        return samples

    samples = make_trigger_cls_dataset(data)
    event_types = sorted(set(s["event_type"] for s in samples))
    event2id = {ev: i for i, ev in enumerate(event_types)}
    id2event = {i: ev for ev, i in event2id.items()}

    for s in samples:
        s["label"] = event2id[s["event_type"]]

    def tokenize_fn(ex):
        enc = tokenizer(ex["tokens"], truncation=True, is_split_into_words=True)
        enc["label"] = ex["label"]
        return enc

    ds = Dataset.from_list(samples).train_test_split(test_size=0.2, seed=42)
    tokenized = ds.map(tokenize_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"],
        num_labels=len(event2id),
        id2label=id2event,
        label2id=event2id
    )

    args = TrainingArguments(
        output_dir=f"{config['output_dir']}/trigger_cls",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        logging_dir=f"{config['output_dir']}/logs",
        push_to_hub=config["push_to_hub"]
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()

    return trainer.evaluate()

def predict_event_types(trigger_results, config, tokenizer):
    from transformers import AutoModelForSequenceClassification
    import torch.nn.functional as F

    model = AutoModelForSequenceClassification.from_pretrained(
        f"{config['output_dir']}/trigger_cls"
    ).eval()

    event_type_preds = []

    for ex in trigger_results:
        tokens = ex["tokens"]
        trig = ex["trigger"]
        if not trig:
            event_type_preds.append(None)
            continue
        marked = tokens.copy()
        marked.insert(trig[1], "</TRIGGER>")
        marked.insert(trig[0], "<TRIGGER>")
        encoding = tokenizer(marked, return_tensors="pt", truncation=True, is_split_into_words=True)
        with torch.no_grad():
            logits = model(**encoding).logits
            pred_id = torch.argmax(logits, dim=1).item()
            label = model.config.id2label[pred_id]
        event_type_preds.append(label)

    return event_type_preds
