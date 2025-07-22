import json
import gc
import os
import torch
from transformers import AutoTokenizer


class TrainPipeline():
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name_or_path"])
        os.makedirs(self.config["output_dir"], exist_ok=True)
        self.metrics_log_path = os.path.join(self.config["output_dir"], "metrics_log.json")
        self.metrics_log = []

    def __call__(self):
        for task in self.config["tasks"]:
            print(f"\n[INFO] Running task: {task.upper()}")

            if task == "trigger_id":
                from train_steps.trigger_identification import run_trigger_identification
                result = run_trigger_identification(self.config, self.tokenizer)

            elif task == "trigger_cls":
                from train_steps.trigger_classification import run_trigger_classification
                result = run_trigger_classification(self.config, self.tokenizer)

            elif task == "arg_extraction":
                from train_steps.argument_extraction import run_argument_extraction
                result = run_argument_extraction(self.config, self.tokenizer)

            else:
                raise ValueError(f"Unknown task: {task}")

            self.log_metrics(task, result)
            self.cleanup()

        with open(self.metrics_log_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
        print(f"\n✅ Pipeline finished. Logs saved to {self.metrics_log_path}")

    def log_metrics(self, task, metrics):
        self.metrics_log.append({
            "task": task,
            "metrics": metrics
        })

    def cleanup(self):
        print("[INFO] Releasing memory...")
        torch.cuda.empty_cache()
        gc.collect()
