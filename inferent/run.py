import os
import time
import json
from utils import prompt, response_parser


def run_inference(
        test_data: list,
        model_fn,
        few_shot_samples: list = None,
        sleep_time: int = 5,
        mode: str = "few",
        output_file: str = "predictions.jsonl"
    ):
    """
    Run inference using a given model function.

    Parameters:
        - test_data: list of dicts, each with 'id' and 'text'
        - model_fn: function to call the model (e.g., gemini2_5.chat_with_gemini2_5)
        - few_shot_samples: list of examples for prompting (optional for zero-shot)
        - sleep_time: int, seconds to wait between requests
        - mode: "few" or "zero"
        - output_file: path to write predictions (jsonl)
    """
    
    relaxing = 0

    for i, example in enumerate(test_data):
        print(f"\n[INFO] Inferring article {i+1}/{len(test_data)} (ID: {example.get('id')})")
        print(f"[INFO] Article Text: {example['text']}")

        # 1. Construct prompt
        if mode == "few":
            prompt_text = prompt.event_extraction_prompt_few_shot(
                article=example['text'],
                samples=few_shot_samples
            )
        elif mode == "zero":
            prompt_text = prompt.event_extraction_prompt_zero_shot(
                article=example['text']
            )
        else:
            raise ValueError("Invalid mode: choose either 'few' or 'zero'")

        # 2. Query model
        raw_response = model_fn(prompt_text)
        time.sleep(sleep_time)
        relaxing += 1
        if relaxing % 10 == 0:
            print(f"[INFO] Relaxing for 60 seconds to avoid rate limits.")
            time.sleep(60)
            relaxing = 0

        # 3. Parse JSON-like response
        structured_output = response_parser.json_string_response_parser(raw_response)

        # Add predictions
        example["events"] = structured_output
        
        print(f"[INFO] Response:\n{structured_output}")
        
        # Write to file
        output_path = "outputs"
        os.makedirs(output_path, exist_ok=True)
        
        output_file_path = os.path.join(output_path, output_file)
        with open(output_file_path, 'a') as f:
            f.write(f"{json.dumps(example, ensure_ascii=False)}\n")

if __name__ == '__main__':
    import argparse
    from llms import (
        gemini2_5,
        llama4_maverik,
        qwen2_5_72b,
        deepseekr1
    )

    parser = argparse.ArgumentParser(description="Run event extraction inference.")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--few_shot_samples", type=str, default=None)
    parser.add_argument("--sleep_time", type=int, default=5)
    parser.add_argument("--mode", type=str, choices=["few", "zero"], default="few")
    parser.add_argument("--output_file", type=str, default="predictions.jsonl")
    parser.add_argument("--model", type=str, default="gemini", choices=[
        "gemini",
        "llama4",
        "qwen2_5_72b",
        "deepseekr1"
        ])

    args = parser.parse_args()

    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    few_shot_samples = None
    if args.few_shot_samples:
        with open(args.few_shot_samples, 'r') as f:
            few_shot_samples = json.load(f)

    # Map model names to functions
    model_map = {
        "gemini": gemini2_5.chat_with_gemini2_5,
        "llama4": llama4_maverik.chat_with_llama4_maverik,
        "qwen2_5_72b": qwen2_5_72b.chat_with_qwen2_5_72b,
        "deepseekr1": deepseekr1.chat_with_deepseekr1
    }

    run_inference(
        test_data=test_data[-2:],
        model_fn=model_map[args.model],
        few_shot_samples=few_shot_samples,
        sleep_time=args.sleep_time,
        mode=args.mode,
        output_file=args.output_file
    )
