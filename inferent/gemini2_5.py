import time
from utils import prompt, response_parser
from llms import gemini2_5


def infer_with_gemini(test_data, few_shot_samples=None, sleep_time=5, mode="few"):
    """
    Run Gemini inference on test_data.

    Parameters:
        - test_data: list of dicts, each with 'id' and 'text'
        - few_shot_samples: list of examples for prompting (optional for zero-shot)
        - sleep_time: int, seconds to wait between requests
        - mode: "few" or "zero"
    
    Returns:
        - List of dicts with 'id' and 'events' as output
    """
    predictions = []

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

        # 2. Query Gemini
        raw_response = gemini2_5.chat_with_gemini2_5(prompt_text)
        time.sleep(sleep_time)

        # 3. Parse JSON-like response
        structured_output = response_parser.json_string_response_parser(raw_response)

        # 4. Store prediction with ID
        predictions.append({
            "id": example.get("id", str(i)),
            "events": structured_output
        })
        
        print(f"[INFO] Response:\n{structured_output}")

    return predictions

if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Run event extraction inference with Gemini 2.5.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data JSON file.")
    parser.add_argument("--few_shot_samples", type=str, default=None, help="Path to few-shot samples JSON file for few-shot (optional).")
    parser.add_argument("--sleep_time", type=int, default=5, help="Sleep time between requests in seconds.")
    parser.add_argument("--mode", type=str, choices=["few", "zero"], default="few", help="Inference mode: 'few' for few-shot or 'zero' for zero-shot.")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="Output file to save predictions.")

    args = parser.parse_args()

    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    # Load few-shot samples if provided
    few_shot_samples = None
    if args.few_shot_samples:
        with open(args.few_shot_samples, 'r') as f:
            few_shot_samples = json.load(f)

    # Run inference
    predictions = infer_with_gemini(test_data, few_shot_samples, args.sleep_time, args.mode)

    # Output predictions
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
