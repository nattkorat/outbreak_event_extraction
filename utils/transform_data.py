import json

def data_transform(data: list, is_remove_non_event: bool = False):
    """
    Transform the parsed JSON data into a structured format for EE.
    
    Args:
        data (list): List of dictionaries containing event data.
        is_remove_non_event (bool): Flag to remove non-event data. Defaults to False.
    
    Returns:
        list: Transformed data in a structured format.
    """
    transformed_data = []
    for item in data:
        tempt = {
            'id': item['id'],
            'text': item['text'],
            'events': []
        }
        if item.get('events') is None:
            continue
        for event in item.get('events', []):
            for e in item['events'][event]:
                event_info = {
                    'event_type': event,
                    'trigger': {'text': e['trigger']},
                    'arguments': [
                        {'role': k, 'text': v}
                        for k, v in e['arguments'].items()
                    ]
                }
                tempt['events'].append(event_info)
        if is_remove_non_event:
            if not tempt['events']:
                continue
        transformed_data.append(tempt)
    return transformed_data

def read_jsonl(file_path):
    """
    Read a JSONL file and return its contents as a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSONL file.
        
    Returns:
        list: List of dictionaries representing the JSONL data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def write_jsonl(data, file_path):
    """
    Write a list of dictionaries to a JSONL file.
    
    Args:
        data (list): List of dictionaries to write.
        file_path (str): Path to the output JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"[INFO] Transformed data written to {file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform JSONL data for event extraction.")
    parser.add_argument('--file_path', type=str, help='Path to the JSONL file to transform.')
    parser.add_argument('--output_path', type=str, help='Path to save the transformed JSONL file.', default='transformed_data.jsonl')
    parser.add_argument('--is_remove_non_event', action='store_true', default=False, help='Flag to remove non-event data.')
    args = parser.parse_args()
    data = read_jsonl(args.file_path)
    transformed_data = data_transform(data, args.is_remove_non_event)
    write_jsonl(transformed_data, args.output_path)