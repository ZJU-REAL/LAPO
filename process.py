import json

# --- Configuration ---
# Input file from Stage 1 training
raw_mapping_file = 'all_mapping_file'
# Final output file for Stage 2 training
final_mapping_file = 'final_mapping_file'


def process_mapping_data(input_path, output_path):
    """
    Reads a raw, line-delimited JSON file, deduplicates entries based on the 'prompt',
    and creates a clean, final mapping of {prompt: median_length}.
    """
    print(f"Reading raw data from: {input_path}")
    
    # Use a dictionary to store the latest entry for each unique prompt
    unique_prompt_data = {}
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    prompt = data.get('prompt')
                    if prompt:
                        unique_prompt_data[prompt] = data
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'. Please check the path.")
        return

    print(f"Found {len(unique_prompt_data)} unique prompts. Processing them...")

    # Directly process the deduplicated data to create the final mapping
    final_q2len_map = {}
    for data in unique_prompt_data.values():
        try:
            prompt = data["prompt"]
            median_length = data["median_length"]
            final_q2len_map[prompt] = median_length
        except KeyError as e:
            print(f"Warning: Skipping entry due to missing key {e}. Data: {data}")

    # Write the final, simplified dictionary to the output file
    print(f"Writing final processed map to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_q2len_map, f, ensure_ascii=False, indent=2)

    print("✅ Processing complete.")


if __name__ == "__main__":
    process_mapping_data(raw_mapping_file, final_mapping_file)
