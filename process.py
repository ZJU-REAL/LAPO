import json

input_file = 'all_mapping_file'
output_file = 'clean_mapping_file'

prompt_dict = {}

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            prompt = data.get('prompt')
            if prompt is not None:
                prompt_dict[prompt] = data
        except json.JSONDecodeError:
            continue

with open(output_file, 'w', encoding='utf-8') as f_out:
    for data in prompt_dict.values():
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

final_file = 'final_mapping_file'
q2len = {}

with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            prompt = data["prompt"]
            median_length = data["median_length"]
            q2len[prompt] = median_length
        except Exception as e:
            print(f"Error in line: {line.strip()} | Exception: {e}")

with open(final_file, 'w', encoding='utf-8') as f:
    json.dump(q2len, f, ensure_ascii=False, indent=2)
