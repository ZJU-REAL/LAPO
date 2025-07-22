import json

input_file = 'stage1_save_file'
output_file = 'clean_file'

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
