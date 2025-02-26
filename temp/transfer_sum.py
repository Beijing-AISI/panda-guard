import json
import csv


# 读取 JSON 文件
def process_json_file(json_file_path, output_csv_path):
    # 打开并读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 用于存储结果的列表
    results = []

    # 遍历 JSON 中的 results 数组
    for result in data['results']:
        goal = result['goal']  # 获取目标描述
        for data_item in result['data']:
            attacker = data_item['usage']['attacker']
            prompt_tokens = attacker['prompt_tokens']
            completion_tokens = attacker['completion_tokens']
            total_tokens = prompt_tokens + completion_tokens

            # 将结果添加到列表
            results.append({
                'goal': goal,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            })

    # 将结果写入 CSV 文件
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['goal', 'prompt_tokens', 'completion_tokens', 'total_tokens']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入数据
        for result in results:
            writer.writerow(result)


# 使用示例
json_file_path = '/home/shensicheng/log/jailbreak_log/meta-llama_Meta-Llama-3.1-8B-Instruct/RandomSearchAttacker_RandomSearch/NoneDefender/results.json'  # 替换为你的 JSON 文件路径
output_csv_path = '/home/shensicheng/code/jailbreak-pipeline/data/AttackResults/token_usage/output.csv'  # 输出 CSV 文件路径
process_json_file(json_file_path, output_csv_path)
print(f"处理完成，结果已保存到 {output_csv_path}")