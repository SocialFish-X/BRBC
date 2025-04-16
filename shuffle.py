import json
import random

# 假设你的文件名为data.json
filename = 'data.json'

# 读取文件并加载数据
with open("shuffled_data.json", 'r', encoding='utf-8') as file:
    # 假设每个JSON对象占据一行
    lines = file.readlines()

# 将每一行解析为JSON对象
data = [json.loads(line) for line in lines]

# 设置随机种子
#random.seed(42)

# 打散数据
random.shuffle(data)

# 如果需要，可以将打散后的数据写回文件
with open('shuffled_data.json', 'w', encoding='utf-8') as file:
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')