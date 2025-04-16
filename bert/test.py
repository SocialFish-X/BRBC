import tokenization

split_tokens = []
basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab='/media/lenovo/SoftwareAndData/xuemengwu/2024Learning/Entity-Relation-Extraction-XMW/pretrained_model/chinese_L-12_H-768_A-12/vocab.txt')
for token in basic_tokenizer.tokenize('“ 超级搜索器”（Super Searcher）最初由英国梅尔设备公司开发研制，工作在X频段（8～12.5GHz）。它与Sea Searcher雷达相比，探测距离更远。'):
    for sub_token in wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)
print(split_tokens)
