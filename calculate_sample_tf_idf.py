from __future__ import print_function
from __future__ import division
from data_structure import dim_item

sample_file = open("/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/sample_dim_items.txt")
train_file = open("/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/train_dim_items.txt")
sample = []
for line in sample_file.readlines():
    line = line.strip('\n')
    tokens = line.split()
    if len(tokens) == 3:
        sample.append(dim_item(tokens[0], tokens[1], tokens[2]))
train = []
for line in sample_file.readlines():
    line = line.strip('\n')
    tokens = line.split()
    if len(tokens) == 3:
        sample.append(dim_item(tokens[0], tokens[1], tokens[2]))

for item in sample:
    cat_items = filter(lambda x: x.cat_id == item.cat_id, train)
    item.cat_calculate_tf_idf_without_dict(cat_items)
    with open("/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/sample_dim_item_tf_idf.txt", 'a+') as f:
        format_str = item.item_id + ' ' + item.cat_id + ' ' + ",".join(item.terms) + ' ' + ",".join(
            map(lambda x: str(x), item.tf_idf)) + "\n"
        f.write(format_str)
        f.close()