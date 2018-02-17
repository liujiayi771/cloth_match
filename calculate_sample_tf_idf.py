from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import data_utils

base_dir = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data"
sample_file = base_dir + "/test_dim_items.txt"
train_file = base_dir + "/dim_items.txt"

sample = data_utils.init_dim_items_from_filename(sample_file)
train = data_utils.init_dim_items_from_filename(train_file)

for item in sample:
    cat_items = filter(lambda x: x.cat_id == item.cat_id, train)
    data_utils.cat_calculate_tf_idf_without_dict(item, cat_items)

    with open(base_dir + "/test_dim_item_tf_idf.txt", 'a+') as f:
        format_str = item.item_id + ' ' + item.cat_id + ' ' + ",".join(item.terms) + ' ' + ",".join(
            map(lambda x: str(x), item.tf_idf)) + "\n"
        f.write(format_str)