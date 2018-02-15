from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import math
import datetime
import data_utils

p = 4
alpha = 0.15
beta = 1


def map_function(n):
    return alpha * math.log(n, math.e) + beta


def calculate_collocation_degree(sample_item, item_match, item_sets):
    print("\nstart predix item: " + sample_item.item_id)
    result_dict = {}
    item_count = 1
    for internal_item in item_sets.values():
        # print("\r", "item_count: " + str(item_count), end="")
        item_count += 1
        if item_match.get(internal_item.item_id) is not None:
            n_item = len(internal_item.terms)
            res = 0
            for x in item_match[internal_item.item_id]:
                if item_sets.get(x) is not None:
                    numerator = data_utils.calculate_similarity(item_sets.get(x), sample_item)
                    denominator = map_function(len(item_sets.get(x).terms))
                    res += math.pow(numerator / denominator, p)
            result_dict[internal_item.item_id] = map_function(n_item) * math.pow(res, 1 / p)

    result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    print(result[:10])
    if len(result) > 200:
        result = result[:200]

    with open("predict_matchsets.txt", 'a+') as f:
        format_str = sample_item.item_id + ' ' + ",".join(map(lambda x: x[0], result)) + "\n"
        f.write(format_str)


if __name__ == "__main__":
    base_dir = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample"
    sample_file = base_dir + "/test.txt"
    match_file = base_dir + "/train_dim_fashion_matchsets.txt"
    item_file = base_dir + "/train_dim_item_tf_idf.txt"
    item_match_file = base_dir + "/train_item_match.txt"

    sample_list = data_utils.init_dim_items_from_filename(sample_file, with_tf_idf=True)
    match_list = data_utils.init_dim_fashion_match_sets_from_filename(match_file)
    item_dict = data_utils.init_item_dict_from_filename(item_file, with_tf_idf=True)
    item_match_dict = data_utils.init_item_match_dict_from_filename(item_match_file)

    """
    item_match_sets = {}
    count = 1
    start = datetime.datetime.now()
    for item in item_dict.values():
        print("\r", "map building:" + str(count), end="")
        count += 1
        for match in match_list:
            if match.find_item(item):
                match_item = match.get_match_item(item)
                if item_match_sets.get(item.item_id) is None:
                    item_match_sets[item.item_id] = match_item
                else:
                    item_match_sets[item.item_id] += match_item
        with open("train_item_match.txt", 'a+') as f:
            if item_match_sets.get(item.item_id) is not None:
                f.write(item.item_id + ' ' + ",".join(item_match_sets[item.item_id]) + '\n')
    end = datetime.datetime.now()
    print("\nmap building time: " + str(end - start))
    """

    for sample in sample_list:
        calculate_collocation_degree(sample, item_match_dict, item_dict)
