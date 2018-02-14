from __future__ import print_function
from __future__ import division
import math
import datetime
from data_structure import dim_item, dim_fashion_match_sets
from pyspark import SparkConf, SparkContext

p = 4
alpha = 0.15
beta = 1


def initialize_from_input_line(line):
    tokens = line.split()
    if len(tokens) == 4:
        return dim_item(tokens[0], tokens[1], tokens[2], tokens[3])
    else:
        return None


def map_function(n):
    return alpha * math.log(n, math.e) + beta


def calculate_collocation_degree(sample_item, item_match, item_sets):
    print("\nstart predix item: " + sample_item.item_id)
    result_dict = {}
    item_count = 1
    for item in item_sets.values():
        print("\r", "item_count: " + str(item_count), end="")
        item_count += 1
        n_item = len(item.terms)
        res = 0
        for x in item_match[item.item_id]:
            if item_sets.get(x) is not None:
                res += math.pow(item_sets.get(x).similarity(sample_item) / map_function(n_item), p)
        result_dict[item.item_id] = map_function(n_item) * math.pow(res, 1 / p)

    print("\nresult_dict_size: " + str(len(result_dict)))
    result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    if len(result) > 200:
        result = result[:200]
    with open("predict_matchsets.txt", 'a+') as f:
        format_str = sample_item.item_id + ' ' + ",".join(map(lambda x: x[0].item_id, result)) + "\n"
        f.write(format_str)
        f.close()


if __name__ == "__main__":

    sample_file = open(
        "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/sample_dim_item_tf_idf.txt")
    match_file = open(
        "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/train_dim_fashion_matchsets.txt")
    item_file = open(
        "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/train_dim_item_tf_idf.txt")
    sample_list = []
    match_list = []
    item_list = {}
    for line in sample_file.readlines():
        line = line.strip('\n')
        tokens = line.split()
        if len(tokens) == 4:
            sample_list.append(dim_item(tokens[0], tokens[1], tokens[2], tokens[3]))
    for line in match_file.readlines():
        line = line.strip('\n')
        tokens = line.split()
        if len(tokens) == 2:
            match_list.append(dim_fashion_match_sets(tokens[0], tokens[1]))
    for line in item_file.readlines():
        line = line.strip('\n')
        tokens = line.split()
        if len(tokens) == 4:
            item_list[tokens[0]] = dim_item(tokens[0], tokens[1], tokens[2], tokens[3])

    item_match_sets = {}
    count = 1
    start = datetime.datetime.now()
    for item in item_list.values():
        print("\r", "map building:" + str(count), end="")
        count += 1
        for match in match_list:
            if match.find_item(item):
                match_item = match.get_match_item(item)
                item_match_sets[item.item_id] = match_list
    end = datetime.datetime.now()
    print("map building time: " + str(end - start))
    for sample in sample_list:
        calculate_collocation_degree(sample, item_match_sets, item_list)
