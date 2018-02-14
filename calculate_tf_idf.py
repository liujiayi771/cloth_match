from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import datetime
import data_utils
from pyspark import SparkConf, SparkContext


def calculate_tf_idf(group):
    print('\n')
    print("cat_id: " + group[0] + "  cat_size: " + str(len(group[1])))

    group_items = group[1]

    cat_dict = {}
    for item in group_items:
        for term in item.terms:
            if term not in cat_dict.keys():
                cat_dict[term] = set()
                cat_dict[term].add(item)
            else:
                cat_dict[term].add(item)

    for item in group_items:
        data_utils.cat_calculate_tf_idf(item, group_items, cat_dict)

        with open("dim_item_tf_idf.txt", 'a+') as f:
            format_str = item.item_id + ' ' + item.cat_id + ' ' + ",".join(item.terms) + ' ' + ",".join(
                map(lambda x: str(x), item.tf_idf)) + "\n"
            f.write(format_str)

    return group_items


if __name__ == "__main__":
    conf = SparkConf().setMaster("local[5]")
    sc = SparkContext(appName="cloth_matches", conf=conf)
    sc.addPyFile("file:///home/joey/Documents/projects/PycharmProjects/cloth_match/data_structure.py")

    file_name = "file:///home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/dim_items.txt"

    start = datetime.datetime.now()

    lines = sc.textFile(file_name, minPartitions=30)
    lines.map(data_utils.init_dim_item_from_input_line) \
        .filter(lambda x: x is not None) \
        .groupBy(lambda x: x.cat_id) \
        .flatMap(calculate_tf_idf) \
        .count()
    sc.stop()

    end = datetime.datetime.now()
    print(end - start)
