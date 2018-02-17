from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import data_utils
from pyspark import SparkConf, SparkContext


# def calculate_collocation(rows, sample_list):
#     collocation_degree = []
#     for sample_item in sample_list:
#         for row in rows:



if __name__ == "__main__":
    base_dir = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data"
    history_file = base_dir + "/user_bought_history.txt"
    sample_file = base_dir + "/test_sample/test_dim_item_tf_idf.txt"

    conf = SparkConf().setMaster("local[5]")
    conf.set("spark.driver.memory", "2g")
    sc = SparkContext(appName="cloth_matches", conf=conf)
    sc.addPyFile("file:///home/joey/Documents/projects/PycharmProjects/cloth_match/data_structure.py")

    file_name = "file://" + history_file

    # sample = data_utils.init_dim_items_from_filename(sample_file)
    # sample_rdd = sc.broadcast(sample)

    lines_rdd = sc.textFile(file_name, minPartitions=30)

    user_bought_list = lines_rdd.map(data_utils.init_user_bought_history_from_input_line)\
        .collect()
    print(len(user_bought_list))