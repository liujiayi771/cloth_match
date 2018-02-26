from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from pyspark.sql import SparkSession
import datetime
import data_utils


def split_history_line(r):
    tokens = r.split()
    return tokens[0] + tokens[2], tokens[1]


if __name__ == "__main__":
    base_dir = "E:\\Project\\PycharmProjects\\cloth_match\\data"
    history_file = base_dir + "\\user_bought_history.txt"
    sample_file = base_dir + "\\test_items.txt"

    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("cloth_match") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    sc = spark.sparkContext

    file_name = "file:\\\\\\" + history_file

    # lines_rdd = sc.textFile(file_name).map(split_history_line)
    # lines_df = spark.createDataFrame(lines_rdd, ["user_id_create_at", "item_id"])
    # lines_df.show(20)
    # lines_df.write.parquet("user_bought_history")

    df = spark.read.parquet("user_bought_history")
    df.registerTempTable("user_bought_history")

    sample = data_utils.init_test_sample_from_filename(sample_file)
    start = datetime.datetime.now()
    for sample_item in sample[4325:]:
        one_collocation = {}
        result = spark.sql(
            f"SELECT item_id FROM user_bought_history WHERE user_id_create_at in \
            (SELECT user_id_create_at FROM user_bought_history WHERE item_id == {sample_item}) \
            AND item_id != {sample_item}"
        ).collect()
        for row in result:
            item_id = row.item_id
            if one_collocation.get(item_id) is None:
                one_collocation[item_id] = 1
            else:
                one_collocation[item_id] += 1
        result = sorted(one_collocation.items(), key=lambda x: x[1], reverse=True)
        if len(result) > 200:
            result = result[:200]

        with open("predict_history.txt", 'a+') as f:
            format_str = sample_item + ' ' + ",".join(map(lambda x: x[0], result)) + "\n"
            f.write(format_str)

    end = datetime.datetime.now()
    print(end - start)
