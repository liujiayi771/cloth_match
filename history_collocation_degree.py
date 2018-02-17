import data_utils

base_dir = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data"
history_file = base_dir + "/user_bought_history.txt"
sample_file = base_dir + "/test_sample/test_dim_item_tf_idf.txt"

history_dict = data_utils.init_history_dict_from_filename(history_file)
sample = data_utils.init_dim_items_from_filename(sample_file)

all_collocation = {}
for sample_item in sample:
    one_collocation = {}
    for key, value in history_dict.items():
        if sample_item in value:
            for item_id in history_dict[key]:
                if one_collocation.get(item_id) is None:
                    one_collocation[item_id] = 1
                else:
                    one_collocation[item_id] += 1

    result = sorted(one_collocation.items(), key=lambda x: x[1], reverse=True)
    if len(result) > 200:
        result = result[:200]

    with open("predict_matchsets.txt", 'a+') as f:
        format_str = sample_item + ' ' + ",".join(map(lambda x: x[0], result)) + "\n"
        f.write(format_str)
