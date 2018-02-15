from data_utils import init_history_dict_from_filename, init_item_id_from_filename

history_file  = "user_bought_history.txt"
sample_file =   "sample_items.txt"

history_dict = data_utils.init_history_dict_from_filename(history_filename)
sample = data_utils.init_dim_items_from_filename(sample_file)

all_collocation = {}
for sample_item in sample:
	one_collocation = {}
	for key,value in history_dict.items():
		if sample_item in value:
			for item_id in history_dict[key]:
				if not one_collocation.has_key(item_id):
					one_collocation[item_id] = 1
				else :
					one_collocation[item_id] += 1

	result = sorted(one_collocation.items(), key=lambda x: x[1], reverse=True)
	if len(result) > 200:
        result = result[:200]

    with open("predict_matchsets.txt", 'a+') as f:
    	format_str = sample_item + ' ' + ",".join(map(lambda x: x[0], result)) + "\n"
    	f.write(format_str)
	

