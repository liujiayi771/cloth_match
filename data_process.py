sample_name = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/sample_items.txt"
file_name = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/dim_items.txt"
sample_file = open(sample_name)
process_file = open(file_name)

sample = []
for line in sample_file.readlines():
    line = line.strip('\n')
    sample.append(line)

for line in process_file.readlines():
    tokens = line.split(' ')
    if tokens[0] in sample:
        with open("/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/sample_dim_items.txt", 'a+') as f:
            f.write(line)
            f.close()


# sample_name = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/sample_matchsets.txt"
# file_name = "/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/dim_fashion_matchsets.txt"
# sample_file = open(sample_name)
# file = open(file_name)
#
# sample = []
# for line in sample_file.readlines():
#     line = line.strip('\n')
#     sample.append(line)
#
# for line in file.readlines():
#     tokens = line.split(' ')
#     if tokens[0] in sample:
#         with open("/home/joey/Documents/data/tianchi_dapei/Taobao_Clothes_Matching_Data/4000_sample/sample_dim_fashion_matchsets.txt", 'a+') as f:
#             f.write(line)
#             f.close()