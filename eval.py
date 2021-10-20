import sys
import os
import pickle

cwd = os.getcwd()
print(cwd)

folder_name = './data/output_a2c_t_dORCA_DO_001'
# file_name = 'all_rewards.data'

# path = os.path.join(folder_name, file_name)
# # print(path)
# my_l = []
# with open(path, 'rb') as f:
#     my_l = pickle.load(f)

# print(my_l)
# print(len(my_l))

data_path = os.path.join(folder_name,'key_data.data')
with open(data_path, 'rb') as f:
    # store the data as binary data stream
    key_data = pickle.load(f)
all_lengths = key_data[0]
average_lengths = key_data[1]
all_rewards = key_data[2]
entropy_term = key_data[3]

print(all_rewards)