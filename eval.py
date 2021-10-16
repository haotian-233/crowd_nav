import sys
import os
import pickle

cwd = os.getcwd()
print(cwd)

folder_name = './data/output_a2c_t_dORCA_DO_001'
file_name = 'all_rewards.data'

path = os.path.join(folder_name, file_name)
# print(path)
my_l = []
with open(path, 'rb') as f:
    my_l = pickle.load(f)

print(my_l)
print(len(my_l))