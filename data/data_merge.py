"""
merge csv data
"""
import pandas as pd 
import os


save_dir = './Datasets/data_all/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

all_df = pd.DataFrame()
for method in ['csp', 'bandpowers', 'dct', 'wavelet']:
    ## concat all data
    method_df = pd.DataFrame()
    for data_name in ['aa', 'al', 'av', 'aw', 'ay']:
        filename = 'data_set_IVa_{}'.format(data_name)
        data_dir = os.path.join("./Datasets/", filename)
        data_path = os.path.join(data_dir, filename + '_' + method + '.csv')
        df = pd.read_csv(data_path)
        method_df = pd.concat([method_df, df])
    
    ## save single method
    save_path = os.path.join(save_dir, 'data_all_'+method+'.csv')
    print("data shape", method_df.shape)
    print("==> save file at: ", save_path)
    print()
    method_df.to_csv(save_path, index=False)


    ## concat feature
    all_df = pd.concat([all_df, method_df], axis=1)

## save all method with concat
save_path = os.path.join(save_dir, 'data_all_all.csv')
all_df.to_csv(save_path, index=False)
print("data shape", all_df.shape)
print("==> save file at: ", save_path)
print()