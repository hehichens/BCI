import os                                                                                                                                                                                                                                                          
import pandas as pd
from utils.options import opt
from utils import utils


## test all model
def run(method, filename='data_all'):
    print("="*40, "information", "="*40)
    print("feature method: {}".format(method))
    data_dir = os.path.join("./Datasets/", filename)
    data_path = os.path.join(data_dir, filename + '_' + method + '.csv')
    df = pd.read_csv(data_path)
    print("data shape: ", df.shape)

    result = utils.test_model(df)
    
    result_dir = os.path.join('./result', filename)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    csv_path = os.path.join(result_dir, filename + '_' + method + '_result' + '.csv')
    utils.save_csv(result, csv_path)

    for i in range(10):
        print()


if __name__ == "__main__":
    # method 'csp', 'bandpowers', 'dct', 'wavelet', 'all'
    for method in ['csp', 'bandpowers', 'dct', 'wavelet', 'all']:
        run(method)
    
