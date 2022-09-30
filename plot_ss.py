import csv 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help="root path containing results")
    args = parser.parse_args()
    results = {}
    dirs = ['mu_w_acsreg', 'mu_wo_acsreg', 'var_w_acsreg', 'var_wo_acsreg']
    names = ['cart_pos', 'cart_vel', 'pole_pos', 'pole_vel']
   
    for dir in dirs:
        results[dir] = {}

        path = args.data_root + '/' + dir + '/'
        for name in names:
            results[dir][name] = {}
            file = path + name + '.csv'
            # print(file)
            with open(file, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for fieldname in fieldnames:
                    results[dir][name][fieldname] = []
                for row in reader:
                    for key in row:
                 
                        results[dir][name][key].append(float(row[key]))


    for i, dir in enumerate(dirs):
        fig, axs = plt.subplots(1)
    
        for idx, name in enumerate(names):
            axs.plot('Step', 'Value', data=results[dir][name], label=name)
        axs.legend(loc='best', shadow=True, framealpha=1)    
        
        plt.savefig(dir + '.jpg')
            
                    
                
    
    
