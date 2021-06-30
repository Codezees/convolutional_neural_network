import convolution_nural_network as co_nn
import numpy as np
import os,sys

def data_files(folder_path):
    files = os.listdir(folder_path)
    total_files = len(files)
    full_path_fill = [f'{folder_path}/{file}' for file in files] 
    return total_files, full_path_fill

def getfile_values(file_path):
    file = open(file_path, 'r')
    data = file.read().split('\n')[:-1]
    file.close()
    return [[int(d) for d in da.split(',')] for da in data]

print('Getting files...')

folder = './classImagesData'
num_files, files_path = data_files(folder)

print('Getting data form the file...')

num_files = 3

labels, data_set = [], []
for i in range(num_files):
    data = []
    data = getfile_values(files_path[i])
    for d in data:
        labels.append(i)
        data_set.append(d)
    
    print(f'>> No of Files Extracted:{i+1}', end='\r')
    
print('\nData Extraction complete.\n')    

nn = co_nn.convolutional_neural_network(epochs=2, hidden_layers=100, output_layer=10, activation_func=['relu', 'softmax'],data_set=data_set, data_labels=labels,  train_data_set=100, test_data_set=15, randamize=True)

del num_files,files_path,labels, data_set, data

nn.neuron_train() 
