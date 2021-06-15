import numpy as np
import os, sys, time

class convolutional_neural_network:

    def __init__(self, data_set, data_labels, output_nuron:int, activation_func:list, epochs:int=1, hidden_layers:int=10, train_data_set:int=100, test_data_set:int=10, layer_structure='norm'):
        '''
            Convolutional_Neural_Network
            ===========================

            Author:- Raint Saha(Code_name:-Codzees)

            Author_github_link :- 

            project_link:- 
            

            Description:- This class "nural_network" is made to train convolutional nural network with ease.


            Methods In it :-
            ----------------

            It contains 7 methods in total. The methods it contains are:-
            1) data_arange
            2) layer_desend
            3) layer_density
            4) save_model
            5) nuron_activation
            6) backward_prop
            7) forward_prop
            8) nuron_train

        '''
        self.data_set = np.array(data_set)
        self.data_labels = np.array(data_labels)

        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.output_nuron = output_nuron
        self.activation_func = activation_func

        self.train_Data_amount = train_data_set
        self.test_Data_set = test_data_set

        self.layer_structure = layer_structure
      
        self.train_Data = []
        self.train_label = []
        self.test_Data = []
        self.test_label = []
        
        self.data_shape = ''
        self.train_data_shape = ''
        self.test_data_shape = ''

        self.weights = []
        self.biases = []
        
        self.unactivated_nu = []
        self.activated_nu = []
       
    def data_arange(self):
        '''
            data_arange()
            ============= 

            This method "data_arange" arranges or splits the data into training and testing sets. By default is will split the data 80% into training and the rest 20% for testing purposes.

            It takes no arguments.
        '''
        print('Arranging/Restructuring the dataset given')
        self.train_Data = self.data_set[:self.train_Data_amount]
        self.train_label = self.data_labels[:self.train_Data_amount]
        self.test_Data = self.data_set[self.train_Data_amount:self.train_Data_amount+self.test_Data_set]
        self.test_label = self.data_labels[self.train_Data_amount : self.train_Data_amount+self.test_Data_set]

        self.data_shape = self.data_set.shape
        self.train_data_shape = self.train_Data.shape
        self.test_data_shape = self.test_Data.shape
        
    def layer_desend(self):
        _, w = self.train_data_shape
        mat = (w - self.output_nuron)/self.hidden_layers
        return w, mat

    def layer_density(self, net_shape='norm'):
        print('Creating the layers of nural network')
        w, inc = self.layer_desend()
        if net_shape == 'norm':
            for i in range(self.hidden_layers):
                
                if inc < 0.01:
                    next_lop = round(w - (inc + 1)) 
                else:
                    next_lop = round(w - inc) 

                # this sets the layer density to tihe output layer number if the density decreaser from the output number
                if (next_lop < self.output_nuron) or (i == self.hidden_layers-2):
                    next_lop = self.output_nuron
                # print(w, next_lop)

                self.weights.append(0.10*np.random.randn(w, next_lop))
                if i == 0:
                    self.biases.append(np.zeros((1, next_lop)))
                elif i > 0:
                    self.biases.append(0.10*np.random.randn(1, next_lop))

                w = next_lop

        else:
            print(f'Unvalid parameter "{net_shape}" given at net_shape')
            sys.exit()

    def save_model(self, file_name=None, type='json'):
        '''
            save_model()
            ============

            This method "save_model" saves the model data in a file. the file name can be provided or the file name will be the current datetime value.  
        '''

        if file_name == None:
            current_time = time.strftime('%A_%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
            file_name = f'{current_time}.txt'

    def nuron_activation(self, inputs, func_type=None):
        if func_type == "softmax":
            return np.exp(inputs)/ np.sum(np.exp(inputs))

        elif func_type == 'Relu':
            return np.maximum(0, inputs)
        
        elif func_type == 'one_hot':
            onne = np.zeros((inputs.size, self.output_nuron))
            onne[np.arange(inputs.size), inputs] = 1
            return onne

        elif func_type == 'Relu_deriv':
            return inputs > 0

        else:
            print(f'fuc_typ needs a valid activation function name. Given {func_type}')
            sys.exit()

    def backward_prop(self, func_type=None):
        if func_type == 'softmax':
            pass
        else:
            print(f'The function given for the backwords propogaetion "{func_type}" is not valid')

    def forward_prop(self):

        '''
            forward_prop()
            ==============

            This method "forward_prop" is the function the handles the forward propagation part of the nural network.

        '''
        
        self.unactivated_nu = []
        self.activated_nu = []

        input = self.train_Data
        for i in range(self.hidden_layers):

            if len(self.activation_func) > 2:
                # couldn't figure out how to do this.
                pass
            else:
                if i == (self.hidden_layers - 1):
                    func_type = self.activation_func[1]
                else:
                    func_type = self.activation_func[0]

            nu = np.dot(input, self.weights[i]) + self.biases[i]

            self.unactivated_nu.append(nu)
            self.activated_nu.append(self.nuron_activation(inputs=nu, func_type=func_type) )

            input = nu
            print(f'Layer no:- {(i + 1)} / [{round(((i+1)/self.hidden_layers)*100)}% |{ "="*round(((i+1)/self.hidden_layers)*100)}{" "*(100 - round(((i)/self.hidden_layers)*100) )}| 100%]', end='\r')
        
        print('\n')

    def nuron_train(self):
        '''
            nuron_train()
            =============

            This function "nuron_train" is the main function the is used to train the convolutional nural network.

        '''
        self.data_arange()
        self.layer_density(net_shape=self.layer_structure)

        print('Training started....')
        train_start = time.time()
        for epo in range(self.epochs):
            epo_start = time.time()
            print(f'Epoch =\t{epo+1}/{self.epochs}\t\tRemaining epoch = {self.epochs - (epo + 1)}')
            
            self.forward_prop()
            # self.back_prop(func_type)

            print(f'Accuracy: Unknown\t\t Error: Unknown\t\t Time_Taken: {round(time.time() - epo_start)} sec\n')

        print(f'Total time taken for training: {round(time.time() - train_start)} sec(s).') 
        print('Training Ended.\n')