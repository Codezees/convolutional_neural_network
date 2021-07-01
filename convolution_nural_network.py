import numpy as np
import sys, time, random
from numpy.core.records import array

np.seterr(divide='ignore', invalid='ignore')

class convolutional_neural_network:

    def __init__(self, data_set, data_labels, activation_func:list, epochs:int=1, output_layer:int=None,hidden_layers:int=10, train_data_set:int=None, test_data_set:int=None, randamize:bool=False, learning_rate=0.01, layer_structure='norm'):
        '''
            Convolutional_Neural_Network
            ===========================

            Author:- Raint Saha(Code_name:-Codzees)

            Author_github_link :- https://github.com/Codezees

            Project_link:- https://github.com/Codezees/convolutional_neural_network
            

            Description:- This class "nural_network" is made to train convolutional nural network with ease.


            Methods In it :-
            ----------------

            It contains 7 methods in total. The methods it contains are:-
                - data_arange
                - layer_desend
                - layer_density
                - save_model
                - neuron_activation
                - backward_prop
                - forward_prop
                - neuron_train

            Note:-
            ------
            Initial parametes that is needed for training the neural network:-

                - data_set: Takes data in the form of a "list" and stored all the data needed to train and test the neural network.
                - data_labels : Takes all the labels of the data in the form of a "list" and stores it for train and testing purpose.
                - output_neuron : This is for setting the number of outputs of the neural network.
                - activation_func : Takes a list of activation functions that the user wants to use for training the neural network.
                - epochs : This sets the epochs/number times the network to train on the data.
                - hidden_layers : For setting the number of hidden layers in the  network.
                - train_data_set : This parameter sets the amount of data to be used for training the neural network.
                - test_data_set : This parameter is to set how many data points to be used for testing the model.
                - layer_structure : This parameter determines the shape of the neural network.

            Parameters
            ----------
            activation_func : {'relu', 'softmax'}
            
        '''

        self.data_set = np.array(data_set, dtype='float')
        self.data_labels = np.array(data_labels, dtype='float')

        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.randomize = randamize

        if output_layer == None: 
            self.output_neuron = len(np.unique(self.data_labels))
        else:
            self.output_neuron = output_layer

        self.train_Data_amount = train_data_set
        self.test_Data_set = test_data_set
        
        self.layer_structure = layer_structure
    
        self.train_Data, self.train_label = [],[]
        self.test_Data,self.test_label = [],[]
        
        self.train_data_shape,self.test_data_shape = '',''
        self.weights,self.biases = [],[]
        self.unactivated_nu,self.activated_nu = [],[]
        
    def data_arange(self):
        '''
            data_arange()
            ============= 

            This method "data_arange" arranges or splits the data into training and testing sets. By default is will split the data 80% into training and the rest 20% for testing purposes.

            It takes no arguments.
        '''
        print('Arranging/Restructuring the dataset given\n')
        data_len = len(self.data_set)

        if self.randomize:
            given_data = list(zip(self.data_set,self.data_labels))
            random.shuffle(given_data)
            self.data_set, self.data_labels = list(zip(*given_data))
            self.data_set, self.data_labels = np.array(self.data_set), np.array(self.data_labels)

        if self.train_Data_amount == None:
            self.train_Data = self.data_set[:int(data_len*0.8)]
            self.train_label = self.data_labels[:int(data_len*0.8)]
        else:
            self.train_Data = self.data_set[:self.train_Data_amount]
            self.train_label = self.data_labels[:self.train_Data_amount]
        
        if self.train_Data_amount == None:
            self.test_Data = self.data_set[len(self.train_Data):]
            self.test_label = self.data_labels[len(self.train_Data):]
        else:
            self.test_Data = self.data_set[self.train_Data_amount:]
            self.test_label = self.data_labels[self.train_Data_amount:]

        self.train_data_shape = self.train_Data.shape
        self.test_data_shape = self.test_Data.shape
        del self.data_set, self.data_labels

    def sys_breaks(self, mesag=None):
        if mesag !=None:
            print(mesag)
        sys.exit()

    def layer_desend(self, layer_shape):
        if layer_shape == 'norm':
            _, w = self.train_data_shape
            return w, (w - self.output_neuron)/self.hidden_layers
        else:
            self.sys_breaks('layer_shape is not defined correctly')
            
    def layer_density(self, net_shape='norm'):
        print('Creating the layers of nural network')
        w, inc = self.layer_desend(net_shape)
        if net_shape == 'norm':
            for i in range(self.hidden_layers):

                if inc < 0.01:
                    next_lop = round(w - (inc + 1)) 
                else:
                    next_lop = round(w - inc) 

                # this sets the layer density to tihe output layer number if the density decreaser from the output number
                if (next_lop < self.output_neuron) or (i == self.hidden_layers-1):
                    next_lop = self.output_neuron

                # print(w, next_lop)
                self.weights.append(0.10*np.random.randn(w, next_lop))
                if i == 0:
                    self.biases.append(np.zeros((1, next_lop)))
                elif i > 0:
                    self.biases.append(0.10*np.random.randn(1, next_lop))

                w = next_lop

        else:
            self.sys_breaks(f'Unvalid parameter "{net_shape}" given at net_shape')

    def save_model(self, file_name=None, type='json'):
        '''
            save_model()
            ============

            This method "save_model" saves the model data in a file. the file name can be provided or the file name will be the current datetime value.  
        '''

        if file_name == None:
            current_time = time.strftime('%A_%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
            file_name = f'{current_time}.txt'

    def testing_model(self, input, model):
        return input, model

    def neuron_activation(self, inputs=None, inputs2=None, func_type=None):
        
        if func_type == "softmax":
            obj = np.exp(inputs - np.max(inputs))
            ret = obj / np.sum(obj, axis=1, keepdims=True)

        elif func_type == 'relu':
            ret = np.maximum(0, inputs)
        
        elif func_type == 'one_hot':
            siz = inputs.size
            onne = np.zeros((siz, self.output_neuron))
            onne[np.arange(siz), inputs.astype(int)] = 1
            ret = onne

        elif (func_type == 'relu_deriv') or (func_type == 'leaky_reLu_deriv') :
            ret = inputs > 0

        elif func_type == 'leaky_reLu':
            ret = np.maximum(0.0000000001, inputs)

        elif func_type == 'entropy_loss':
        ret = -np.sum(self.neuron_activation(func_type='one_hot', inputs=self.train_label) * np.log(self.activated_nu[-1]))

        elif func_type == 'mean_sqr_error':
            ret = 1 / len(inputs) * np.sum((inputs - inputs2) ** 2, axis=0) 

        else:
            self.sys_breaks(f'fuc_typ needs a valid activation function name. Given {func_type}')

        return  np.nan_to_num(ret)

    def dot_producter(self, item1, item2, item3=None):

        item1_shape, item2_shape = np.array(item1, dtype='float').shape,  np.array(item2, dtype='float').shape
    
        if item1_shape[1] == item2_shape[0]:
            tok = np.dot(item1, item2)

        elif item1_shape[0] == item2_shape[0]:
            tok = np.dot(item1.T, item2)

        elif item1_shape[0] == item2_shape[1]:
            tok = np.dot(item1.T, item2.T)

        elif item1_shape[1] == item2_shape[1]:
            tok = np.dot(item1, item2.T)
        
        else:
            self.sys_breaks(f'Can not do dot-product with shapes-> {item1_shape} & {item2_shape}  ')

        return tok

    def updata_params(self, weights, biases):
        '''Updates all the weights and biases.'''
        confu_w,confu_b =[],[] 

    
        for i in range(len(weights)):
            
            to = []
            w = weights[i] * self.learning_rate 
            _, b_shape =  self.weights[i].shape 
            to = [self.weights[i][l] - w[l] for l in range(b_shape)]
            # confu_w.append(to)
            self.sys_breaks(np.array(to).shape)


        for i in range(len(biases)):
            b_o = self.learning_rate * biases[i]
            b = self.biases[i] - b_o
            confu_b.append(b)

    
        self.weights, self.biases = confu_w, confu_b
        self.sys_breaks()

    def backward_prop(self,func_type):
        nur = len(self.activated_nu)

        re_wi,re_bi = [],[]
        m,one_hot_Y = self.train_label.size, self.neuron_activation(func_type='one_hot', inputs=self.train_label)

        un_activa, activa, weig =  self.unactivated_nu[::-1], self.activated_nu[::-1], self.weights[::-1]

        error_found = self.neuron_activation(func_type='mean_sqr_error', inputs=activa[0], inputs2=m)
            
        for i in range(self.hidden_layers):
            
            if func_type == 'softmax':

                if i == 0:
                    data = activa[0] - one_hot_Y
                    re_wi.append(1/self.dot_producter(data, activa[1])*m)
                    re_bi.append(1/m * np.sum(data))
                    
                elif i < (self.hidden_layers - 1):

                    d1 = self.dot_producter(data, weig[i-1])
                    data = d1 * self.neuron_activation(func_type='relu_deriv', inputs=activa[i])

                    re_wi.append(1/self.dot_producter(data, un_activa[i]) * m)
                    re_bi.append(1/m*np.sum(data))

                elif i == (self.hidden_layers - 1):
                    d1 = self.dot_producter(data, weig[i-1])
                    data = d1 * self.neuron_activation(func_type='relu_deriv', inputs=activa[i])

                    re_wi.append(1/m*self.dot_producter(data, self.train_Data).T)
                    re_bi.append(1/m*np.sum(data))

            else:
                self.sys_breaks(f'The function given for the backwords propogaetion "{func_type}" is not valid')

            print(f'BackProp layer no:- {(i + 1)} / [{round(((i+1)/nur)*100)}% |{ "="*round(((i+1)/nur)*100)}{" "*(100 - round(((i)/nur)*100) )}| 100%]', end='\r')
        self.updata_params(re_wi[::-1],re_bi[::-1])
        print('\n')
        return error_found

    def forward_prop(self, test_input=None):

        '''
            forward_prop()
            ==============

            This method "forward_prop" is the function the handles the forward propagation part of the nural network.

        '''
        
        self.unactivated_nu,self.activated_nu = [],[]
        
        if test_input != None:
            inputs = test_input
        else: 
            inputs = self.train_Data

        for i in range(self.hidden_layers):
            
            nu = np.dot(inputs, self.weights[i]) + self.biases[i]

            if len(self.activation_func) > 2:
                # couldn't figure out how to do this.
                pass
            else:

                if i == (self.hidden_layers - 1):
                    ac_f = self.neuron_activation(inputs=nu, func_type=self.activation_func[1])
                else:
                    ac_f = self.neuron_activation(inputs=nu, func_type=self.activation_func[0])
                        
            self.unactivated_nu.append(nu)
            self.activated_nu.append(ac_f)
            inputs = nu
            del ac_f
            print(f'Layer no:- {(i + 1)} / [{round(((i+1)/self.hidden_layers)*100)}% |{ "="*round(((i+1)/self.hidden_layers)*100)}{" "*(100 - round(((i)/self.hidden_layers)*100) )}| 100%]', end='\r')
        
        del inputs
        print('\n')

    def neuron_train(self):
        
        '''
            neuron_train()
            =============

            This function "neuron_train" is the main function the is used to train the convolutional nural network.

        '''
        self.data_arange()
        self.layer_density(net_shape=self.layer_structure)

        print('Training started...\n')
        train_start = time.time()
        for epo in range(self.epochs):
            accuracy,loss = 0,0
            epo_start = time.time()
            print(f'Epoch =\t{epo+1}/{self.epochs}\t\tRemaining epoch = {self.epochs - (epo + 1)}')
            
            self.forward_prop()
            self.backward_prop('softmax')

            print(f'Accuracy: {accuracy}\t\t Error/Loss: {loss}\t\t Time_Taken: {round(time.time() - epo_start)} sec\n')

        print(f'Total time taken for training: {round(time.time() - train_start)} sec(s).') 
        print('Training Ended.\n')
    
