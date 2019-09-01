import numpy as np



#Função logistica, também conhecida como função sigmoidal
def logistic(x, der=False):
	return 1/(1 + np.exp(-x)) if not der else np.exp(x)/((1 + np.exp(x))**2)

#Função tangente hiperbolica
def tanh(x, der=False):
	return np.tanh(x)  if not der else  1 - np.tanh(x)**2

#Função ReLU(rectified linear unit)
def ReLU(x, der=False):
	return  np.max(0, x) if not der else (1 if (x>0) else 0)

#Função softplus(suavização da ReLU)
def softplus(x, der=False):
	return  np.log(1 + np.exp(x)) if not der else 1/(1 + np.exp(-x))

#Função linear f(x)=x
def linear(x, der=False):
	return x if not der else 1



#Função de Loss mean squared error
def MSE(pred, true, der):
	return np.sum(np.square(pred-true))/2 if not der else (pred-true)



# Calcula a acuracia dados a saida predita e a real
def accuracy(pred, true):		
	return np.count_nonzero(np.argmax(pred, axis=1) == np.argmax(true, axis=1))/len(pred)



class MLP:
	#Gera a MLP
	def __init__(self, input_size, layers_sizes, actv_funcs=None, loss_func=MSE, metrics=[("Accuracy", get_acc)]):
		#Define todas as funções de ativação como sendo a logistica caso não definidas
		if not actv_funcs:
			actv_funcs = [logistic for _ in range(len(layers_sizes))]

		#Verificação da entrada
		assert input_size > 0
		assert all(layers_sizes > 0)
		assert len(layers_sizes) == len(actv_funcs)

		#Atribuição das funções e métricas
		self.set_actv_funcs(actv_funcs)
		self.set_loss_func(loss_func)
		self.set_metrics(metrics)

		#Gera as camadas com pesos entre -1 e 1
		self.layers = [np.random.random_sample((layers_sizes[0], input_size+1))*2.0 - 1]
		self.layers += [np.random.random_sample((layers_sizes[i], layers_sizes[i-1]+1))*2.0 - 1 for i in range(1, len(layers_sizes))]

		#Vetores auxiliares
		self.last_input_vects = [np.ones((input_size+1, ))] + [np.ones((layer_size+1, )) for layer_size in layers_sizes]
		self.last_nets = [np.empty((layer_size, )) for layer_size in layers_sizes]
		self.last_ders = [np.zeros(layer.shape) for layer in layers]

	#Atribuição das funções de ativação
	def set_actv_funcs(actv_funcs):
		#Verificação da entrada e atribuição
		assert all([callable(actv_func) for actv_func in actv_funcs])
		self.actv_funcs = actv_funcs

	#Atribuição da função de loss
	def set_loss_func(loss_func):
		#Verificação da entrada e atribuição
		assert callable(loss_func)
		self.loss_func = loss_func

	#Atribuição das métricas
	def set_metrics(metrics):
		#Verificação da entrada e atribuição
		assert all([callable(metric[1]) for metric in metrics])
		self.metrics = metrics

	#Faz o forward da MLP para uma sample
	def _predict_single(x):
		#Executar o forward camada por camada
	  	self.last_input_vects[0][:-1] = x
	  	for i in range(len(self.layers)):
	  		self.last_nets[i] = np.sum(self.last_input_vects[i] * self.layers[i], axis=1)
	  		self.last_input_vects[i+1][:-1] = self.actv_funcs[i](self.last_nets[i], False)

	  	return self.last_input_vects[-1][:-1]

	#Faz o forward da MLP para todo o dataset de entrada
	def predict(x):
		#Verificação da entrada
	  	assert type(x) == np.ndarray
	  	assert len(x.shape) == 2
	  	assert x.shape[1] == self.layers[0].shape[1]-1

	  	return np.stack([self._predict_single(sample) for sample in x], axis=0)

	def _get_loss():

	def evaluate(x, y):

	def fit
		






def fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=10000):
	count = 0 # variavel que controla o número de iterações
	error = thresh+1 # inicia o erro com um valor > thresh para o treinamento nao parar
	while(error > thresh and count < max_it):
		error = 0
		for i in np.arange(dataset_x.shape[0]):
			network_out, hidden_out = feed_mlp(dataset_x[i], hidden_layer, out_layer)# realiza a ativação da rede
      ep = np.sum(np.square(network_out-dataset_y[i]))
			error += ep # calcula o erro da ultima camada

      # calculo do delta de  camada de saida
			aux_hidden_out = append_one(hidden_out)#adiciona o 1 para o bias
			aux_der = np.empty(out_layer.shape[0])
			der_out = np.empty(out_layer.shape)
			for oi in range(out_layer.shape[0]):
				aux_der[oi] = 2 * (network_out[oi]-dataset_y[i, oi]) * derivate_log(network_out[oi]) #  2 * erro é a derivada da função de perda mean square error (MSE)
				for oj in range(out_layer.shape[1]):
					der_out[oi, oj] = aux_der[oi] * aux_hidden_out[oj] 
			
      #calculo do delta na camada escondida utilizando o delta da camada de saida
			aux_dataset_x = append_one(dataset_x[i])
			der_hidden = np.empty(hidden_layer.shape)
			for hi in np.arange(hidden_layer.shape[0]):
				for hj in np.arange(hidden_layer.shape[1]):
					aux_der_sum = 0
					for oi in np.arange(out_layer.shape[0]):
						aux_der_sum += aux_der[oi] * out_layer[oi, hi] * derivate_log(hidden_out[hi]) * aux_dataset_x[hj]
					der_hidden[hi, hj] = aux_der_sum

			out_layer = out_layer - eta*der_out # atualiza os pesos da ultima camada
			hidden_layer = hidden_layer - eta*der_hidden # atualiza os pesos da  camada oculta

		error /= dataset_x.shape[0]
		count += 1
		#print("Current error:", error)

	return hidden_layer, out_layer