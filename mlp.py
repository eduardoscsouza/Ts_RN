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
def MSE(pred, true, der=False):
	return np.sum(np.square(pred-true))/2 if not der else (pred-true)



# Calcula a acuracia dados a saida predita e a real
def accuracy(pred, true):		
	return np.count_nonzero(np.argmax(pred, axis=1) == np.argmax(true, axis=1))/len(pred)



class MLP:
	#Gera a MLP
	def __init__(self, input_size, layers_sizes, actv_funcs=None, loss_func=MSE, metrics=[("Accuracy", accuracy)]):
		#Define todas as funções de ativação como sendo a logistica caso não definidas
		if not actv_funcs:
			actv_funcs = [logistic for _ in range(len(layers_sizes))]

		#Verificação da entrada
		assert input_size > 0
		assert all([layer_size > 0 for layer_size in layers_sizes])
		assert len(layers_sizes) == len(actv_funcs)

		#Atribuição das funções e métricas
		self.set_actv_funcs(actv_funcs)
		self.set_loss_func(loss_func)
		self.set_metrics(metrics)

		#Gera as camadas com pesos entre -1 e 1
		self.layers = [np.random.random_sample((layers_sizes[0], input_size+1))*2.0 - 1]
		self.layers += [np.random.random_sample((layers_sizes[i], layers_sizes[i-1]+1))*2.0 - 1 for i in range(1, len(layers_sizes))]

		#Vetores auxiliares
		self.last_inputs = [np.ones((input_size+1, ))] + [np.ones((layer_size+1, )) for layer_size in layers_sizes]
		self.last_nets = [np.empty((layer_size, )) for layer_size in layers_sizes]
		self.last_ders = [np.zeros(layer.shape) for layer in self.layers]

	#Atribuição das funções de ativação
	def set_actv_funcs(self, actv_funcs):
		#Verificação da entrada e atribuição
		assert all([callable(actv_func) for actv_func in actv_funcs])
		self.actv_funcs = actv_funcs

	#Atribuição da função de loss
	def set_loss_func(self, loss_func):
		#Verificação da entrada e atribuição
		assert callable(loss_func)
		self.loss_func = loss_func

	#Atribuição das métricas
	def set_metrics(self, metrics):
		#Verificação da entrada e atribuição
		assert all([callable(metric[1]) for metric in metrics])
		self.metrics = metrics

	#Faz o forward da MLP para uma sample
	def _predict_single(self, x):
		#Executar o forward camada por camada
		self.last_inputs[0][:-1] = x
		for i in range(len(self.layers)):
			self.last_nets[i] = np.sum(self.last_inputs[i] * self.layers[i], axis=1)
			self.last_inputs[i+1][:-1] = self.actv_funcs[i](self.last_nets[i], False)

		#Retorna a saida da ultima camada
		return np.copy(self.last_inputs[-1][:-1])

	#Faz o forward da MLP para todo o dataset de entrada
	def predict(self, x):
		#Verificação da entrada
		assert type(x) == np.ndarray
		#Caso seja apenas uma sample, executa predict_single
		if (len(x.shape) == 1):
			assert len(x) == self.layers[0].shape[1]-1
			return _predict_single(x)
		assert len(x.shape) == 2
		assert x.shape[1] == self.layers[0].shape[1]-1

		#Executa predict_single para cada sample, e retorna as saidas em uma matriz
		return np.stack([self._predict_single(sample) for sample in x], axis=0)

	def _evaluate_no_loss(self):
		pass

	def evaluate(self, x, y):
		pass

	#Faz o backpropagation da MLP
	def fit(self, x, y, epochs=1, loss_thresh=0.001, learning_rate=0.1, momentum=0.01, reset_momentum=False, return_history=False, verbose=False):
		#Verificação da entrada
		assert (type(x) == np.ndarray) and (type(y) == np.ndarray)
		assert (len(x.shape) == 2) and (len(y.shape) == 2)
		assert len(x) == len(y)
		assert epochs > 0
		assert loss_thresh >= 0
		assert learning_rate > 0
		assert (momentum >= 0) and (momentum < 1)

		#Faz o loop do fit até o numero máximo de epochs
		#ou até a loss ser menor que o threshold
		cur_epoch = 0
		cur_epoch_loss = loss_thresh+1
		while(cur_epoch_loss > loss_thresh and cur_epoch < epochs):
			#Variavel acumuladora da loss da epoch
			cur_epoch_loss = 0
			#Fitar sample a sample
			for sample, ground_truth in zip(x, y):
				#Calcular a saída e a loss da saída
				cur_out = self._predict_single(sample)
				cur_epoch_loss += self.loss_func(cur_out, ground_truth, False)

				#------Derivar e propagar para todas as camada------
				#---Derivação inicialmente da última camada, fora do loop por ser diferente das outras---
				#A variável der_backprop armazena a derivada que esta sendo propagada para trás
				der_backprop = self.loss_func(cur_out, ground_truth, True) * self.actv_funcs[-1](self.last_nets[-1], True)
				#Necessrio para multiplicar na linha de baixo, apenas altera o formato do array
				der_backprop = np.expand_dims(der_backprop, axis=1)
				#A variável cur_der armazena a derivada para os pesos da camada atual na iteração atual, antes do momentum
				cur_der = der_backprop * np.tile(self.last_inputs[-2], (len(self.layers[-1]), 1))
				#A variável self.last_ders armazena o estado atual do momentum da derivada para cada camada
				self.last_ders[-1] = momentum * self.last_ders[-1] + learning_rate * cur_der
				#---Derivação em loop para as camadas escondidas, da camada mais proxima a saída à mais próxima a entrada---
				for i in range(len(self.layers)-2, -1, -1):
					der_backprop = np.sum(der_backprop * self.layers[i+1][:, :-1], axis=0) * self.actv_funcs[i](self.last_nets[i], True)
					der_backprop = np.expand_dims(der_backprop, axis=1)
					cur_der = der_backprop * np.tile(self.last_inputs[i], (len(self.layers[i]), 1))
					self.last_ders[i] = momentum * self.last_ders[i] + learning_rate * cur_der

				#Atualização de todos os pesos de todas as camadas pelo gradiente considerendo momentum
				self.layers = [layer - last_der for layer, last_der in zip(self.layers, self.last_ders)]

			#Pegar a loss média da epoch
			cur_epoch_loss /= len(x)
			cur_epoch += 1
			print(cur_epoch_loss, accuracy(self.predict(x), y))



from sklearn.datasets import load_iris
x, aux_y = (np.asarray(iris) for iris in load_iris(True))
y = np.zeros((len(x), 3))
for i in range(len(x)):
	y[i, aux_y[i]] = 1

order = np.random.choice(len(x), len(x), replace=False)
x = x[order]
y = y[order]

#print(x)
#print(y)
#print(x.shape, y.shape)

mlp = MLP(4, [20, 3])
mlp.fit(x, y, 200000, loss_thresh=0.0001, learning_rate=0.01, momentum=0.005)