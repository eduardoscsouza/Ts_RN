import numpy as np



#Função logistica, também conhecida como função sigmoidal
def logistic(x, der):
	return 1/(1 + np.exp(-x)) if not der else np.exp(x)/((1 + np.exp(x))**2)

#Função tangente hiperbolica
def tanh(x, der):
	return np.tanh(x)  if not der else  1 - np.tanh(x)**2

#Função ReLU(rectified linear unit)
def ReLU(x, der):
	return  np.max(0, x) if not der else (1 if (x>0) else 0)

#Função softplus(suavização da ReLU)
def softplus(x, der):
	return  np.log(1 + np.exp(x)) if not der else 1/(1 + np.exp(-x))

#Função linear f(x)=x
def linear(x, der):
	return x if not der else 1



class MLP:

	#Gera a MLP
	def __init__(self, input_size, layers_sizes, actv_funcs=None):

		#Define todas as funções de ativação como sendo a logistica caso não definidas
		if not actv_funcs:
			actv_funcs = [logistic for _ in range(len(layers_sizes))]

		assert len(layers_sizes) == len(actv_funcs)

		#Gera as camadas com pesos entre -1 e 1
		self.layers = [np.random.random_sample((layers_sizes[0], input_size+1))*2.0 - 1]
		self.layers += [np.random.random_sample((layers_sizes[i], layers_sizes[i-1]+1))*2.0 - 1 for i in range(1, len(layers_sizes))]
		








# Adiciona 1 aos vetores. Utilizado para adicionar o bias
def append_one(x):
	aux = np.ones(x.shape[0]+1)
	aux[0:x.shape[0]] = x
	return aux

# Calcula a acuracia da MLP utilizando uma amostra de entrada e suas respectivas saidas esperadas.
def get_acc(dataset_x, dataset_y, hidden_layer, out_layer):
	acc = 0
	for i in range(dataset_x.shape[0]):
		out = feed_mlp(dataset_x[i], hidden_layer, out_layer)[0]
		if(np.argmax(out) == np.argmax(dataset_y[i])):
			acc += 1

	return acc/dataset_x.shape[0]

"""### Geração da Rede

A função abaixo cria as camadas da rede, que são basicamente vetores que armazenam os pesos de cada neurônio. Nesse exercício, foi usada apenas uma camada escondida (**hidden_layer**), além da camada de saída (**out_layer**). A função pode ser usada para geração de qualquer rede MLP, uma vez que os tamanhos da entrada, da camada escondida e da camada de saída são especificados como parâmetros.

Os pesos são inicializados com valores aleatórios entre -0.1 e 0.1, conforme especificado no exercício.
"""

# Essa função gera duas camadas, uma de entrada e outra de sáida.
def generate_mlp(input_size, hidden_size, out_size):
	hidden_layer = np.random.random_sample((hidden_size, input_size+1))#generate value 0 to 1
	out_layer = np.random.random_sample((out_size, hidden_size+1)) # generate value 0 to 1
	# normalizado entre -0.1 e 0.1 para atender os requisitos do trabalho
	hidden_layer = minmax_scale(hidden_layer, feature_range=(-0.1,0.1))
	out_layer = minmax_scale(out_layer, feature_range=(-0.1,0.1))
	return hidden_layer, out_layer

"""### Função de Ativação

Como função de ativação, g(), foi usada a função logística, por atender os requisitos de ser uma função contínua, estritamente crescente e limitada. No processo de aprendizado é utilizada a derivada da função g(). Ambas são definidas abaixo.
"""


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

"""### Ativação da Rede

A ativação da rede consiste em calcular uma saída a partir de uma entrada, utilizando os pesos calculados durante o treinamento.

*A* função **feed_mlp** implementa este processo. O passo inicial é adicionar um *bias* à entrada (linha 4). Então é calculada a saída da camada escondida, conforme a fórmula abaixo:
$$ F(o_1, ..., o_j) = g(\sum_{i} w_{ij} x_i )$$
Na fórmula acima, g() é a função de ativação.

Este processo é repetido para a camada de saída, utilizando como entrada a saída da camada escondida, calculada anteriormente.
"""

# essa função ativa a rede
def feed_mlp(network_input, hidden_layer, out_layer):
  
	aux_network_input = append_one(network_input) # adicionas 1 na entrada para o bias
	hidden_out = log(np.sum(aux_network_input * hidden_layer, axis=1)) # multiplica os pesos ou seja ativa  a rede 

	aux_hidden_out = append_one(hidden_out)#adiciona 1 para o bias
	network_out = log(np.sum(aux_hidden_out * out_layer, axis=1)) # calcula a ativação

	return network_out, hidden_out

"""### Criação dos Datasets e Execução dos Testes

Para cada um dos problemas, XOR e mapeamento identidade, são executados os passos abaixo:
* criação das amostras de teste e saídas esperadas;
* criação da MLP, especificando o tamanho das camadas de acordo com o problema;
* treinamento da MLP;
* cálculo da acurácia, através da ativação da MLP para cada um dos itens da amostra de teste.

A alteração do valor da taxa de aprendizagem (**eta**) pode impactar na acurácia da MLP para o caso do mapeamento identidade. No código abaixo, para cada problema, são executados treinamentos com taxas de aprendizagem 0.1 e 0.5, para demonstrar essa possível diferença.
"""

dataset_x = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
dataset_y = np.asarray([[0], [1], [1], [0]])

hidden_layer, out_layer = generate_mlp(2, 2, 1)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=1000)
print("XOR Accuracy (eta = 0.1):", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))

hidden_layer, out_layer = generate_mlp(2, 2, 1)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=1000)
print("XOR Accuracy (eta = 0.5):", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))


dataset_x = np.identity(8)
dataset_y = dataset_x
print(dataset_x)

hidden_layer, out_layer = generate_mlp(8, 3, 8)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=1000)
print("8-Dim Identity Accuracy (eta = 0.1):", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))

hidden_layer, out_layer = generate_mlp(8, 3, 8)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.5, max_it=1000)
print("8-Dim Identity Accuracy (eta = 0.5):", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))

print()

dataset_x = np.identity(15)
dataset_y = dataset_x
print(dataset_x)

hidden_layer, out_layer = generate_mlp(15, 4, 15)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=1000)
print("15-Dim Identity Accuracy (eta = 0.1):", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))

hidden_layer, out_layer = generate_mlp(15, 4, 15)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.5, max_it=1000)
print("15-Dim Identity Accuracy (eta = 0.5):", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))