import numpy as np

def append_one(x):
	aux = np.ones(x.shape[0]+1)
	aux[0:x.shape[0]] = x

	return aux

def sig(x):
	return 1/(1 + np.exp(-x))

def generate_mlp(input_size, hidden_size, out_size):
	hidden_layer = np.random.random_sample((hidden_size, input_size+1))/5.0 - 0.1
	out_layer = np.random.random_sample((out_size, hidden_size+1))/5.0 - 0.1

	return hidden_layer, out_layer

def forward_mlp(network_input, hidden_layer, out_layer):
	aux_network_input = append_one(network_input)
	hidden_out = sig(np.sum(aux_network_input * hidden_layer, axis=1))

	aux_hidden_out = append_one(hidden_out)
	network_out = sig(np.sum(aux_hidden_out * out_layer, axis=1))

	return network_out, hidden_out

def fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=10000):
	count = 0
	error = thresh+1
	while(error > thresh and count < max_it):
		error = 0
		for i in np.arange(dataset_x.shape[0]):
			network_out, hidden_out = forward_mlp(dataset_x[i], hidden_layer, out_layer)
			error += np.sum(np.square(network_out-dataset_y[i]))

			aux_hidden_out = append_one(hidden_out)
			aux_der = np.empty(out_layer.shape[0])
			der_out = np.empty(out_layer.shape)
			for oi in range(out_layer.shape[0]):
				aux_der[oi] = 2 * (network_out[oi]-dataset_y[i, oi]) * (network_out[oi]*(1-network_out[oi]))
				for oj in range(out_layer.shape[1]):
					der_out[oi, oj] = aux_der[oi] * aux_hidden_out[oj]
			
			aux_dataset_x = append_one(dataset_x[i])
			der_hidden = np.empty(hidden_layer.shape)
			for hi in np.arange(hidden_layer.shape[0]):
				for hj in np.arange(hidden_layer.shape[1]):
					aux_der_sum = 0
					for oi in np.arange(out_layer.shape[0]):
						aux_der_sum += aux_der[oi] * out_layer[oi, hi] * (hidden_out[hi]*(1-hidden_out[hi])) * aux_dataset_x[hj]
					der_hidden[hi, hj] = aux_der_sum

			out_layer = out_layer - eta*der_out
			hidden_layer = hidden_layer - eta*der_hidden

		error /= dataset_x.shape[0]
		count += 1
		#print("Current error:", error)

	return hidden_layer, out_layer

def get_acc(dataset_x, dataset_y, hidden_layer, out_layer):
	acc = 0
	for i in range(dataset_x.shape[0]):
		out = forward_mlp(dataset_x[i], hidden_layer, out_layer)[0]
		if(np.argmax(out) == np.argmax(dataset_y[i])):
			acc += 1

	return acc/dataset_x.shape[0]


dataset_x = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
dataset_y = np.asarray([[0], [1], [1], [0]])

hidden_layer, out_layer = generate_mlp(2, 2, 1)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=1000)
#hidden_layer = np.asarray([[-1, -1, 0.25], [1, 1, -1.75]])
#out_layer = np.asarray([[1, 1, -0.5]])

print("XOR Accuracy:", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))



dataset_x = np.identity(8)
dataset_y = dataset_x

hidden_layer, out_layer = generate_mlp(8, 3, 8)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.5, max_it=1000)

print("8-Dim Identity Accuracy:", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))



dataset_x = np.identity(15)
dataset_y = dataset_x

hidden_layer, out_layer = generate_mlp(15, 4, 15)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.5, max_it=1000)

print("15-Dim Identity Accuracy:", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))



from sklearn.datasets import load_iris
iris = load_iris(True)
dataset_x = np.asarray(iris[0])
aux_y = np.asarray(iris[1])
dataset_y = np.zeros((dataset_x.shape[0], 3))
for i in range(dataset_x.shape[0]):
	dataset_y[i, aux_y[i]] = 1


hidden_layer, out_layer = generate_mlp(4, 30, 3)
hidden_layer, out_layer = fit_mlp(dataset_x, dataset_y, hidden_layer, out_layer, thresh=0.01, eta=0.1, max_it=1000)

print("Iris Accuracy:", get_acc(dataset_x, dataset_y, hidden_layer, out_layer))