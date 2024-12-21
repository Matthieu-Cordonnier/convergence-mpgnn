import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

big_n=1000  #size of the "limit" graph
ns = np.logspace(1, 3, 40).astype(int) #sizes of growing graphs
agg='mean'
# agg='max'
nb_approx_cGNN = 10 #The cGNN "limit" is calculated as the mean over nb_approx_cGNN random graph GNNs of size big_n
dims = [2,2,2,2] #len(dims) is the number of MPGNN layers. Each MP layers uses a single-layer MLP of width dims[i]. 
latent_dims = np.array([2, 3, 5, 10]) #Dimensionality of the latent space
if agg == 'mean': #We plot the mean of nexp random experiments.
    nexp = 50 
elif agg == 'max':
    nexp=100


########################
#### Define MPGNN ######
########################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def MLP(inputs, theta):
    outputs = sigmoid(np.dot(inputs, theta))
    return outputs

def message_passing_layer(inputs, W, agg='mean'):
    outputs = W[:,:,None] * inputs[:,None,:]
    if agg =='mean':
        outputs = outputs.mean(axis=0)
    elif agg =='max':
        outputs = np.max(outputs, axis=0)
    return outputs

def multilayer_MPGNN(inputs, W, Theta, agg='mean'):
    for i in range(len(Theta)):
        mssg = MLP(inputs, Theta[i])
        outputs = message_passing_layer(inputs=mssg, W=W, agg=agg)
        inputs = outputs
    if agg =='mean':
         return outputs.mean(axis=0)
    elif agg == 'max':
         return np.max(outputs, axis=0)




##############################
### Run the experiments ######
##############################

results = np.zeros((len(ns), len(latent_dims), nexp))

# for max the computation of the continuous limit is very imprecise.
# hence we place ourselves in a case where it is known, with positive parameters
Theta = [np.random.rand(1, dims[0])]
for i in range(len(dims)-1) : 
    Theta.append(np.random.rand(dims[i], dims[i+1]))


for _ in range(nexp):
    for di, d in enumerate(latent_dims):
        a = np.random.rand(d,1)
        if agg=='mean':
            cGNN_output = np.zeros((nb_approx_cGNN, dims[-1]))
            for i in range(nb_approx_cGNN):
                Pos = np.random.rand(big_n, d)
                X = Pos@a
                W = np.exp(-(cdist(Pos, Pos)**2))
                cGNN_output[i] = (multilayer_MPGNN(inputs=X, W=W, Theta=Theta, agg=agg))
                del(W)
            cGNN_output = np.mean(cGNN_output, axis = 0)
        elif agg=='max':
            # the continuous max limit is reached when all points are in the (1,...,1) corner
            Pos = np.ones((2, d))
            X = Pos@a
            W = np.exp(-(cdist(Pos, Pos)**2))
            cGNN_output = (multilayer_MPGNN(inputs=X, W=W, Theta=Theta, agg=agg))
        for ni, n in enumerate(ns): 
            pos = np.random.rand(n, d)
            x = pos@a
            w = np.exp(-(cdist(pos,pos))**2)
            GNN_output = multilayer_MPGNN(inputs=x, W=w, Theta=Theta, agg=agg)
            error = np.linalg.norm(GNN_output-cGNN_output, ord=np.inf)
            results[ni, di, _] = error
            print('expé numéro', _+1, 'dimension', d, 'sample numéro', ni+1, ':', results[ni][di][_])
            

resultsm = results.mean(axis=-1)
resultsstd = results.std(axis=-1)



col = ['b', 'g', 'r', 'k']
plt.figure(figsize=(10,10))
for (di, d) in enumerate(latent_dims):
  plt.loglog(ns, resultsm[:,di], '--', c=col[di], label=f'd={d}', linewidth=5)
  plt.gca().fill_between(ns, resultsm[:,di] + resultsstd[:,di],
                         resultsm[:,di] - resultsstd[:,di], alpha=.1, facecolor=col[di])
  if agg=='max':
      plt.loglog(ns, resultsm[0,di]*ns[0]**(1/d)/(ns**(1/d)), c=col[di],
                 label=f'd={d}, theory', linewidth=5) # shifted
if agg=='mean':
    plt.loglog(ns, 2*resultsm[0,0]*ns[0]**(1/2)/np.sqrt(ns), c='orange',
               label=f'1/sqrt(n):theory', linewidth=5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(numpoints=1, fontsize=23, ncol=2)
plt.xlabel('n', fontsize=25)
plt.ylabel('error', fontsize = 25)

# plt.savefig(f'{agg}.pdf', bbox_inches='tight')
plt.show()
            

