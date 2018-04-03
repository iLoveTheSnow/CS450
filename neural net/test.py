import numpy
from sknn.mlp import Classifier, Layer
X=numpy.array([[0,1],[0,0],[1,0]])
print(X.shape)
y=numpy.array([[1],[0],[2]])
print(y.shape)
nn=Classifier(layers=[Layer("Sigmoid",units=2),Layer("Sigmoid",units=3)],n_iter=10)
nn.fit(X,y)
