import numpy as np
import theano.tensor as T
from theano import function
import theano
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score




if __name__ == '__main__':

    X, target = make_classification(n_samples=50, n_features=2, n_classes=2, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class=1)



    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=.2)

    x = T.dmatrix('x')
    y = T.dvector('y')

    w_1 = theano.shared(np.random.randn(2))
    w_3 = theano.shared(np.random.randn(2))
    b_1 = theano.shared(0., name='b_1')
    b_3 = theano.shared(0., name='b_3')

    a_1 = 1/(1 + T.exp(-T.dot(x, w_1)-b_1))
    a_1_out = a_1 > 0.5
    #a_2 = 1/(1 + T.exp(-T.dot(x, w_2)-b))
    #a_2_out = a_2 > 0.5

    a_3 = 1/(1 + T.exp(-T.dot(a_1, w_3) - b_3))
    a_3_out = a_3 > 0.5

    ent = -y * T.log(a_3) - (1-y)*T.log(1-a_3)
    cost = ent.mean()

    g_w_1, g_w_3, g_b_1, g_b_3 = T.grad(cost, [w_1, w_3, b_1, b_3])

    train = theano.function(inputs=[x,y], outputs=[a_3_out, ent],
            updates=((w_1, w_1-0.1*g_w_1), (w_3, w_3-0.1*g_w_3), (b_1, b_1-0.01*g_b_1), (b_3, b_3-0.01*g_b_3)))

    predict = theano.function(inputs=[x], outputs=a_3_out)
    
    for i in range(10):
        train(X_train, y_train)
