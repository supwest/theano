import numpy as np
import theano.tensor as T
from theano import function
from sklearn.datasets import make_classification, make_regression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error

x=T.dscalar('x')
y = T.dscalar('y')

z = x+y

f = function([x,y], z)

print f(4,3)
print f(67, 2)


x = T.dmatrix('x')
y = T.dmatrix('y')

z = x + y
f = function([x,y], z)

print f([[1,2], [3,4]], [[50, 60], [70, 80]])

j = x + y
g = function([x,y], j)

print g(np.array([[1,2],[3,4]]), np.array([[50, 60], [70, 80]]))

import theano
a = theano.tensor.vector()
out = a+a**10
f = function([a], out)
print f([0,1,2])

#Logistic Regression using gradient descent with Theano
N = 500
feats = 100

X, y = make_classification(n_samples = N, n_features=100, n_informative=80, n_redundant=20, n_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


D = (np.random.randn(N, feats), np.random.randint(size=N, low=0, high=2))
training_steps = 10000

x = T.dmatrix('x')
y = T.dvector('y')

w = theano.shared(np.random.randn(feats), name='w')

b = theano.shared(0., name='b')

print "Initial model:"
print w.get_value()
print b.get_value()

p_1 = 1/(1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1-y)*T.log(1-p_1) #cross-entropy loss function
cost = xent.mean() + 0.01 * (w**2).sum()    #cost function to minimize

gw, gb = T.grad(cost, [w,b])


train = theano.function(inputs=[x,y], outputs=[prediction, xent],
                        updates=((w, w-0.1*gw), (b, b-0.01*gb)))

predict = theano.function(inputs=[x], outputs=prediction)
print "training"
for i in range(training_steps):
    #pred, err = train(X_train, y_train)
    train(X_train, y_train)
print "Final model:"
print w.get_value()
print b.get_value()
print "target values for D:"
print y_test
print "prediction on D:"
print predict(X_test)


from sklearn.metrics import precision_score, recall_score

print "Recall"
print recall_score(y_test, predict(X_test))
print "Precision"
print precision_score(y_test, predict(X_test))


lr = LogisticRegression()
lr.fit(X_train, y_train)
print "Logistic Recall"
print recall_score(y_test, lr.predict(X_test))
print"Logistic Precision"
print precision_score(y_test, lr.predict(X_test))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print "RF Recall"
print recall_score(y_test, rf.predict(X_test))
print"RF Precision"
print precision_score(y_test, rf.predict(X_test))


#Linear Regression
N = 500
feats = 100

X, y = make_regression(n_samples = N, n_features=100, n_informative=80, n_targets=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


training_steps = 10000

x = T.dmatrix('x')
y = T.dvector('y')

w = theano.shared(np.random.randn(feats), name='w')

b = theano.shared(0., name='b')

print "Initial model:"
print w.get_value()
print b.get_value()


#beta = T.nlinalg.dot(T.nlinalg.matrix_inverse(T.nlinalg.matrix_dot(T.transpose(x), x)), T.nlinalg.dot(T.transpose(x), y))

rss = (y - T.dot(x, w)-b)**2
prediction = T.dot(x, w) + b
#cost = rss.mean() + 0.01 * (w**2).sum()    #cost function to minimize
cost = rss.mean() + .5*(0.01*(w**2).sum()) + (1-.5)*(0.01*abs(w).sum())

gw, gb = T.grad(cost, [w,b])


train = theano.function(inputs=[x,y], outputs=[prediction, cost],
                        updates=((w, w-0.1*gw), (b, b-0.01*gb)))

predict = theano.function(inputs=[x], outputs=prediction)
print "training"
for i in range(training_steps):
    #pred, err = train(X_train, y_train)
    train(X_train, y_train)
print "Final model:"
print w.get_value()
print b.get_value()
print "target values for D:"
print y_test
print "prediction on D:"
print predict(X_test)

import statsmodels.api as sm

lin_reg = sm.OLS(y_train, X_train).fit()
print lin_reg.summary()

print "Theano Scores"
print "R2 Score is {}".format(r2_score(y_test, predict(X_test)))
print "MSE is {}".format(mean_squared_error(y_test, predict(X_test)))
print "*"*40
print "Statsmodels score"
print "R2 Scores is {}".format(r2_score(y_test, lin_reg.predict(X_test)))
print "MSE is {}".format(mean_squared_error(y_test, lin_reg.predict(X_test)))

print w.get_value()-lin_reg.params
