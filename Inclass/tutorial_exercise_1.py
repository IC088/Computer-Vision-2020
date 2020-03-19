import numpy as np

w = np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.1]])


x = np.array([1,2,2,1]).reshape(4,1)

print(w.shape)
print(x.shape)


'''
calc W*X + b
'''

res = np.matmul(w,x)


print(res)
# '''
# function softmax is a cheap knockoff
# '''

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


softmaxed = softmax(res)

print(softmaxed)

cross_entropy_loss = -np.log(softmaxed)


print(cross_entropy_loss)

