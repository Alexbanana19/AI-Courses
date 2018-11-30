from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
import numpy as np

img_size = 32*32*3
img_class = 10

def save_results(data, filename):
    with open(filename,'wb') as f:
        f.truncate()
    print "Saving File..."
    np.save(filename,data)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, img_class)
y_test = to_categorical(y_test, img_class)
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')

X_train /= 255
X_test /= 255

X = np.concatenate((X_train,X_test),axis=0)
y = np.concatenate((y_train,y_test),axis=0)

X_sort = {}
y_sort = {}
for i in range(10):
	X_sort[i] = []
	y_sort[i] = []

for i in range(X.shape[0]):
	t = np.argmax(y[i])
	X_sort[t].append(X[i])
	y_sort[t].append(y[i])


for i in range(10):
	s = np.arange(6000)
	np.random.shuffle(s)
	X_sort[i] = np.asarray(X_sort[i])
	X_sort[i] = X_sort[i][s]
	y_sort[i] = np.asarray(y_sort[i])
	y_sort[i] = y_sort[i][s]
	X_sort[i] = np.asarray(np.split(X_sort[i],6))
	y_sort[i] = np.asarray(np.split(y_sort[i],6))


X_split = X_sort[0]
y_split = y_sort[0]
for i in range(1,10):
	X_split = np.concatenate((X_split,X_sort[i]),axis=1)
	y_split = np.concatenate((y_split,y_sort[i]),axis=1)

print "Done Splitting"
save_results(X_split, "X.npy")
save_results(y_split, "y.npy")