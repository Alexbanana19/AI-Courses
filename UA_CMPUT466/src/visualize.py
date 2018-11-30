import matplotlib.pyplot as plt
import pickle
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return 1.*e_x / e_x.sum()

original = 2252
same = 2253

X = np.load("X_5.npy")
X = X

d0 = np.load("mnin_logits.npy")
d0[d0>=0.5] = 1
d0[d0<0.5] = 0

#print d0[original]
#print np.sum(d0[original], axis=0)
print softmax(np.sum(d0[original], axis=0))

min_dist = 99999
similar = None
sindex = 0

for i in range(10000):
    dist = np.sum(np.abs(d0[original]-d0[i]))
    if dist < min_dist and i != original:# and np.argmax(np.sum(d0[i],axis=0))!=np.argmax(np.sum(d0[original],axis=0)):
        min_dist = dist
        similar = d0[i]
        sindex = i
#print similar
#print np.sum(similar, axis=0)
d2 = np.sum(np.abs(d0[original]-similar))
print softmax(np.sum(d0[sindex], axis=0))

min_dist = 99999
different = None
dindex = 0

for i in range(10000):
    dist = np.sum(np.abs(d0[original]-d0[i]))
    if dist < min_dist and i != original and np.argmax(np.sum(d0[i],axis=0))!=np.argmax(np.sum(d0[original],axis=0)):
        min_dist = dist
        different = d0[i]
        dindex = i
#print different
#print np.sum(different, axis=0)
d3 = np.sum(np.abs(d0[original]-different))
print softmax(np.sum(d0[dindex], axis=0))

#print d0[same]
#print np.sum(d0[same], axis=0)
d4 = np.sum(np.abs(d0[original]-d0[same]))
print softmax(np.sum(d0[same], axis=0))

plt.subplot(221),plt.title('origin image: bird')
plt.imshow(X[original]),plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.title('similar image same class, hamming dist: '+ str(d2))
plt.imshow(X[sindex]),plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.title('similar image different class, hamming dist: '+ str(d3))
plt.imshow(X[dindex]),plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.title('different image same class, hamming dist: '+ str(d4))
plt.imshow(X[same]),plt.xticks([]), plt.yticks([])



fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.matshow(d0[original].T,cmap=plt.cm.gray_r)
ax1.set_ylabel('class')
ax1.set_xlabel('channel')
ax1.set_title('origin image: bird')
ax1.set_xticks([])
ax1.set_yticks([])

ax2.matshow(d0[sindex].T,cmap=plt.cm.gray_r)
ax2.set_ylabel('class')
ax2.set_xlabel('channel')
ax2.set_title('similar image same class, hamming dist: '+ str(d2))
ax2.set_xticks([])
ax2.set_yticks([])

ax3.matshow(d0[dindex].T,cmap=plt.cm.gray_r)
ax3.set_ylabel('class')
ax3.set_xlabel('channel')
ax3.set_title('similar image different class, hamming dist: '+ str(d3))
ax3.set_xticks([])
ax3.set_yticks([])

ax4.matshow(d0[same].T,cmap=plt.cm.gray_r)
ax4.set_ylabel('class')
ax4.set_xlabel('channel')
ax4.set_title('different image same class, hamming dist: '+ str(d4))
ax4.set_xticks([])
ax4.set_yticks([])

plt.show()
