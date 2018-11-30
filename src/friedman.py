import stac
import numpy as np
d0 = []
d1 = []
d2 = []
for i in range(6):
    d0.append(np.load("run"+str(i)+"/test/student/test_accuracy_lnin_lsn_run"+str(i)+".npy"))
for i in range(6):
    d1.append(np.load("run"+str(i)+"/test/student/test_accuracy_dnin_dsn_run"+str(i)+".npy"))
for i in range(6):
    d2.append(np.load("run"+str(i)+"/test/student/test_accuracy_mnin_msn_run"+str(i)+".npy"))

d0 = np.mean(np.asarray(d0),axis=1)
d1 = np.mean(np.asarray(d1),axis=1)
d2 = np.mean(np.asarray(d2),axis=1)

print "Mean:", np.mean(d0), np.mean(d1), np.mean(d2)
print "Standard Deviation: ", np.std(d0), np.std(d1), np.std(d2),'\n'
stac.nonparametric_tests.friedman_test(d0,d1,d2)