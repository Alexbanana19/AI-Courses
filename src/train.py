from __future__ import absolute_import
import numpy as np
import random
import tensorflow as tf
import pickle
import sys, os
sys.path.append('../')
sys.path.append('/home/share/minghan/ssae/')
sys.path.append('/home/share/minghan/keras/lib/python2.7/site-packages')

from argparse import ArgumentParser

from models import SN,LNIN, DNIN, MNIN, LSN, DSN, MSN, SSAE,AE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_size = 32*32*3
img_class = 10
batch_size = 128
channels = 80

def save_results(choice, teacher_model, student_model, loss, accuracy):
    if choice['student'] == 'none':
        if choice['teacher'] == 'lnin_ssae':
            ssae.save_model()
        elif choice['teacher'] == 'lnin_ae':
            ae.save_model()
        else:
            teacher_model.save_model()

    else:
        student_model.save_model()

    loss = np.asarray(loss)
    accuracy = np.asarray(accuracy)
    if choice['student'] != 'none':
        np.save("loss_"+choice['teacher'] +'_'+choice['student']+"_run"+str(choice['run'])+".npy", loss)
        np.save("accuracy_"+choice['teacher']+'_'+choice['student']+"_run"+str(choice['run'])+".npy",accuracy)

    else:
        np.save("loss_"+choice['teacher']+"_run"+str(choice['run'])+".npy",loss)
        np.save("accuracy_"+choice['teacher']+"_run"+str(choice['run'])+".npy",accuracy)

def train(choice, X_train, y_train, X_test, y_test):

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    student_model = None
    teacher_model = None
    with tf.Session(config = config) as sess:
        #Teacher Model
        global ssae, ae
        if choice['teacher'] == 'lnin':
            teacher_model = LNIN(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'lnin_ae':
            teacher_model = DNIN(img_size, img_class, sess, choice)
            ae = AE(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'lnin_ssae':
            teacher_model = DNIN(img_size, img_class, sess, choice)
            ae = AE(img_size, img_class, sess, choice)
            ssae = SSAE(img_size, img_class, sess, choice)
	elif choice['teacher'] == 'dnin':
	    teacher_model = DNIN(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'mnin':
            teacher_model = MNIN(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'tnin':
            teacher_model = TNIN(img_size, img_class, sess, choice)
        else:
            teacher_model = SN(img_size, img_class, sess, choice)
        #Student Model
        if choice['student'] == 'lsn':
            student_model = LSN(img_size, img_class, sess, choice)
        elif choice['student'] == 'dsn':
            student_model = DSN(img_size, img_class, sess, choice)
        elif choice['student'] == 'msn':
            student_model = MSN(img_size, img_class, sess, choice)

        if choice['student'] != 'none':
            if not student_model.save:
                sess.run(tf.initialize_all_variables())
                print "Initialize"
        else:
            if choice['teacher'] == 'lnin_ssae':
                if not ssae.save:
                    sess.run(tf.initialize_all_variables())
                    print "Initialize"
            elif choice['teacher'] == 'lnin_ae':
                if not ae.save:
                    sess.run(tf.initialize_all_variables())
                    print "Initialize"
            else:
                if not teacher_model.save:
                    sess.run(tf.initialize_all_variables())
                    print "Initialize"

        max_epochs = choice['epochs']
        l = 0
        a = 0
        test_accuracy = []

        step = 0
        epoch = 0
        loss =[]#np.load("loss_"+choice['teacher'] +'_'+choice['student']+"_run"+str(choice['run'])+".npy").tolist()
        accuracy=[]#np.load("accuracy_"+choice['teacher']+'_'+choice['student']+"_run"+str(choice['run'])+".npy").tolist()

        while epoch <= choice['epochs']:
    		if step*batch_size > X_train.shape[0] or step == 0:
    			s = np.arange(X_train.shape[0])
    			np.random.shuffle(s)

    			X_train = X_train[s]
    			y_train = y_train[s]
    			step = 0

		        print "Epoch:%d, loss: %f, accuracy: %f"%(epoch, l, a)
		        save_results(choice, teacher_model, student_model, loss, accuracy)

		        epoch += 1

		X_batch = X_train[step*batch_size:min(X_train.shape[0],(step+1)*batch_size), :]
		y_batch = y_train[step*batch_size:min(y_train.shape[0],(step+1)*batch_size), :]
        	if choice['student'] != 'none':
            		s = np.random.choice(X_batch.shape[0],X_batch.shape[0])
            		X_batch2 = X_batch[s]
           		eps = np.random.rand(X_batch.shape[0])[:,None,None,None]
			eps = np.tile(eps,[1,32,32,3])
            		X_batch3 = eps*X_batch + (1-eps)*X_batch2
                    	X_batch = np.concatenate((X_batch[:X_batch.shape[0]],X_batch3[X_batch3.shape[0]:]))

    		if choice['student'] != 'none':
		        if choice['teacher'] == 'lnin_ssae':
                    		#X_batch, y_batch = teacher_model.correct(X_batch, y_batch)
		           	hidden = teacher_model.get_hidden(X_batch)
                    		hidden = ae.get_embedding(hidden)
                    		labels = ssae.get_embedding(hidden)
		        elif choice['teacher'] == 'dnin':
		           	labels = teacher_model.predict(X_batch)
                    		#labels = y_batch
                	elif choice['teacher'] == 'tnin':
		           	labels = teacher_model.get_embedding(X_batch)
                    		labels[labels>=0.5] = 1
                    		labels[labels<0.5] = 0
                	elif choice['teacher'] == 'mnin':
                    		labels = teacher_model.get_sub_prediction(X_batch)
                    	elif choice['teacher'] == 'lnin':
                            	labels = teacher_model.get_logits(X_batch)

                	l,a = student_model.train(X_batch,y_batch,labels)
    		else:
                	if choice['teacher'] == 'lnin_ae':
                            	#X_batch, y_batch = teacher_model.correct(X_batch, y_batch)
                    		hidden = teacher_model.get_hidden(X_batch)
                    		l,a = ae.train(hidden)
                    	elif choice['teacher'] == 'lnin_ssae':
                            	#X_batch, y_batch = teacher_model.correct(X_batch, y_batch)
                    		hidden = teacher_model.get_hidden(X_batch)
                            	hidden = ae.get_embedding(hidden)
                    		l,a = ssae.train(hidden,y_batch)
                	else:
    			    	l,a = teacher_model.train(X_batch,y_batch)

           	loss.append(l)
    		accuracy.append(a)

           	if epoch == choice['epochs']:
                    a = 0
                    for i in range(100):
                        if choice['student'] != 'none':
                            a+=student_model.test(X_test[i*100:(i+1)*100],y_test[i*100:(i+1)*100])
                        else:
                            a+=teacher_model.test(X_test[i*100:(i+1)*100],y_test[i*100:(i+1)*100])
                    a = 1.*a/100
                    print "Test Accuracy ", a
                    test_accuracy.append(a)
                    if step == 100:
                        np.save("test_accuracy_"+choice['teacher']+'_'+choice['student']\
                            +"_run"+str(choice['run'])+".npy",np.asarray(test_accuracy))
                        print "Student Done"
                        break
    		step += 1

def test(choice, X_test, y_test):

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        #Teacher Model
        if choice['teacher'] == 'lnin':
            model = LNIN(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'lnin_ae':
            model = DNIN(img_size, img_class, sess, choice)
            ae = AE(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'lnin_ssae':
            model = DNIN(img_size, img_class, sess, choice)
            ae = AE(img_size, img_class, sess, choice)
            ssae = SSAE(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'dnin':
            model = DNIN(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'tnin':
            model = TNIN(img_size, img_class, sess, choice)
        elif choice['teacher'] == 'mnin':
            model = MNIN(img_size, img_class, sess, choice)
        else:
            model = SN(img_size, img_class, sess, choice)
        #Student Model
        if choice['student'] == 'lsn':
            model = LSN(img_size, img_class, sess, choice)

        elif choice['student'] == 'dsn':
            model = DSN(img_size, img_class, sess, choice)

        elif choice['student'] == 'msn':
            model = MSN(img_size, img_class, sess, choice)
        elif choice['student'] == 'ssn':
            model = SSN(img_size, img_class, sess, choice)

        if not model.save:
            #sess.run(tf.initialize_all_variables())
            print "No Model to test"
            return

        a = 0
        for i in range(100):
            if choice['student'] == 'none':
                if choice['teacher'] == 'lnin_ssae':
                    hidden = model.get_hidden(X_test[i*100:(i+1)*100])
                    hidden = ae.get_embedding(hidden)
                    a += ssae.test(hidden, y_test[i*100:(i+1)*100])

                else:
                    a += model.test(X_test[i*100:(i+1)*100], y_test[i*100:(i+1)*100])
            else:
                a += model.test(X_test[i*100:(i+1)*100], y_test[i*100:(i+1)*100])

        print " Test Average Accuracy: ", 1.*a/100

        return a

def main(choice):
    #========loading data=============
    # preprocessing
    print choice

    X = np.load("X.npy")
    y = np.load("y.npy")
    r = choice['run']

    X_test = X[r]
    y_test = y[r]

    X_train = np.delete(X, r, 0).reshape(50000,32,32,3)
    y_train = np.delete(y, r, 0).reshape(50000,10)

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    if choice['exp'] == 'train':
        #X_train = X_train.flatten().reshape(X_train.shape[0], img_size)
        train(choice, X_train, y_train,X_test, y_test)

    else:
        #X_test = X_test.flatten().reshape(X_test.shape[0], img_size)
        test(choice, X_test, y_test)

if __name__ == '__main__':
    arg_parser = ArgumentParser('Mini Project 4.0')
    arg_parser.add_argument('-e', '--experiment', type=str, default='train',
                            choices=['train', 'test'])

    arg_parser.add_argument('-t', '--teacher', type=str, default='tnin',
                            choices=['lnin','dnin', 'mnin', 'sn', 'tnin','lnin_ssae','lnin_ae'])

    arg_parser.add_argument('-s', '--student', type=str, default='none',
                            choices=['lsn', 'dsn','msn','none','ssn'])

    arg_parser.add_argument('-p', '--epochs', type=int, default=1)

    arg_parser.add_argument('-r', '--run', type=int, default=0)

    args = arg_parser.parse_args()
    choice = {'exp': args.experiment,'teacher': args.teacher,\
                'student':args.student, 'epochs': args.epochs, 'run': args.run}
    main(choice)
