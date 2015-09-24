from numpy import *
from scipy.sparse import csr_matrix
import scipy.sparse as sps
from sklearn import cross_validation
import gensim
from gensim import models,corpora

def getLda(datafile,row,column=496411,ntopics=200):
	file1 = open(datafile)
	datamatrix = csr_matrix((row,column))
	mtx_size = (row,column)#94535
	X = sps.lil_matrix(mtx_size)
	for lidx, line in enumerate(open(datafile)):
		parts = line.strip().split("\t")
		# get the column indices from the strings
		cols = [int(x.split(":")[0]) for x in parts[1:]]
		# the row index is always the same as the line index
		rows = [lidx] * len(cols)
		# get the values from the strings
		vals = [int(x.split(":")[1]) for x in parts[1:]]
		#update the sparse matrix 
		for r, c, v in zip(rows, cols, vals):
			X[r, c] += v
		# Let Liang know the script is still working!
		if (lidx + 1) % 1000 == 0:
			print "Parsed %d docs." % (lidx + 1,)

	X = X.tocsr()
	print "We now have a CSR matrix! Sleep!!"
	
	corpus =  gensim.matutils.Sparse2Corpus(X,False)
	lda = models.LdaModel(corpus,ntopics)
	return lda
	
def getClassifier(filename='inf200.txt',row1=7061,row2=87474):
	nrows = row1+row2
	
	x = readDataset(filename)
	
	y = np.ones(nrows)
	y[row1:nrows]=0
	
	#X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1, random_state=0)
	kf = cross_validation.KFold(x.shape[0], n_folds=10, shuffle=True)
	for train, test in kf:
		xtrain = x[train, :]
		ytrain = y[train]
		xtest = x[test, :]
		ytest = y[test]
		md = svm.SVC(kernel='linear', C=1.0,max_iter=500)
		md.fit(xtrain, ytrain)
		ypred=md.predict(xtest)
		print precision_score(ytest, ypred), recall_score(ytest, ypred)

def readDataset(filename,row1=7061,row2=87474):
#matrix = ldabot.readDataset('ldaoutput.txt') 
	matrix = genfromtxt(filename, delimiter=' ')
	return matrix