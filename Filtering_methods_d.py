import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from collections import defaultdict
import astropy.visualization as av
import time
import os
import sys
from scipy.optimize import *
import copy
from astropy.stats import bayesian_blocks
from datetime import datetime


Inf=float('inf')

########################################################################################################
class Network:
	def __init__(self, N, x):
	    	self.N = N #number of nodes in the network
	    	self.T = x[:,0].max()+1 #total observation window
        
	def Bayesian(self,x): 
		I = bayesian_blocks(x[:,0], p0=0.01) #Interval partition
		I = np.ceil(I).astype(int)
		self.I = zip(I[:-1],I[1:]) 
		if self.I[-1][0]==self.I[-1][1]:
			del self.I[-1]
		
		
	def weights_links(self, x):
		self.links = {} #All links observed at least once in the overall network evolution
		self.wo = {} #weight carried by each link in the network
		
		for i in range(len(x)):
		    self.wo.setdefault((x[i][1], x[i][2]), 0) 
		    self.wo[(x[i][1], x[i][2])] += 1
		    self.links[(x[i][1], x[i][2])] = None
		    
		    self.wo.setdefault((x[i][2], x[i][1]), 0) 
		    self.wo[(x[i][2], x[i][1])] += 1
		    self.links[(x[i][2], x[i][1])] = None
		    
		self.tot_strength = np.sum(self.wo.values())
	
	
	def compute_weEADAM_Accuracy(self, x):
		#only to plot W(t) as a function of time
		self.weEADAM = np.zeros((self.N,self.N))
		
		x = np.array([[a]+[b,c] for a,b,c in x if b!=c] + [[a]+[c,b] for a,b,c in x if b!=c])	#we always remove self-loopss
		
		for i,j in self.I:
		    sqin = np.zeros(self.N) #incoming strength
		    sqout = np.zeros(self.N) #outgoing strength
		    if int(j)==self.T-1:
				ed = x[(x[:,0]>=i)*(x[:,0]<=j),1:]
		    else:	
				ed = x[(x[:,0]>=i)*(x[:,0]<j),1:]
		    
		    if len(ed)==0: continue
		    
		    r = Counter(ed[:,0])
		    sqin[np.array(r.keys())] = r.values()
		    
		    r = Counter(ed[:,1])
		    sqout[np.array(r.keys())] = r.values()
		    tot = sqin.sum()
		    
		    self.weEADAM+=np.outer(sqin,sqout)/(float(tot)-1) 
		
		
	
	def compute_weEADAM(self, x):	
		self.weEADAM = {}#expected weigth according to the EADAM
		for i,j in self.links:
			self.weEADAM.setdefault((i, j), 0.)
		
		x = np.array([[a]+[b,c] for a,b,c in x if b!=c] + [[a]+[c,b] for a,b,c in x if b!=c])	#we always remove self-loops
		
		for i,j in self.I:
		    sqin = np.zeros(self.N) #incoming strength
		    sqout = np.zeros(self.N) #outgoing strength
		    if int(j)==self.T-1:
				ed = x[(x[:,0]>=i)*(x[:,0]<=j),1:]
		    else:	
				ed = x[(x[:,0]>=i)*(x[:,0]<j),1:]
		    
		    if len(ed)==0: continue
		    
		    r = Counter(ed[:,0])
		    sqin[np.array(r.keys())] = r.values()
		    
		    r = Counter(ed[:,1])
		    sqout[np.array(r.keys())] = r.values()
		    tot = sqin.sum()

		    for l,k in self.links:
		    	self.weEADAM[(l,k)] += sqin[l]*sqout[k]/float(tot-1)
			    			
	
	def compute_strength_SVN(self, x):
		x = np.array([[a]+[b,c] for a,b,c in x if b!=c] + [[a]+[c,b] for a,b,c in x if b!=c])	#we always remove self-loops
	
		self.strength_a_out, self.strength_a_in = {}, {}
		for i in range(len(x)):
		    self.strength_a_out.setdefault(x[i][1], 0)
		    self.strength_a_in.setdefault(x[i][2], 0)
		    self.strength_a_out[x[i][1]]+=1
		    self.strength_a_in[x[i][2]]+=1
	

########################################################################################################
def Read_file(path, name_file, multiedges, dt, column_time, sep, remove_nights):
	x = pd.read_csv(path+name_file,sep=sep,header=None)	
	
	if column_time==0:
		x = np.array(zip(x[0],x[1],x[2]))
	elif column_time==1:
		x = np.array(zip(x[1],x[0],x[2]))
	elif column_time==2:	
		x = np.array(zip(x[2],x[0],x[1]))
	else:	
		sys.exit('please make sure your columns for time and edgelist are within the first three')
		
	x = np.array([[a]+[b,c] for a,b,c in x if b!=c])	#we always remove self-loops
	x[:,0] = x[:,0]-x[:,0].min()
	x[:,0] = x[:,0]/dt
	
	if multiedges == 'no':
		x = np.array(sorted(set(map(tuple,x))))
	elif multiedges != 'yes':
		sys.exit('multiedges requires either yes or no')

	x[:,0] = x[:,0]-x[:,0].min()
	ID = np.unique(np.concatenate((x[:,1],x[:,2])))
	ID = dict(zip(ID,range(len(ID))))
	N = len(ID)

	for i in range(len(x)):
	    x[i,1] = ID[x[i,1]]
	    x[i,2] = ID[x[i,2]]
	    
	x = x[x[:,1]!=x[:,2]]
	
	if remove_nights == 'yes': 
		c = sorted(Counter(x[:,0]).keys())
		c = dict(zip(c,range(len(c))))
		x[:,0] = [c[i] for i in x[:,0]]
	elif remove_nights != 'no':
		sys.exit('remove_nights requires either yes or no')
	
	return x, N
	
#######################################################################################################################
def Relative_error_weEADAM_Accuracy(Net, myfile_relative_error, dt):
	#Relative Error
	RE = lambda th,exp: float(abs(th-exp))/th
	
	wo = 0.
	weEADAM = 0.
	for i in range(len(Net.weEADAM)):
		for j in range(len(Net.weEADAM)):
			if i!=j:
				weEADAM += Net.weEADAM[i,j]
				
	for i,j in Net.wo:
		wo += Net.wo[i,j]

	
	myfile_relative_error.write(str(dt)+'	'+str(RE(wo, weEADAM))+'\n')	
	
########################################################################################################
def links_EADAM(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt):	
	if Bonferroni_corr=='yes':
		nl = len(Net.links)/2.#total number of links in the network
	else:
		nl = 1. #no Bonferroni correction!
	
	edEADAM =  [(i,j) for i,j in Net.links if st.poisson.sf(Net.wo[i,j]-1,Net.weEADAM[i,j])<alpha/nl]
	
	for i in range(len(edEADAM)):
		myfile_edgelist.write(str(edEADAM[i][0])+'	'+str(edEADAM[i][1])+'\n')
	
	myfile_links.write(str(dt)+'	'+str(len(edEADAM))+'\n')
	
########################################################################################################
def links_SVN(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt):	
	if Bonferroni_corr=='yes':
		nl = len(Net.links)/2.#total number of links in the network
	else:
		nl = 1. #no Bonferroni correction!
	
	edSVN =  [(i,j) for i,j in Net.links if st.hypergeom.sf(Net.wo[i,j]-1, Net.tot_strength, Net.strength_a_out[i], Net.strength_a_in[j])<alpha/nl]
	
	for i in range(len(edSVN)):
		myfile_edgelist.write(str(edSVN[i][0])+'	'+str(edSVN[i][1])+'\n')
	
	myfile_links.write(str(dt)+'	'+str(len(edSVN))+'\n')
	
########################################################################################################
'it is faster for small networks and  weEADAM is equal to wo. For very large networks (N>10^5 or 10^4) using this code Memory Error occurs.'
def Backbone_detection(path_in = 'DATASET/Example_PrimarySchool/', name_file='primaryschool.csv', sep='\t', dt=[20., 60.*1., 60.*5., 60.*15.], multiedges = ['no'], directory_out = 'Output', column_time=0, remove_nights='yes', alpha = 0.01, Bonferroni_corr = 'yes', model = 'EADAM'):
	time_start = time.time() 
	
	if multiedges == ['no']:
		multiedges = ['no']*len(dt)
	
	if len(dt)!=len(multiedges):
		sys.exit('dt and multiedges must have same dimensions')
	path_out = path_in+directory_out
	os.system("mkdir	"+path_out)
	myfile_links = open(path_out+'/links_'+model+'.txt','w')
	if model=='EADAM_Accuracy':
		myfile_relative_error = open(path_out+'/relative_error_'+model+'.txt','w')

	for i in range(len(dt)):
		myfile_edgelist = open(path_out+'/edgelist_'+str(int(dt[i]))+'_'+model+'.txt','w')
		
		x, N = Read_file(path_in, name_file, multiedges[i], dt[i], column_time, sep, remove_nights)
		Net = Network(N, x)
		Net.weights_links(x)
		
		if model=='EADAM' or model=='EADAM_Accuracy':
			Net.Bayesian(x)
		
		
		if model=='EADAM_Accuracy':
			Net.compute_weEADAM_Accuracy(x) 
			Relative_error_weEADAM_Accuracy(Net, myfile_relative_error, dt[i])
			links_EADAM(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt[i])
		elif model=='EADAM':
			Net.compute_weEADAM(x)	
			links_EADAM(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt[i])
		elif model=='SVN':
			Net.compute_strength_SVN(x)
			links_SVN(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt[i])
		else:
			sys.exit('Three are the acceptable keywords: SVN, EADAM, or EADAM_Accuracy. In the ReadMe file we explain their meaning')
		 
		myfile_edgelist.close()
		
	myfile_links.close()
	if model=='EADAM_Accuracy':
		myfile_relative_error.close()
	
	elapsed = time.time() - time_start
	print "cpu time: ", elapsed

