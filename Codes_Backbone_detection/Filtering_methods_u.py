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
import itertools
from scipy.optimize import *
import copy
from astropy.stats import bayesian_blocks
from datetime import datetime
import csv


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
		

	def Find_Intervals(self, x, N_I):
		self.N_I = N_I #number of estimated intervals
		self.t_estimated = []#time between two successive intervals
		
		self.W_t = [0. for _ in itertools.repeat(None, self.T)] #total number of links per snapshot
		for i in range(len(x)):
			self.W_t[x[i][0]] += 1
		
		path_bin = './oksPublic/oksPublic/binrt/' #path to executable directory, from which the segmentation code runs
		name_exe = 'oksm' #segmentation code chosen
		
		path_data = './oksPublic/oksPublic/dataset/' #path to dataset directory, from which we analize the data
		name_data_file = 'time_series.csv' #name of the data file
		
		#actual function
		with open(path_data+name_data_file, mode='w') as csvfile:
	    		writer = csv.writer(csvfile, delimiter=',', quotechar='|')
    			for t in range(self.T):
	    			writer.writerow(['S', self.W_t[t], t, (t+0.999)])#I assumed that the time step is equal to 1. 
	    	
	    	if self.N_I>1:
	    		a = []
	    		while(len(a)<self.N_I):
			    	os.system(path_bin+name_exe+' -i '+path_data+name_data_file+' -n '+str(self.T)+' -c '+str(self.N_I)+' -t 1 -o ./  > a.txt')#I assumed that the time step is equal to 1. 
			    	myfile = open('a.txt', 'r')

			    	for i in myfile:
			    		x1, x2 = i.split()
			    		a = x2.split(',')
			    		del a[-1]
			    		break
			    		
		    	self.t_estimated.append(0)
	    		for i in range(self.N_I):
	    			self.t_estimated.append(int(a[self.N_I-1-i]))
	    		self.t_estimated.append(self.T)
		else:
			self.t_estimated.append(0)#first time is 0
			self.t_estimated.append(self.T)#last time
		
		self.I = []
		for i in range(len(self.t_estimated)-1):
			self.I.append((self.t_estimated[i], self.t_estimated[i+1]))

		if self.I[0][0]==self.I[0][1]:
			del self.I[0]
			
		if self.I[-1][0]==self.I[-1][1]:
			del self.I[-1]
		
		
		
	def weights_links(self, x):
		self.links = {} #All links observed at least once in the overall network evolution
		self.wo = {} #weight carried by each link in the network
		
		for i in range(len(x)):
		    self.wo.setdefault((x[i][1], x[i][2]), 0) 
		    self.wo[(x[i][1], x[i][2])] += 1
		    self.links[(x[i][1], x[i][2])] = None
		    
		self.tot_strength = 2*np.sum(self.wo.values())
	
        def compute_weEADM_Accuracy(self, x):
		#only to plot W(t) as a function of time
		self.weEADM = np.zeros((self.N,self.N))
		for i,j in self.I:
		    sq = np.zeros(self.N)
		    if int(j)==self.T-1:
				ed = x[(x[:,0]>=i)*(x[:,0]<=j),1:]
		    else:	
				ed = x[(x[:,0]>=i)*(x[:,0]<j),1:]
		    if len(ed)==0: continue
		    r = Counter(ed[:,0])
		    sq[np.array(r.keys())] = r.values()
		    r = Counter(ed[:,1])
		    sq[np.array(r.keys())] += r.values()
		    tot = sq.sum()
		    self.weEADM+=np.outer(sq,sq)/(float(sq.sum())-1) 
		    
		
	def compute_weEADM(self, x):	
		self.weEADM = {}#expected weigth according to the EADM
		for i,j in self.links:
			self.weEADM.setdefault((i, j), 0.)
			
		for i,j in self.I:
		    sq = np.zeros(self.N)
		    if int(j)==self.T-1:
				ed = x[(x[:,0]>=i)*(x[:,0]<=j),1:]
		    else:	
				ed = x[(x[:,0]>=i)*(x[:,0]<j),1:]
		    
		    if len(ed)==0: continue
		    
		    r = Counter(ed[:,0])
		    
		    sq[np.array(r.keys())] = r.values()
		    r = Counter(ed[:,1])
		    sq[np.array(r.keys())] += r.values()
		    tot_strength = sq.sum()

		    for l,k in self.links:
		    	self.weEADM[(l,k)] += sq[l]*sq[k]/float(tot_strength-1)
	
	
	def compute_strength_SVN(self, x):
		self.strength_a = {}
		for i in range(len(x)):
		    self.strength_a.setdefault(x[i][1], 0)
		    self.strength_a.setdefault(x[i][2], 0)
		    self.strength_a[x[i][1]]+=1
		    self.strength_a[x[i][2]]+=1
	

	def initialize_variables_likelihood(self, x):
		self.w_ij = Counter(map(tuple,x[:,1:]))

		self.strength_a = {}
		for i in range(len(x)):
		    self.strength_a.setdefault(x[i][1], 0)
		    self.strength_a.setdefault(x[i][2], 0)
		    self.strength_a[x[i][1]]+=1
		    self.strength_a[x[i][2]]+=1
		
		self.optimal_activity = {} #key 1: source node, value: optimal value of the activity 
		self.m_ij_o = defaultdict(dict) #key 1: source node, key 2: target node, value: weight WEIGHTS CAN BE ZERO
		    
        
        
        def compute_expected_weigths_TFM(self):
        	self.weTFM = np.zeros((len(self.optimal_activity), len(self.optimal_activity)))
		for i in range(len(self.optimal_activity)):
			for j in range(len(self.optimal_activity)):
				self.weTFM[i][j] += self.optimal_activity[i]*self.optimal_activity[j]*self.T
        
        
	def initialize_m_ij_o(self):
		for i in self.strength_a:
		    for j in self.strength_a:
		        if i<j:
		            self.w_ij.setdefault((i,j),0)
		            self.m_ij_o[i][j] = self.w_ij[(i,j)]
		            
		for i in self.strength_a:
		    for j in self.strength_a:
		        if j<i:
		            self.w_ij.setdefault((i,j),0)
		            self.m_ij_o[i][j] = self.m_ij_o[j][i]           		
		
		            
	def function_source(self, source_node):
		supp= 0.
		for target_node in self.m_ij_o[source_node]:
		    if target_node != source_node:
		        supp += float(self.m_ij_o[source_node][target_node] - float(self.T)*self.optimal_activity[source_node]*self.optimal_activity[target_node])/(1. - self.optimal_activity[source_node]*self.optimal_activity[target_node])
		return supp

	'N fuctions, one for each node, optimized according to the ML'	
	def Functions_to_be_optimized(self, z):
		i = 0
		for source_node in self.m_ij_o:
		    self.optimal_activity[source_node] = float(z[i])
		    i+=1
		        
		        
		tot_equations = len(self.m_ij_o)
		H = np.empty((tot_equations))
		    
		i = 0
		for source_node in self.m_ij_o:
		    H[i] = self.function_source(source_node)#
		    i+=1
		
		return H
            
            
	'Optimal value of the activity, according to ML estimation'	
	def estimate_activity(self):
		z_Guess = []
		activity_guess = {}
		for source_node in self.m_ij_o:
		    #self.optimal_activity[source_node] = 0.
		    activity_guess[source_node] = float(self.strength_a[source_node])/np.sqrt((self.tot_strength-1)*self.T)

		for source_node in self.m_ij_o:
		    z_Guess.append(float(activity_guess[source_node]))
		        
		z = fsolve(self.Functions_to_be_optimized, z_Guess, factor = 10)
		    
		i=0
		for source_node in self.m_ij_o:
		    self.optimal_activity[source_node] = float(z[i])
		    i+=1
			    			
        
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
		
	x = np.array([[a]+sorted([b,c]) for a,b,c in x if b!=c])	#we always remove multiple links
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
def Relative_error_weEADM_Accuracy(Net, myfile_relative_error, dt):
	#Relative Error
	RE = lambda th,exp: float(abs(th-exp))/th
	
	wo = 0.
	weEADM = 0.
	for i in range(len(Net.weEADM)):
		for j in range(len(Net.weEADM)):
			if i<j:
				weEADM += Net.weEADM[i,j]
				
	for i,j in Net.wo:
		wo += Net.wo[i,j]

	
	myfile_relative_error.write(str(alpha)+'	'+str(RE(wo, weEADM))+'\n')	
	
#######################################################################################################################
def Relative_error_weTFM(Net, myfile_relative_error, dt):
	#Relative Error
	RE = lambda th,exp: float(abs(th-exp))/th
	
	wo = 0.
	weTFM = 0.
	for i in range(len(Net.weTFM)):
		for j in range(len(Net.weTFM)):
			if i<j:
				weTFM += Net.weTFM[i,j]
				
	for i,j in Net.wo:
		wo += Net.wo[i,j]

	myfile_relative_error.write(str(alpha)+'	'+str(RE(wo, weTFM))+'\n')	

########################################################################################################
def links_TFM(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt):	
	if Bonferroni_corr=='yes':
		nl = len(Net.links)#total number of links in the network
	else:
		nl = 1. #no Bonferroni correction!
	
	edTFM = [(i,j) for i,j in Net.links if st.binom.sf(Net.wo[i,j]-1,Net.T, Net.optimal_activity[i]*Net.optimal_activity[j])<alpha/nl]
	
	for i in range(len(edTFM)):
		myfile_edgelist.write(str(edTFM[i][0])+'	'+str(edTFM[i][1])+'\n')
	
	myfile_links.write(str(alpha)+'	'+str(len(edTFM))+'\n')
	
########################################################################################################
def links_EADM(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt):	
	if Bonferroni_corr=='yes':
		nl = len(Net.links)#total number of links in the network
	else:
		nl = 1. #no Bonferroni correction!
	
	edEADM =  [(i,j) for i,j in Net.links if st.poisson.sf(Net.wo[i,j]-1,Net.weEADM[i,j])<alpha/nl]
	
	for i in range(len(edEADM)):
		myfile_edgelist.write(str(edEADM[i][0])+'	'+str(edEADM[i][1])+'\n')
	
	myfile_links.write(str(alpha)+'	'+str(len(edEADM))+'\n')
	
########################################################################################################
def links_SVN(Net, myfile_links, myfile_edgelist, alpha, Bonferroni_corr, dt):	
	if Bonferroni_corr=='yes':
		nl = len(Net.links)#total number of links in the network
	else:
		nl = 1. #no Bonferroni correction!
	
	edSVN =  [(i,j) for i,j in Net.links if st.hypergeom.sf(Net.wo[i,j]-1, Net.tot_strength, Net.strength_a[i], Net.strength_a[j])<alpha/nl]
	
	for i in range(len(edSVN)):
		myfile_edgelist.write(str(edSVN[i][0])+'	'+str(edSVN[i][1])+'\n')
	
	myfile_links.write(str(alpha)+'	'+str(len(edSVN))+'\n')
	
########################################################################################################
'it is faster for small networks and  weEADM is equal to wo. For very large networks (N>10^5 or 10^4) using this code Memory Error occurs.'
def Backbone_detection(path_in = 'DATASET/Example_PrimarySchool/', name_file='primaryschool.csv', sep='\t', dt=[20., 60.*1., 60.*5., 60.*15.], multiedges = ['no'], directory_out = 'Output', column_time=0, remove_nights='yes', alpha = [0.01], Bonferroni_corr = 'yes', model = 'EADM_BB', N_I = 1.):
	time_start = time.time() 
	
	if multiedges == ['no']:
		multiedges = ['no']*len(dt)
	
	if len(dt)!=len(multiedges):
		sys.exit('dt and multiedges must have same dimensions')
	path_out = path_in+directory_out
	os.system("mkdir	"+path_out)
	
	

	for i in range(len(dt)):
		myfile_links = open(path_out+'/links_'+str(int(dt[i]))+'_'+model+'.txt','w')
		
		if model=='EADM_BB_Accuracy' or model=='EADM_I_Accuracy' or model=='TFM':
			myfile_relative_error = open(path_out+'/relative_error_'+str(int(dt[i]))+'_'+model+'.txt','w')
		
		x, N = Read_file(path_in, name_file, multiedges[i], dt[i], column_time, sep, remove_nights)
		Net = Network(N, x)
		Net.weights_links(x)
		
		if model=='EADM_BB' or model=='EADM_BB_Accuracy':
			Net.Bayesian(x)
		elif model=='EADM_I' or model=='EADM_I_Accuracy':
			Net.Find_Intervals(x, N_I)
		
		if model=='EADM_BB_Accuracy' or model=='EADM_I_Accuracy':
			Net.compute_weEADM_Accuracy(x) 
			Relative_error_weEADM_Accuracy(Net, myfile_relative_error, dt[i])
		elif model=='EADM_BB' or model=='EADM_I':
			Net.compute_weEADM(x)	
		elif model=='SVN':
			Net.compute_strength_SVN(x)
		elif model=='TFM':	
			Net.initialize_variables_likelihood(x)
			Net.initialize_m_ij_o()
			Net.estimate_activity()
			Net.compute_expected_weigths_TFM()
			Relative_error_weTFM(Net, myfile_relative_error, dt[i])
		else:
			sys.exit('These are the acceptable keywords: TFM, SVN, EADM_BB, EADM_I, EADM_I_Accuracy, or EADM_BB_Accuracy. In the ReadMe file we explain their meaning')
			
		for j in range(len(alpha)):	
			myfile_edgelist = open(path_out+'/edgelist_'+str(int(dt[i]))+'_'+str(alpha[j])+'_'+model+'.txt','w')
			if model=='EADM_BB_Accuracy' or model=='EADM_I_Accuracy':
				links_EADM(Net, myfile_links, myfile_edgelist, alpha[j], Bonferroni_corr, dt[i])
			elif model=='EADM_BB' or model=='EADM_I':
				links_EADM(Net, myfile_links, myfile_edgelist, alpha[j], Bonferroni_corr, dt[i])
			elif model=='SVN':
				links_SVN(Net, myfile_links, myfile_edgelist, alpha[j], Bonferroni_corr, dt[i])
			elif model=='TFM':	
				links_TFM(Net, myfile_links, myfile_edgelist, alpha[j], Bonferroni_corr, dt[i])
			else:
				sys.exit('These are the acceptable keywords: TFM, SVN, EADM_BB, EADM_I, EADM_I_Accuracy, or EADM_BB_Accuracy. In the ReadMe file we explain their meaning')
			
		 	myfile_edgelist.close()
		 	
		myfile_links.close()
		
		
	
		if model=='EADM_BB_Accuracy' or model=='EADM_I_Accuracy' or model=='TFM':
			myfile_relative_error.close()
	
	elapsed = time.time() - time_start
	print "cpu time: ", elapsed


    
