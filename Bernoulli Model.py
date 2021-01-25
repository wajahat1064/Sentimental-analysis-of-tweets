import pandas as pd
import numpy as np
import time
import sys
import math

#importing the dataset as a numpy dataframe
test_x = pd.read_csv('question-4-test-features.csv')
test_y = pd.read_csv('question-4-test-labels.csv')
train_x = pd.read_csv('question-4-train-features.csv')
train_y = pd.read_csv('question-4-train-labels.csv')

# Numpy dataframe is converted to a numpy array so that the features can be assessed
train_x = np.array(train_x.iloc[:,:].as_matrix()) #train features
train_y = np.array(train_y.iloc[:,:].as_matrix()) #train labels
test_x = np.array(test_x.iloc[:,:].as_matrix())   #test features
test_y = np.array(test_y.iloc[:,:].as_matrix())   #test labels
train_x[train_x>0] =1 #All the values greater than 1 will be changed to 1 according to the Bernoulli MODEL
test_x[test_x>0]=1
#This function gets the count of the all the features
def get_feature_counts(train_x, train_y): 
    words= train_x.shape[1]
    tweets=train_x.shape[0]
    #generating a null array to be filled later
    pos_vector= np.zeros((words))  
    neg_vector=np.zeros((words))
    neut_vector=np.zeros((words))
    for i in range(0,words):
      cnt_pst=0
      cnt_neg=0
      cnt_neut=0
      for j in range(0,tweets):  
          number = train_x[j,i]
          if train_y[j] == 'positive':
            pos_vector[i] = pos_vector[i] + number
          if train_y[j] == 'negative':
            neg_vector[i] = neg_vector[i]+ number
          if train_y[j] == 'neutral':
            neut_vector[i]= neut_vector[i]+ number
            
    return pos_vector, neg_vector, neut_vector

pos_vector, neg_vector, neut_vector = get_feature_counts(train_x,train_y)

#This function calculates the probability of likelihood function
def probabilityoflikelihood(pos_vector,neg_vector, neut_vector): 
  denforpositive= np.sum(pos_vector)
  denfornegative=np.sum(neg_vector)
  denforneutral=np.sum(neut_vector)
  return pos_vector/denforpositive, neg_vector/ denfornegative,neut_vector/ denforneutral

problp,probln,problnu = probabilityoflikelihood(pos_vector,neg_vector, neut_vector)
print(problp,probln,problnu)
#This function calculates and returns the probability of the prior function
def probabilityofprior(train_y):
  count_positive=0
  count_negative=0
  count_neutral=0
  for i in range(0,len(train_y)):
    if train_y[i] == 'positive':
      count_positive+= 1
    if train_y[i] == 'negative':
      count_negative+= 1
    if train_y[i] == 'neutral':
      count_neutral+= 1
  return (count_positive/len(train_y)), (count_negative/len(train_y)), (count_neutral/len(train_y))

probpp,probpn,probpnu= probabilityofprior(train_y)
print(probpp,probpn,probpnu)

#This function includes the testing and mapping for multinomial naive bayes model
def bernoullifinalprobability(test_x,test_y):
    score=0
    words= test_x.shape[1]
    tweets=test_x.shape[0]
    pos_vector= np.zeros((words))
    neg_vector=np.zeros((words))
    neut_vector=np.zeros((words))
    for i in range(0,tweets):
      finalpp=1
      finalpn=1
      finalpnu=1
      ninf= float('-Inf')
      for j in range(0,words):  
          number = test_x[i,j]
         # print(number)
          finalpp *=   (number)*(problp[j]) + (1-number)*(1-problp[j])
          #print('finalpp: '+ str(finalpp))
          finalpn *=   (number)*(probln[j]) + (1-number)*(1-probln[j])
          #print('finalpn: '+ str(finalpn))
          finalpnu *=   (number)*(problnu[j]) + (1-number)*(1-problnu[j])
      if np.log(finalpp) == ninf:
        finalpp += -2000000
      else:
        finalpp += np.log(finalpp)
      if np.log(finalpn) == ninf:
        finalpn += -2000000 
      else:
        finalpn += np.log(finalpn)
      if np.log(finalpnu)== ninf:
        finalpnu += -2000000
      else:
        finalpnu += np.log(finalpnu)
        
      if math.log(probpp) == ninf:
        finalpp += -2000000
      else:
        finalpp += math.log(probpp)
      if math.log(probpn) ==ninf:
        finalpn += -2000000
      else:
        finalpn += math.log(probpn)
      if math.log(probpnu) ==ninf:
        finalpnu += -2000000
      else:
        finalpnu += math.log(probpnu)

      vector = np.zeros((3))
      vector[0]= finalpp
      vector[1]= finalpn
      vector[2]=finalpnu
      finalvalue = np.argmax(vector)
      if test_y[i] == 'positive' and finalvalue == 0:
          score+=1 #score keeps increasing when the predicted result is the same as the test label
          print('pos: '+str(score))
      if test_y[i] == 'negative' and finalvalue == 1:
          score+=1
          print('neg: '+str(score))
      if test_y[i] == 'neutral' and finalvalue == 2:
          score+=1
          print( 'neu: '+ str(score))
      
        
    return int(score/len(test_y))
y= bernoullifinalprobability(test_x,test_y)
print(y)
