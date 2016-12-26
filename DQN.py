
# coding: utf-8

# In[1]:
import sys
sys.path.insert(0, 'Resources/MagicCube/code/')
import os
import matplotlib.pyplot as plt
import random as rd
from bitstring import BitArray
import math
import copy
import tensorflow as tf
import random
import pickle

from cube import *
from utils import *


# In[2]:

N = 3 #cube size

#Q-learning parameters
r = 0.15
gamma = 1 / (1 + r) #discount of the model
C = 1.
epsilon = 0.05
beta = 3./4 


# In[3]:

def all_actions(N): #rotate by +90° / by -90° 
    actions = []
    c = Cube(N)
    for face_name in ["F","U","R"]: #the list is in the end ['U','D','F']
        for layer in range(c.N):
            for times in [1,-1]:
                actions.append([face_name,layer,times])
    return actions

# def reward_cube(c):
#     edges = computeEdges(c)
#     corners = computeCorners(c)
#     ncf = numCompleteFaces(c)
#     nce = numCompleteEdges(c,edges)
#     ncc = numCompleteCorners(c,corners)
    
# #     return (-1 + 10*ncf + 2*nce + 3*ncc + 100*(ncf == 6))/700
#     return (ncf == 6)

def reward_cube(c):
    ncf = numCompleteFaces(c)
    return (-1 + entropy(c) + 100*(ncf == 6))/100
#    return (ncf == 6)

def entropy(c):
    ent = 0
    for f in range(6):
        pi = len(np.unique(c.stickers[f]))
        ent -= pi*np.log(pi)       
    return ent
        

def state_cube(c):
    #determining the new state
    edges = computeEdges(c)
    corners = computeCorners(c)
    edges_state = []
    corners_state = []
    faces_state = []
#     for e in edges:
#         edges_state.append(e.isDone(c))
    for corner in corners:
        corners_state.append(corner.isDone(c))
    nFaces = 6
    for f in range(nFaces):
        faces_state.append(np.sum(c.stickers[f] != c.stickers[f,0,0]) == 0)
#     #conversion from binary list to int
#     e = BitArray(edges_state).uint
    c = BitArray(corners_state).uint
    f = BitArray(faces_state).uint
#     ncf = numCompleteFaces(c)
#     nce = numCompleteEdges(c,edges)
#     ncc = numCompleteCorners(c,corners)
#     return ncf,nce,ncc
    return c,f

def test_function_state_cube():
    c = Cube(3)
    print(state_cube(c))
    c.randomize(1)
    print(state_cube(c))
    


# In[4]:

actions = all_actions(N) #rotate by +90° / by -90° 
nb_actions = len(actions)

class network():
    
    def __init__(self,W1,W2,b1,b2):
        
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        
        self.Q1 = tf.matmul(x/6,self.W1) + self.b1
        self.Qs1 = tf.nn.tanh(self.Q1)
        self.Q2 = tf.matmul(self.Qs1,self.W2) + self.b2 #tf.nn.relu(tf.matmul(Qs1,W2))# + b2)
        
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
         
        self.network_params = tf.trainable_variables()
        self.tau = .001
    
    
    
    def update_target_network_(self, trainNet):
        
        self.update_target_network_params = \
            [self.network_params[i].assign(tf.mul(trainNet.network_params[i], self.tau) + \
                tf.mul(self.network_params[i], 1. - self.tau)) \
                for i in range(len(self.network_params))]


    def update_target_network(self):
        
        self.sess.run(self.update_target_network_params)

# In[29]:

c_init=Cube(3)

resume = sys.argv[1] == "True"

#==============================================================================
#                           DEFINE NEURAL NETWORK
#==============================================================================

# sess = tf.InteractiveSession() 
with tf.device("/gpu:0"):
 
    x = tf.placeholder(tf.float32, shape=[None,6*c_init.N**2])
    Q_ = tf.placeholder(tf.float32, shape=[None,nb_actions])
    Qs = tf.placeholder(tf.float32, shape=[None,nb_actions])
 
 
    if not resume:
        W1 = tf.Variable(tf.random_normal([6*c_init.N**2,5000], stddev=1e-2))
        b1 = tf.Variable(tf.random_normal([5000], stddev=1e-2))
 
        W2 = tf.Variable(tf.random_normal([5000,nb_actions], stddev=1e-2))
        b2 = tf.Variable(tf.random_normal([nb_actions], stddev=1e-2))  
    else:
        load = pickle.load(open('save.p', 'rb'))
        W1 = tf.Variable(load[0])
        W2 = tf.Variable(load[1])
        b1 = tf.Variable(load[2])
        b2 = tf.Variable(load[3])
 
 
    trainNet = network(W1,W2,b1,b2)
    
    targetNet = network(W1,W2,b1,b2)
    
#     Qs = targetNet.Q2

    
    loss_function = tf.reduce_mean(tf.square(tf.sub(Q_,trainNet.Q2)))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    trainNet.sess.run(init_op)
    targetNet.sess.run(init_op)
    
    targetNet.update_target_network_(trainNet)
    
#==============================================================================


# In[37]:

def DQN(c_init,Tmax,nb_episodes, n_moves):
    detail_mode = False #prints lots of things. Put = True for one move only

    plt.ion()
    done = 0
    lActions = np.zeros(18)
    lActions_batch = np.zeros(lActions.shape[0] + 1) #shape different from lActions; see actualization.
    print("moves","\t","ep.","\t","Loss Function","\t","Min Q","\t\t", "Reward", "", "NB.","\t", "Prcent.","\t","Mn. Prcent.")
    
    mineps = 0.1
    def eps(episode):
        return min(1,max(mineps,100/episode))
    
    lenBatch = 1
    episode = 1
    percentDone = []
    tries = 1
    dones = np.empty([0])
    targetNet.tau = 1
    D = []
    
    while np.sum(dones[-1000:])/min(1000,tries) < .8 and episode < nb_episodes:  
        
        episode += 1
        
        s = copy.deepcopy(c_init)
#        print(targetNet.sess.run(targetNet.Q2,feed_dict={x:np.reshape(s.stickers,(1, 54))}))
        while numCompleteFaces(s) == 6: #in order not to start with the solved cube
            s.randomize(n_moves) #we randomize n_moves times in order to have a "well mixed" cube
#        s.move("R",2,-1)
        cum_reward = []
        cum_reward_fill = 0
        
        tries += 1
        done = 0
        
        for i in range(Tmax):
            
            #Choose an action by greedily (with e chance of random action) from the Q-network
            S = copy.copy(np.reshape(s.stickers,(1, 54)))
            Qout = targetNet.sess.run(targetNet.Q2,feed_dict={x:S})
            if(rd.random() > eps(episode)):
                a = np.argmax(Qout)
                if detail_mode:
                    print("Maximizing (choice of action)")
            else:
                a = rd.randint(0,nb_actions-1)
                if detail_mode:
                    print("Random (choice of action)")
            lActions[a] += 1
            if detail_mode:
                print("action:",actions[a])
            #print(Qout)
            
            #Get new state and reward from environment
            f,l,d = actions[a]
            #print(actions[a])
            s.move(f,l,d)            
            r = reward_cube(s)
            cum_reward_fill += r
            D.append(copy.deepcopy([S, a, r, np.reshape(s.stickers,(1, 54)), numCompleteFaces(s)]))
            

#==============================================================================
#            if numCompleteFaces(s) != 6:
#                #Obtain the Q' values by feeding the new state through our network
#                #print(S)
#                #print(np.reshape(s.stickers,(1, 54)))
#                Qprime = targetNet.sess.run(targetNet.Q2,feed_dict={x:np.reshape(s.stickers,(1, 54))})
#                
#                #Obtain maxQ' and set our target value for chosen action.
#                maxQprime = np.max(Qprime)
#                targetQ = Qout
#                #print(targetQ)
#                if detail_mode:
#                    print("not solved")
#                    print("before we have:",targetQ[0,a],"after we want:", r + gamma*maxQprime)
#                targetQ[0,a] = r + gamma*maxQprime
#            else:
#                targetQ = Qout
#                #print(targetQ)
#                if detail_mode:
#                    print("solved")
#                    print("before we have:",targetQ[0,a],"after we want:", r)
#                targetQ[0,a] = r
#            
#            #Train our network using target and predicted Q values
#            if detail_mode:
#                print("effect (with targetNet.tau =",targetNet.tau,"):")
#                print(targetNet.sess.run(targetNet.Q2,feed_dict={x: S}))
##            trainNet.sess.run(train_step,feed_dict={Q_: targetQ, x: S}) #what was written before by Marc
#            targetNet.sess.run(train_step,feed_dict={Q_: targetQ, x: S}) #what Vincent modified (=> only one network)
##            print(trainNet.sess.run(trainNet.Q2,feed_dict={x: S})) #to uncomment when the target and train networks will work
#            if detail_mode:
#                print(targetNet.sess.run(targetNet.Q2,feed_dict={x: S})) #try by Vincent (=> only one network)
#                print(actions)
#                print()
#==============================================================================
            
# ============================================================================== 
#                           EXPERIENCE REPLAY      
# ==============================================================================
    
    
            if len(D) == 16: # BATCH SIZE by Guillaume Lample
                batch = copy.deepcopy(np.array(D))
                random.shuffle(batch)
                tts = np.empty([0,nb_actions])
                
                for i in range(len(batch)):
                  
                    faces_done = batch[i][-1]
                    Qprime = targetNet.sess.run(targetNet.Q2,feed_dict={x:batch[i][-2]})
                    maxQprime = np.max(Qprime)
                    
                    tt = targetNet.sess.run(targetNet.Q2,feed_dict={x:batch[i][0]})
                    if faces_done > 6:
                        tt[0,batch[i][1]] = batch[i][-3]
                    else:
                        tt[0,batch[i][1]] = batch[i][-3] + gamma*maxQprime
                  
                    tts = np.concatenate((tts,tt),0)
                  
                targetNet.sess.run(train_step,feed_dict={Q_: tts, x: batch[:,0][0]})
                
                D = []
    
# ============================================================================== 
#                           
# ==============================================================================


            if numCompleteFaces(s) == 6:
                done = 1
                break
            
        dones = np.append(dones,done)
        cum_reward.append(cum_reward_fill)
        

            
#        print("updating Q2 for the target network")    
#        print("   before")
#        print(targetNet.sess.run(targetNet.Q2,feed_dict={x: S}))
        targetNet.update_target_network() #cette ligne (qu'il y avait précedemment dans le code) n'actualise en fait pas targetNet
#        targetNet.update_target_network_(trainNet) #je pense que c'est plutôt ça qu'il faut faire
#        print("   after")
#        print(targetNet.sess.run(targetNet.Q2,feed_dict={x: S}))
        

            
        if episode%100 == 1:
#            print(targetNet.sess.run(targetNet.W1))
#             sess.run(loss_function,feed_dict={Q_: targetQ, x: S}),"\t",
            print(n_moves,"\t",episode,"\t",round(np.mean(trainNet.sess.run(trainNet.W1)),2),"\t", round(np.mean(cum_reward[-1000:]),2), "\t", int(np.sum(dones[-1000:])),"\t", round(100*np.mean(dones[-1000:]),2),"\t", round(100*np.mean(dones),2))
            percentDone.append(100*np.mean(dones[-1000:]))
    #             print(lActions)
#            print(np.var(sess.run(Q2,feed_dict={x:S})))
        
            
        if episode%1000 == 1:
            plt.clf()
            plt.plot(percentDone, linewidth = 2)
            plt.title("n_moves: "+str(n_moves))
            plt.pause(0.0001)
            topickle = [targetNet.sess.run(targetNet.W1),targetNet.sess.run(targetNet.W2),targetNet.sess.run(targetNet.b1),targetNet.sess.run(targetNet.b2)]
            pickle.dump(topickle, open('save.p', 'wb'))
#            if lActions_batch.shape != lActions.shape:
#                lActions_batch = np.copy(lActions)
#                print(lActions)
#            else:
##                print("check:",lActions,lActions_batch)
#                print(lActions - lActions_batch)
#                lActions_batch = np.copy(lActions)

            
def longTrain(c_init,n_moves_init, n_moves_max):

    for i in range(n_moves_init,1+n_moves_max):
        print("==============================================================================")
        print("\t",i,"Moves","\t")
        print("==============================================================================")
        DQN(c_init=c_init,Tmax=i,nb_episodes=int(sys.argv[2]),n_moves = i)

# In[ ]:

with tf.device("/gpu:0"):
    longTrain(Cube(3),1,10)
#    longTrain(Cube(3),1,1) #mandatory for detail_mode = True (and we need also to have a fixed move, not a random one)
#    DQN(c_init=Cube(3),Tmax=int(sys.argv[4]),nb_episodes=int(sys.argv[2]),n_moves = int(sys.argv[3]))


# In[ ]:

topickle = [targetNet.sess.run(targetNet.W1),targetNet.sess.run(targetNet.W2),targetNet.sess.run(targetNet.b1),targetNet.sess.run(targetNet.b2)]
pickle.dump(topickle, open('save.p', 'wb'))

# In[ ]:


# In[ ]:

