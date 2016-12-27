
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, 'Resources/MagicCube/code/')
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


# In[29]:

c_init=Cube(3)

resume = sys.argv[1] == "True"

# sess = tf.InteractiveSession()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
with tf.device("/gpu:0"):

    x = tf.placeholder(tf.float32, shape=[None,6*c_init.N**2])
    Q_ = tf.placeholder(tf.float32, shape=[None,nb_actions])
    n_layers = 5

    if not resume:
        middle_layer = 500
        W1 = tf.Variable(tf.random_normal([6*c_init.N**2,middle_layer], stddev=1e-2))
        b1 = tf.Variable(tf.random_normal([middle_layer], stddev=1e-2))

        W2 = tf.Variable(tf.random_normal([middle_layer,nb_actions], stddev=1e-2))
        b2 = tf.Variable(tf.random_normal([nb_actions], stddev=1e-2)) 

        for i in range(n_layers):
            globals()['Wm_%s'%i] = tf.Variable(tf.random_normal([middle_layer,middle_layer], stddev=1e-2))
            globals()['bm_%s'%i]  = tf.Variable(tf.random_normal([middle_layer], stddev=1e-2)) 
        
    else:
        load = pickle.load(open('save.p', 'rb'))
        W1 = tf.Variable(load[0])
        W2 = tf.Variable(load[1])
        b1 = tf.Variable(load[2])
        b2 = tf.Variable(load[3])
        for i in range(n_layers):
            globals()['Wm_%s'%i] = tf.Variable(load[4][i])
            globals()['bm_%s'%i]  = tf.Variable(load[5][i])


    Q1 = tf.nn.tanh(tf.matmul(x/6,W1) + b1)
    
    Qm_0 = tf.nn.tanh(tf.matmul(Q1,Wm_0) + bm_0)
    
    for i in range(1,n_layers):
        
        globals()['Qm_%s'%i] = tf.nn.tanh(tf.matmul(globals()['Qm_%s'%(i-1)],globals()['Wm_%s'%i]) + globals()['bm_%s'%i])
        
        
    Q2 = tf.matmul(globals()['Qm_%s'%(n_layers-1)],W2) + b2#tf.nn.relu(tf.matmul(Qs1,W2))# + b2)
    # Qs = tf.matmul(Q2,act)
    Qs = Q2

    
    loss_function = tf.reduce_mean(tf.square(tf.sub(Q_,Qs)))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)


# In[37]:

def DQN(c_init,Tmax,nb_episodes, n_moves):
    
    done = 0
    lActions = np.zeros(18)
    print("moves","\t","ep.","\t","Loss Function","\t","Min Q","\t\t", "Reward", "", "NB.","\t", "Prcent.")
    
    with tf.device("/gpu:0"):
        train_step = tf.train.RMSPropOptimizer(0.002).minimize(loss_function)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        
    mineps = 5./1000
    epssteps = 1e5
    def eps(episode):
        return 1-(1-mineps)*min(1,episode/(epssteps))
    
    percentDone = []

    episode = 1
    
    tries = 1
    
    dones = np.empty([0])
    
    D = []
    
    while np.sum(dones[-1000:])/min(1000,tries) < .9*((1-mineps)**n_moves) and episode < nb_episodes:  
        
        episode += 1
        
        s = copy.deepcopy(c_init)
        s.randomize(n_moves) #we randomize n_moves times in order to have a "well mixed" cube
        #s.move("R",2,-1)
        cum_reward = []
        
        tries += 1
        done = 0
            
        for i in range(Tmax):
            
            #Choose an action by greedily (with e chance of random action) from the Q-network
            S = copy.deepcopy(np.reshape(s.stickers,(1, 54)))
            Qout = sess.run(Q2,feed_dict={x:S})
            if(rd.random() > eps(episode)):
                a = np.argmax(Qout)
            else:
                a = rd.randint(0,nb_actions-1)

            lActions[a] += 1
            
            #print(actions[a])
            #print(Qout)
            #Get new state and reward from environment
            f,l,d = actions[a]
            #print(actions[a])
            s.move(f,l,d)            
            r = reward_cube(s)
            cum_reward.append(r)
            D.append(copy.deepcopy([S, a, r, np.reshape(s.stickers,(1, 54)) , numCompleteFaces(s)]))
            
            #print(S)
            #print(np.reshape(s.stickers,(1, 54)))
            
            #Obtain the Q' values by feeding the new state through our network
#==============================================================================
#            Qprime = sess.run(Q2,feed_dict={x:np.reshape(s.stickers,(1, 54))})
             #Obtain maxQ' and set our target value for chosen action.
#            maxQprime = np.max(Qprime)
#            targetQ = Qout
#            targetQ[0,a] = r + gamma*maxQprime
             #Train our network using target and predicted Q values
#                
#             sess.run(train_step,feed_dict={Q_: targetQ, x: S})
#==============================================================================
            
            #print(targetQ)

            
# ============================================================================== 
#                           EXPERIENCE REPLAY      
# ==============================================================================
    
    
            if len(D) == 16: # BATCH SIZE by Guillaume Lample
                batch = copy.deepcopy(np.array(D))
                random.shuffle(batch)
                tts = np.empty([0,nb_actions])
                
                for i in range(len(batch)):
                  
                    faces_done = batch[i][-1]
                    Qprime = sess.run(Q2,feed_dict={x:batch[i][-2]})
                    maxQprime = np.max(Qprime)
                    
                    tt = sess.run(Q2,feed_dict={x:batch[i][0]})
                    if faces_done >= 6:
                        tt[0,batch[i][1]] = batch[i][-3]
                    else:
                        tt[0,batch[i][1]] = batch[i][-3] + gamma*maxQprime
                  
                    tts = np.concatenate((tts,tt),0)
                  
                sess.run(train_step,feed_dict={Q_: tts, x: np.vstack(batch[:,0])})
                
                D = []
    
# ============================================================================== 
#                           
# ==============================================================================

            if numCompleteFaces(s) == 6:
                done = 1
                break
            
        dones = np.append(dones,done)
            
        
            
        if episode%100 == 1:
#             sess.run(loss_function,feed_dict={Q_: targetQ, x: S}),"\t",
            print(n_moves,"\t",episode,"\t",min(sess.run(Q2,feed_dict={x:S})[0]),"\t", round(np.mean(cum_reward[-1]),2), "\t", np.sum(dones[-1000:]),"\t", round(100*np.sum(dones[-1000:])/min(1000,tries),2))
    #             print(lActions)
            percentDone.append(100*np.sum(dones[-1000:])/min(1000,tries))
            
        if episode%1000 == 1:
            plt.clf()
            plt.plot(percentDone, linewidth = 2)
            plt.title("n_moves: "+str(n_moves))
            plt.pause(0.0001)
            
        if episode%10000 == 1:
            topickle = [sess.run(W1),sess.run(W2),sess.run(b1),sess.run(b2)]
            topickle.append([sess.run(globals()['Wm_%s'%i]) for i in range(n_layers)])
            topickle.append([sess.run(globals()['bm_%s'%i]) for i in range(n_layers)])
            pickle.dump(topickle, open('save.p', 'wb'))




            
def longTrain(c_init,n_moves_max):

    for i in range(1,1+n_moves_max):
        print("==============================================================================")
        print("\t",i,"Moves","\t")
        print("==============================================================================")
        DQN(c_init=c_init,Tmax=i,nb_episodes=int(sys.argv[2]),n_moves = i)

# In[ ]:

with tf.device("/gpu:0"):
    longTrain(Cube(3),10)
#    DQN(c_init=Cube(3),Tmax=int(sys.argv[4]),nb_episodes=int(sys.argv[2]),n_moves = int(sys.argv[3]))


# In[ ]:
    
topickle = [sess.run(W1),sess.run(W2),sess.run(b1),sess.run(b2)]
topickle.append([sess.run(globals()['Wm_%s'%i]) for i in range(n_layers)])
topickle.append([sess.run(globals()['bm_%s'%i]) for i in range(n_layers)])
pickle.dump(topickle, open('save.p', 'wb'))


# In[ ]:




# In[ ]:
