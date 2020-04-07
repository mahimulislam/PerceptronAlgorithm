import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(1234)
randomweight=np.zeros([1,6])
for ran in range(0,6):
    x=random.randint(1, 10)
    x=x/10
    randomweight[:,ran]=x
print(randomweight)
df=pd.read_csv('train.txt',sep=" ",header=None,dtype='Float64')
df_arr=df.values

xtrain=df_arr[:,0]
ytrain=df_arr[:,1]
classtrain=df_arr[:,2]

y6d=[]
flagc1=0
flagc2=0

for i in range(0,len(xtrain)):
    if classtrain[i] == 1:
        if flagc1==0:
            plt.scatter(xtrain[i],ytrain[i],s=20,c='r',marker='o',label='class1 train')
            flagc1=1
        else:
            plt.scatter(xtrain[i], ytrain[i], s=20, c='r', marker='o')
        y6d.append([xtrain[i] ** 2, ytrain[i] ** 2, xtrain[i] * ytrain[i], xtrain[i], ytrain[i], 1])
    if classtrain[i] == 2:
        if flagc2==0:
            plt.scatter(xtrain[i], ytrain[i], s=20, c='g', marker='x', label='class2 train')
            flagc2=1
        else:
            plt.scatter(xtrain[i], ytrain[i], s=20, c='g', marker='x')
        y6d.append(
            [-1 * xtrain[i] ** 2, -1 * ytrain[i] ** 2, -1 * xtrain[i] * ytrain[i], -1 * xtrain[i], -1 * ytrain[i],
             -1 * 1])

plt.legend(loc='best')
plt.show()

weight1 = np.ones([1, 6])
weight2 = np.zeros([1, 6])
weight3 = randomweight

weightfn=[weight1,weight2,weight3]
for k in range(0,3):
    alphas=np.arange(0.1,1.1,.1)
    singleit=[]
    for alpha in alphas:
        weight = weightfn[k]
        counter=200
        itnum=0
        for count in range(0,counter):
            checkmark=0
            for i in range(0,len(y6d)):
                comparison=np.dot(weight,y6d[i])
                if comparison[0]<=0:
                        weight=np.add(weight,np.dot(alpha,y6d[i]))
                elif comparison[0]>0:
                        checkmark=checkmark+1
            if checkmark==6:
                itnum=count+1
                break
        singleit.append(itnum)


    batchalphas=np.arange(0.1,1.1,.1)
    batchit=[]
    for batchalpha in batchalphas:
        weightbatch = weightfn[k]
        counterbatch=200
        itnumbatch=0
        plus = np.zeros([1, 6])
        for countbatch in range(0,counterbatch):
            checkmarkbatch=0
            for i in range(0,len(y6d)):
                comparison=np.dot(weightbatch,y6d[i])
                if comparison[0]<=0:
                    plus=np.add(plus,y6d[i])
                elif comparison[0]>0:
                    checkmarkbatch=checkmarkbatch+1
            if checkmarkbatch==6:
                itnumbatch=countbatch+1
                break
            else:
                weightbatch = np.add(weightbatch, np.dot(batchalpha, plus))
        batchit.append(itnumbatch)

    print("Value of Alpha\t One at a Time\t Many at a Time")
    for i in range(0,len(batchalphas)):
        print("%.1f" % batchalphas[i],'\t\t\t\t',singleit[i],'           ',batchit[i])


    N = 10

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, singleit, width, label='One at a time')
    plt.bar(ind + width, batchit, width,
        label='Many at a time')

    plt.ylabel('Number Of Iterations')
    plt.xlabel('Learning rate')

    plt.title('Comparison between batch and single perceptron algorithm')

    plt.xticks(ind + width / 2, ('0.1', '0.2', '0.3', '0.4', '0.5','0.6', '0.7', '0.8', '0.9', '1.0'))
    plt.legend(loc='best',prop={'size': 7})
    plt.show()