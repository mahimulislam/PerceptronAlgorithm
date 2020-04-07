import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('iris.data',sep=",",header=None)
df_arr=df.values

xtrain=df_arr[0:100,0]
ytrain=df_arr[0:100,2]
classtrain=df_arr[0:100,4]

y6d=[]
flagc1=0
flagc2=0

for i in range(0,len(xtrain)):
    if classtrain[i] == 'Iris-setosa':
        if flagc1==0:
            plt.scatter(xtrain[i],ytrain[i],s=20,c='r',marker='o',label='class1 train')
            flagc1=1
        else:
            plt.scatter(xtrain[i], ytrain[i], s=20, c='r', marker='o')
        y6d.append([xtrain[i] ** 2, ytrain[i] ** 2, xtrain[i] * ytrain[i], xtrain[i], ytrain[i], 1])
    if classtrain[i] == 'Iris-versicolor':
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
print(y6d)
weight1 = np.ones([1, 6])
weight2 = np.zeros([1, 6])


weightfn=[weight1,weight2]
for k in range(0,1):
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
            if checkmark==100:
                itnum=count+1
                break
        singleit.append(itnum)

    print("Value of Alpha\t One at a Time\t Many at a Time")
    for i in range(0, len(alphas)):
        print("%.1f" % alphas[i], '\t\t\t\t', singleit[i])


    N = 10

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, singleit, width, label='One at a time')
    plt.ylabel('Number Of Iterations')
    plt.xlabel('Learning rate')

    plt.title('Single perceptron algorithm')

    plt.xticks(ind + width / 2, ('0.1', '0.2', '0.3', '0.4', '0.5','0.6', '0.7', '0.8', '0.9', '1.0'))
    plt.legend(loc='best',prop={'size': 7})
    plt.show()