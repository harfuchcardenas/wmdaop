def main():
    # libraries
    import csv
    import pandas as pd
    import statistics
    import numpy as np
    import math
    from matplotlib.ticker import FuncFormatter
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv("agustin_thesis.csv",low_memory=False)
    df=df.sort_values(['time'], inplace=False, ascending=True)
#     print("df:",df)
#     print("len(df):",len(df))
#     print("df.shape:",df.columns)
    ADD=[]
    TYPE=[]
    TYPEN=[]
    x=0
    lim = len(df)
    while x < lim:
        ADD.append(df.value[x][4:20])   #Checked[4:20]
        TYPE.append(df.value[x][18:20])
        x+=1

    df['Address']=ADD #Added column of Address to the sorted list
    df['Type']=TYPE #Added column of TYPE pf meter to sorted list
    singles=pd.unique(ADD) #OK
    diff_list = []
    xx=0
    df_mean=pd.DataFrame()
    df_stadv=pd.DataFrame()
    flag=False
    df_wake = pd.DataFrame()
    df_type = pd.DataFrame()
    df_diferencia = pd.DataFrame(columns=['Address','Diferencia'])
    f=[]
    d=pd.DataFrame()
    sfilter=pd.DataFrame()
    filtradof=pd.DataFrame()
    df_scatter=[]
    te=[]

    while xx<len(singles):
        tempx=singles[xx]
        filtrado=df.query('Address==@tempx',inplace=False)
        filtrado=filtrado.sort_values(['time'], inplace=False, ascending=True)
        filtrado['time']=filtrado['time'].div(1e+12) #[seg]

        if len(filtrado)==1:
            flag=True
            average=0
        else:
            flag=False
            diferencia=filtrado.time.diff() #Result is a series
            smallest=statistics.mode(diferencia)
            diferencia=diferencia.to_frame()
            filtrado['diferencia']=diferencia
            perc=.15
            temp0=smallest*(1-perc) #lowest range to filter the data
            temp1=smallest*(1+perc) #Highest range to filter the data
            diferencia = diferencia.apply (pd.to_numeric, errors='coerce')
            diferencia = diferencia.dropna()
            diferencia=diferencia.query('time>=@temp0 and time<=@temp1',inplace=False)
            diff= diferencia.values.tolist()
            average=diferencia.mean()
            stadvt=diferencia.std(axis=0)
            stadv=math.sqrt(stadvt)

        if flag == False:
                df_mean.at[1,xx]=singles[xx]    #only the addresses of the devices
                df_mean.at[2,xx]=average[0]        #mean wakeup time of every device
                df_stadv.at[1,xx]=singles[xx]
                df_stadv.at[2,xx]=stadv
                df_wake.at[1,xx]=singles[xx]
                df_wake.at[2,xx]=filtrado['time'].iloc[0]*1000   #offset of n submeter
                df_type.at[1,xx]=singles[xx]
                df_type.at[2,xx]=filtrado['Type'].iloc[0]   #offset of n submeter
                df_type.at[3,xx]=average[0]        #mean wakeup time of every device
                df_scatter.append(diff)

        xx+=1

    df_mean=df_mean.dropna(axis=1)
    df_stadv=df_stadv.dropna(axis=1)
    del df_type[34]
    tiempos=df_mean[1:]   #Cool! Keep going!
    stdev=df_stadv[1:]
    x=np.arange(len(df_scatter[0])) + 1
    flatList = [ item for elem in df_scatter[0] for item in elem]  #To take the values inside the inner lists to ground level of the main list
    a = x +1
    stand0=pd.DataFrame({'x_axis': a, 'y_axis': flatList })

    plt.ylabel('Filtered time interval (min)',fontsize=17)
    plt.xlabel('Consecutive ocurrences',fontsize=17)
    plt.title('Time interval distribution for meter No.0 between consecutive reception of messages from the meters',fontsize=22)
    plt.grid(True)
    plt.plot( 'x_axis', 'y_axis', data=stand0, linestyle='-', marker='o')

    i=0
    line1 = []
    line2 = []
    while i<len(x):
        line1.append(.0185)
        line2.append(0.0145)
        i+=1
    x1 = np.arange(len(x)) + 1
    x2 = x1
    error1=pd.DataFrame({'x': x1, 'y': line1})
    error2=pd.DataFrame({'x': x1, 'y': line2})
    plt.plot( 'x', 'y', data=error1, linestyle='-', color='r')
    plt.plot( 'x', 'y', data=error2, linestyle='-', color='r')
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['legend.framealpha'] = None
    plt.rcParams['legend.edgecolor'] = 'inherit'
    plt.show()

    temporal=tiempos.values.tolist()
    temporal2=stdev.values.tolist()
    t = []  #mean times list
    s = []  #Standard Deviation list
    i=0

    # print("type(temporal):",type(temporal))
    while i<len(temporal[0]):
        t.append(temporal[0][i])
        s.append(temporal2[0][i])
        i+=1
    # print("type(t):",type(t))
    x = np.arange(len(df_mean.columns)) #OK
    x = x+1
    fig = plt.figure()
    ax = fig.add_axes([0.07,0.07,.9,.9])
    ax.bar(x,t,
        yerr=s,
        align='center',
        alpha=0.5,
        ecolor='black',
        capsize=15,
        color=(0.2, 0.4, 0.6, 0.6))
    plt.ylabel('Mean duty cycle (min)', fontsize=17)
    plt.xlabel('Meters', fontsize=17)
    plt.title('Mean interval period with standard deviation for every meter', fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    x = np.arange(len(s)) + 1
    stand=pd.DataFrame({'x_axis': x, 'y_axis': s })

    plt.ylabel('Variance: \u03C3^2',fontsize=17)
    plt.xlabel('Devices',fontsize=17)
    plt.title('Variance of all the detected meters',fontsize=22)
    plt.grid(True)
#     plt.tight_layout()
    plt.plot( 'x_axis', 'y_axis', data=stand, linestyle='-', marker='o')

    i=0
    line1 = []
    line2 = []
    while i<=len(s):
        line1.append(.141)
        line2.append(0.02)
        i+=1
    x1 = range(len(s)+1)
    # print("x1:",len(x1))
    # print("line1:",len(line1))
    error1=pd.DataFrame({'x': x1, 'y': line1})
    error2=pd.DataFrame({'x': x1, 'y': line2})
    plt.plot( 'x', 'y', data=error1, linestyle='-', color='r')
    plt.plot( 'x', 'y', data=error2, linestyle='-', color='r')
    plt.show()

    a=df_type[0].iloc[2]
    b=df_type[15].iloc[2]
    c=df_type[11].iloc[2]
    d=df_type[3].iloc[2]

    barWidth = 0.5
    fig, ax = plt.subplots(figsize =(12, 8))

    cuatro =    [df_type[0].iloc[2] * 60, df_type[1].iloc[2] * 60, df_type[2].iloc[2] * 60, df_type[4].iloc[2] * 60, df_type[5].iloc[2] * 60, df_type[6].iloc[2] * 60, df_type[7].iloc[2] * 60, df_type[8].iloc[2] * 60, df_type[14].iloc[2] * 60, df_type[17].iloc[2] * 60, df_type[22].iloc[2] * 60, df_type[24].iloc[2] * 60, df_type[25].iloc[2] * 60, df_type[28].iloc[2] * 60]
    seis =      [df_type[15].iloc[2] * 60, df_type[10].iloc[2] * 60, df_type[12].iloc[2] * 60, df_type[15].iloc[2] * 60, df_type[16].iloc[2] * 60, df_type[23].iloc[2] * 60, df_type[26].iloc[2] * 60, df_type[33].iloc[2] * 60]
    siete =     [df_type[11].iloc[2] * 60, df_type[18].iloc[2] * 60, df_type[19].iloc[2] * 60, df_type[20].iloc[2] * 60, df_type[27].iloc[2] * 60, df_type[29].iloc[2] * 60, df_type[31].iloc[2] * 60]
    tys =       [df_type[3].iloc[2] * 60, df_type[9].iloc[2] * 60]

    cuatro = np.round(cuatro,2)
    seis = np.round(seis,2)
    siete = np.round(siete,2)
    tys = np.round(tys,2)

    br1 = np.arange(len(cuatro))
    br1 = br1 + 1

    br2 = np.arange(len(cuatro),len(cuatro)+len(seis))
    br3 = np.arange(len(cuatro)+len(seis),len(cuatro)+len(seis)+len(siete))
    br4 = np.arange(len(cuatro)+len(seis)+len(siete),len(cuatro)+len(seis)+len(siete)+len(tys))
    width = 0.35

    plt.ylabel('Mean duty cycle (s)',fontsize=17)
    plt.xlabel('Meters',fontsize=17)
    plt.title('Mean interval period',fontsize=22)
    plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['legend.framealpha'] = None
    plt.rcParams['legend.edgecolor'] = 'inherit'
    plt.grid(True, which='both')
    plt.tight_layout()

    # pps = ax.bar(br1 - width/2, cuatro, width, label='population')

    pps = plt.bar(br1-width/2,cuatro, color = 'r', label='Heat meters')
    for p in pps:
       height = p.get_height()
       ax.annotate('{}'.format(height),
          xy=(p.get_x() + p.get_width() / 2, height),
          xytext=(0, 3), # 3 points vertical offset
          textcoords="offset points",
          ha='center', va='bottom', fontsize=10)

    pps2=plt.bar(br2-width/2,seis, color = 'b', label='Warm water meter')
    for p in pps2:
       height = p.get_height()
       ax.annotate('{}'.format(height),
          xy=(p.get_x() + p.get_width() / 2, height),
          xytext=(0, 3), # 3 points vertical offset
          textcoords="offset points",
          ha='center', va='bottom', fontsize=10)

    pps3=plt.bar(br3,siete,color='y', label='Water meter')
    for p in pps3:
       height = p.get_height()
       ax.annotate('{}'.format(height),
          xy=(p.get_x() + p.get_width() / 2, height),
          xytext=(0, 3), # 3 points vertical offset
          textcoords="offset points",
          ha='center', va='bottom', fontsize=10)

    pps4=plt.bar(br4,tys,color='g',label='Radio converter')
    for p in pps4:
       height = p.get_height()
       ax.annotate('{}'.format(height),
          xy=(p.get_x() + p.get_width() / 2, height),
          xytext=(0, 3), # 3 points vertical offset
          textcoords="offset points",
          ha='center', va='bottom', fontsize=10)

    plt.legend(labels=['Heat meters', 'Warm water meter','Water meter','Radio converter'], fontsize=14)
    plt.show()

    df=df.sort_values(['time'], inplace=False, ascending=True)
    dfini=df.head(1)
    dflas=df.tail(1)
    # print("dfini:",dfini)
    # print("dflas:",dflas)
    a=dfini['time'].iloc[0]/1e9

    train = pd.DataFrame(columns=['Address','TimPo'])

    a=[]
    dp=[]
    #start date: 1611201603
    #end date:   1612018798
    #          b=1612065603 #One day POSIX from EPOCH
#     b=1611417603*100
    # b=1611208803*100 # two hours
    # b=1611205203*100 # one hour
    b=1611202503*100 #15min
    # b=161130667600 #
    i=1
    j=0
    while i <len(df_mean.columns):

        if i==13:
            print("trece")
            print(i)
            i+=1
            print("len(a)",len(a))
        elif i==21:
            print("veintiuno")
            print(i)
            i+=1
            print("len(a)",len(a))
        elif i==30:
            print("treinta")
            print(i)
            i+=1
            print("len(a)",len(a))
        elif i==32:
            print("treintaidos")
            print(i)
            i+=1
            print("len(a)",len(a))
        else:
            print("else")
            print(i)
            while (df_wake[i][2]+df_mean[i][2]*j)*100 <b:
                res1=df_mean[i][1]
                a.append(res1)
                res2=(df_wake[i][2]+df_mean[i][2]*j)*100
                dp.append(res2)
                j+=1
            print("len(a)",len(a))
            j=1
            i+=1
        print("salio")

    train['Address']=a
    train['TimPo']=dp
    pd.options.display.float_format = '{:.5f}'.format
    s_train=train.sort_values(['TimPo'], inplace=False, ascending=True)#>>>>>IMPORTANT: This value remains constant through all the meters<<<<<<<<<<<

    train.to_csv("train.csv")
    s_train.to_csv("train_new1.csv")

    add=df_mean[1][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print(s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514677935930004']=1

    add=df_mean[2][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'A511901576654004']=1

    add=df_mean[3][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'AC48941300005037']=1

    add=df_mean[4][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514611135840004']=1

    add=df_mean[5][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514707935930004']=1


    add=df_mean[6][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514551135840004']=1


    add=df_mean[7][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514531135840004']=1


    add=df_mean[8][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514667935930004']=1

    add=df_mean[9][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'AC48981300005037']=1


    add=df_mean[10][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514233010930306']=1


    add=df_mean[11][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514530450930307']=1


    add=df_mean[12][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514213010930306']=1


    add=df_mean[14][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514571135840004']=1


    add=df_mean[15][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514450170940306']=1


    add=df_mean[16][1]
    s_train1=train.query('Address==@add')
    for elem in s_train1.index:
        train.at[elem,'C514460170940306']=1

    pd.options.display.float_format = '{:.5f}'.format
    s_train=train.sort_values(['TimPo'], inplace=False, ascending=True)#>>>>>IMPORTANT: This value remains constant through all the meters<<<<<<<<<<<

    s_train=s_train.fillna(0)

    x=np.linspace(161120160300+129533,161120160300+130533,1001)

#     train.to_csv("train_inside.csv")
    # print("\ntrain[['C514910170940307']]:",train[['C514910170940307']])
    y1=s_train['C514677935930004']
    y2=s_train['AC48941300005037']
    y3=s_train['C514611135840004']
    # y4=s_train['C514707935930004']
    # y5=s_train['C514551135840004']
    # y6=s_train['C514531135840004']
    # y7=s_train['C514667935930004']
    y8=s_train['AC48981300005037']
    print("y8",y8[y8.iloc[:]==1])
    y9=s_train['C514233010930306']
    print("y9",y9[y9.iloc[:]==1])
    y10=s_train['C514530450930307']
    print("y10",y10[y10.iloc[:]==1])
    y11=s_train['C514213010930306']
    print("y11",y11[y11.iloc[:]==1])
    # y12=train['C514571135840004']
    # y13=train['C514450170940306']
    # y14=train['C514460170940306']
    # y16=train[['C514440170940306']]
#     y17=train[['C514581135840004']]
#     y18=train[['C514501135840004']]
#     y19=train[['C514410170940306']]
    # y20=train[['C514590450930307']]
#     y21=train[['C514717935930004']]
    # y22=train[['C514910170940307']]
#     y23=train[['C514970170940307']]
#     y24=train[['C514203010930306']]
#     print("y4:",y5[y5['C514233010930306']==1])
#     plt.stem(x[0:30], y2[0:30], '-.')
#     plt.stem(x[0:300], y3[0:300], '-.')
#     plt.stem(x[0:300], y4[0:300], '-.')
#     # plt.stem(x[0:300], y5[0:300], '-.')
#     # plt.stem(x[0:300], y6[0:300], '-.')
#     # plt.stem(x[0:300], y7[0:300], '-.')
#     # plt.stem(x[0:300], y8[0:300], '-.')
#     plt.stem(x[0:300], y9[0:300], '-.')
#     plt.stem(x[0:300], y10[0:300], '-.')
#     plt.stem(x[0:300], y11[0:300], '-.')
#     # plt.stem(x[0:300], y12[0:300], '-.')
#     # plt.stem(x[0:300], y13[0:300], '-.')
#     plt.stem(x[0:300], y14[0:300], '-.')
#     plt.stem(x[0:300], y15[0:300], '-.')
#     plt.stem(x[0:300], y16[0:300], '-.')
#     plt.stem(x[0:300], y17[0:300], '-.')
#     plt.stem(x[0:300], y18[0:300], '-.')
#     # plt.stem(x[0:300], y19[0:300], '-.')
#     # plt.stem(x[0:300], y20[0:300], '-.')
#     plt.stem(x[0:300], y21[0:300], '-.')
#     plt.stem(x[0:300], y22[0:300], '-.')
    # plt.stem(x[0:300], y23[0:300], '-.')
    # plt.stem(x[0:300], y24[0:300], '-.')

    color_list = ['C0o','C1o','C2o','C3o','C4o','C5o','C6o','C7o','C8o','C9o','C0x','C1x','C2x']

    print("len(y1):",len(y1[129533:130534]))
    print("len(x)",len(x))
    plt.stem(x, y1[129533:130534], linefmt='C0-.',markerfmt='C0o')
    plt.stem(x, y3[129533:130534], linefmt='C2-.',markerfmt='C2o')
    plt.stem(x, y2[129533:130534], linefmt='C1-.',markerfmt='C1o')
    # plt.stem(x[129533:130533], y8[129533:130533], linefmt='C3-.',markerfmt='C3o')
    plt.stem(x, y9[129533:130534], linefmt='C4-.',markerfmt='C4o')
    plt.stem(x, y11[129533:130534], linefmt='C6-.',markerfmt='C6o')
    plt.stem(x, y10[129533:130534], linefmt='C5-.',markerfmt='C5o')
    # plt.stem(x[0:80000], y20[0:8000], linefmt='C7-.',markerfmt='C7o')
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 'xx-small'
    plt.rcParams['legend.framealpha'] = None
    plt.rcParams['legend.edgecolor'] = 'inherit'
    plt.title('Sending intervals of the meters',fontsize=22)
    plt.xlabel('Time [POSIX] s',fontsize=17)
    plt.ylabel('Logic', fontsize=17)
    plt.legend(['Heat meter 1', 'Radio converter 1','Heat meter 2', 'Radio converter 2','Warm water meter 1','Water meter 1','Warm water meter 2'], fontsize=14)
    plt.grid(True)
    plt.show()



    # # plot the stem plot using matplotlib
    #
    # markerline, stemlines, baseline = plot.stem(stems, marks, '-.')


    train.set_index('TimPo')
    tiem1=train.iloc[:,2]
    # print("train[:1]:",train[1:])   #OK parece que los "1s" estan en donde deberian estar
    # tiem1:",tiem1.iloc[:60,3]
    # print("tiem1",tiem1[5:50])
    temporal1=train['C514677935930004'].values.tolist()
    # print("temporal1:",temporal1)
    # temporal1=temporal1[0:32]
    print("len(temporal1):",len(temporal1))
    x1=len(temporal1)



if __name__ == "__main__":
    main()

import IPython.nbformat.current as nbf
nbf.write(nb, open('test.ipynb', 'w'), 'ipynb')
