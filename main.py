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
    plt.tight_layout()
    plt.plot( 'x_axis', 'y_axis', data=stand, linestyle='-', marker='o')

    i=0
    line1 = []
    line2 = []
    while i<=len(s):
        line1.append(.141)
        line2.append(0.02)
        i+=1
    x1 = range(len(s)+1)
    print("x1:",len(x1))
    print("line1:",len(line1))
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

    cuatro =    [df_type[0].iloc[2] * 60, df_type[1].iloc[2] * 60, df_type[2].iloc[2] * 60, df_type[4].iloc[2] * 60, df_type[5].iloc[2] * 60, df_type[6].iloc[2] * 60, df_type[7].iloc[2] * 60, df_type[8].iloc[2] * 60, df_type[14].iloc[2] * 60, df_type[17].iloc[2] * 60, df_type[22].iloc[2] * 60, df_type[24].iloc[2] * 60, df_type[25].iloc[2] * 60, df_type[28].iloc[2] * 60, df_type[34].iloc[2] * 60]
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
    plt.grid(True)
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
    print("dfini:",dfini)
    print("dflas:",dflas)
    a=dfini['time'].iloc[0]/1e9
    # b=dflas['time'].iloc[0]/1e9
    print("a:",a)
    print("b:",b)
    # b=1611288003 #One day POSIX from EPOCH
    b=1611205203 #One hour POSIX from EPOCH
    tdiff=(b-a)
    tdiff=int(tdiff)


    # print("type(tdiff):",type(tdiff))
    # print("len(df_mean.columns):",len(df_mean.columns))
    train = pd.DataFrame(columns=['Address','TimPo'])
    # print("train.columns:",train.columns)
    a=[]
    dp=[]
    print("df_mean.columns:",df_mean.columns)
    print("df_wake.columns:",df_wake.columns)
    print("df_mean[1][:]:", df_mean[0:])

    for elem in df_mean.columns:
        print("df_mean[add][1]:",df_mean[elem][1])
        print("df_mean[add][2]:",df_mean[elem][2])


    for elem in df_wake.columns:
        # print("df_wake[add][1]:",df_wake[elem][1])
        # print("df_wake[add][2]:",df_wake[elem][2])
        a.append(df_wake[elem][1])
        dp.append(df_wake[elem][2])
#Last column (Address) from df_wake is not included in df_mean
    i=1
    j=1
    while i <len(df_mean.columns):
        if i==13 or i==21 or i==30 or i==32:
            i+=1
        while df_wake[i][2]+df_mean[i][2]*j <=b:
            a.append(df_mean[i][1])
            dp.append(df_wake[i][2]+df_mean[i][2]*j)
            # print("a:",a[j])
            # print("dp:",dp[j])
            j+=1
        i+=1

    print("i:",i)
    train['Address']=a
    train['TimPo']=dp
    print("train['TimPo'].head()",train['TimPo'].head())
    print("mean:",df_mean)
    s_train=train.sort_values(['TimPo'], inplace=False, ascending=True)

    add=df_mean[0][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'A511881576654004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'A511881576654004']=1
    print("train.head():",train.head())
    print("train['A511881576654004']:",train['A511881576654004'])
    train['A511881576654004'] = train['A511881576654004'].fillna(0)

    add=df_mean[1][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514677935930004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514677935930004']=1
    print("train.head():",train.head())
    print("train['C514677935930004']:",train['C514677935930004'])
    train['C514677935930004'] = train['C514677935930004'].fillna(0)


    add=df_mean[2][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'A511901576654004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'A511901576654004']=1
    print("train.head():",train.head())
    print("train['A511901576654004']:",train['A511901576654004'])
    train['A511901576654004'] = train['A511901576654004'].fillna(0)


    add=df_mean[3][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'AC48941300005037']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'AC48941300005037']=1
    print("train.head():",train.head())
    print("train['AC48941300005037']:",train['AC48941300005037'])
    train['AC48941300005037'] = train['AC48941300005037'].fillna(0)

    add=df_mean[4][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514611135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514611135840004']=1
    print("train.head():",train.head())
    print("train['C514611135840004']:",train['C514611135840004'])
    train['C514611135840004'] = train['C514611135840004'].fillna(0)

    add=df_mean[5][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514707935930004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514707935930004']=1
    print("train.head():",train.head())
    print("train['C514707935930004']:",train['C514707935930004'])
    train['C514707935930004'] = train['C514707935930004'].fillna(0)


    add=df_mean[6][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514551135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514551135840004']=1
    print("train.head():",train.head())
    print("train['C514551135840004']:",train['C514551135840004'])
    train['C514551135840004'] = train['C514551135840004'].fillna(0)


    add=df_mean[7][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514531135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514531135840004']=1
    print("train.head():",train.head())
    print("train['C514531135840004']:",train['C514531135840004'])
    train['C514531135840004'] = train['C514531135840004'].fillna(0)


    add=df_mean[8][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514667935930004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514667935930004']=1
    print("train.head():",train.head())
    print("train['C514667935930004']:",train['C514667935930004'])
    train['C514667935930004'] = train['C514667935930004'].fillna(0)


    add=df_mean[9][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'AC48981300005037']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'AC48981300005037']=1
    print("train.head():",train.head())
    print("train['AC48981300005037']:",train['AC48981300005037'])
    train['AC48981300005037'] = train['AC48981300005037'].fillna(0)


    add=df_mean[10][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514233010930306']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514233010930306']=1
    print("train.head():",train.head())
    print("train['C514233010930306']:",train['C514233010930306'])
    train['C514233010930306'] = train['C514233010930306'].fillna(0)


    add=df_mean[11][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514530450930307']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514530450930307']=1
    print("train.head():",train.head())
    print("train['C514530450930307']:",train['C514530450930307'])
    train['C514530450930307'] = train['C514530450930307'].fillna(0)


    add=df_mean[12][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514213010930306']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514213010930306']=1
    print("train.head():",train.head())
    print("train['C514213010930306']:",train['C514213010930306'])
    train['C514213010930306'] = train['C514213010930306'].fillna(0)

    add=df_mean[14][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514571135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514571135840004']=1
    print("train.head():",train.head())
    print("train['C514571135840004']:",train['C514571135840004'])
    train['C514571135840004'] = train['C514571135840004'].fillna(0)


    add=df_mean[15][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514450170940306']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514450170940306']=1
    print("train.head():",train.head())
    print("train['C514450170940306']:",train['C514450170940306'])
    train['C514450170940306'] = train['C514450170940306'].fillna(0)

    add=df_mean[16][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514460170940306']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514460170940306']=1
    print("train.head():",train.head())
    print("train['C514460170940306']:",train['C514460170940306'])
    train['C514460170940306'] = train['C514460170940306'].fillna(0)


    add=df_mean[17][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514521135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514521135840004']=1
    print("train.head():",train.head())
    print("train['C514521135840004']:",train['C514521135840004'])
    train['C514521135840004'] = train['C514521135840004'].fillna(0)


    add=df_mean[18][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514950170940307']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514950170940307']=1
    print("train.head():",train.head())
    print("train['C514950170940307']:",train['C514950170940307'])
    train['C514950170940307'] = train['C514950170940307'].fillna(0)


    add=df_mean[19][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514580450930307']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514580450930307']=1
    print("train.head():",train.head())
    print("train['C514580450930307']:",train['C514580450930307'])
    train['C514580450930307'] = train['C514580450930307'].fillna(0)


    add=df_mean[20][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514930170940307']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514930170940307']=1
    print("train.head():",train.head())
    print("train['C514930170940307']:",train['C514930170940307'])
    train['C514930170940307'] = train['C514930170940307'].fillna(0)

    add=df_mean[22][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514561135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514561135840004']=1
    print("train.head():",train.head())
    print("train['C514561135840004']:",train['C514561135840004'])
    train['C514561135840004'] = train['C514561135840004'].fillna(0)


    add=df_mean[23][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514440170940306']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514440170940306']=1
    print("train.head():",train.head())
    print("train['C514440170940306']:",train['C514440170940306'])
    train['C514440170940306'] = train['C514440170940306'].fillna(0)


    add=df_mean[24][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514581135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514581135840004']=1
    print("train.head():",train.head())
    print("train['C514581135840004']:",train['C514581135840004'])
    train['C514581135840004'] = train['C514581135840004'].fillna(0)

    add=df_mean[25][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514501135840004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514501135840004']=1
    print("train.head():",train.head())
    print("train['C514501135840004']:",train['C514571135840004'])
    train['C514501135840004'] = train['C514501135840004'].fillna(0)

    add=df_mean[26][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514410170940306']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514410170940306']=1
    print("train.head():",train.head())
    print("train['C514410170940306']:",train['C514410170940306'])
    train['C514410170940306'] = train['C514410170940306'].fillna(0)


    add=df_mean[27][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514590450930307']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514590450930307']=1
    print("train.head():",train.head())
    print("train['C514590450930307']:",train['C514590450930307'])
    train['C514590450930307'] = train['C514590450930307'].fillna(0)


    add=df_mean[28][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514717935930004']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514717935930004']=1
    print("train.head():",train.head())
    print("train['C514717935930004']:",train['C514717935930004'])
    train['C514717935930004'] = train['C514717935930004'].fillna(0)

    add=df_mean[29][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514910170940307']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514910170940307']=1
    print("train.head():",train.head())
    print("train['C514910170940307']:",train['C514910170940307'])
    train['C514910170940307'] = train['C514910170940307'].fillna(0)

    add=df_mean[31][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514970170940307']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514970170940307']=1
    print("train.head():",train.head())
    print("train['C514970170940307']:",train['C514970170940307'])
    train['C514970170940307'] = train['C514970170940307'].fillna(0)

    add=df_mean[33][1]
    print("add:",add)
    s_train1=s_train.query('Address==@add')
    print("s_train1.head()",s_train1.head())
    # s_train1['@add']=1 #OK
    s_train1.loc[:,'C514203010930306']=1
    print("s_train1:",s_train1)
    print("s_train.index",s_train1.index)
    for elem in s_train1.index:
        train.at[elem,'C514203010930306']=1
    print("train.head():",train.head())
    print("train['C514203010930306']:",train['C514203010930306'])
    train['C514203010930306'] = train['C514203010930306'].fillna(0)

    # df = pd.DataFrame(np.random.randn(1000, 4),index = ts.index, columns = list('ABCD'))
    train.set_index('TimPo')

    tiem1=train[:1]
    temporal1=tiem1.values.tolist()

    tiem2=train[:2]
    temporal2=tiem2.values.tolist()
    tiem3=train[:3]
    temporal3=tiem3.values.tolist()
    tiem4=train[:4]
    temporal4=tiem4.values.tolist()
    tiem5=train[:5]
    temporal5=tiem5.values.tolist()
    tiem6=train[:6]
    temporal6=tiem6.values.tolist()
    tiem7=train[:7]
    temporal7=tiem7.values.tolist()
    tiem8=train[:8]
    temporal8=tiem8.values.tolist()
    tiem9=train[:9]
    temporal9=tiem9.values.tolist()
    tiem10=train[:10]
    temporal10=tiem10.values.tolist()
    tiem11=train[:11]
    temporal11=tiem11.values.tolist()
    tiem12=train[:12]
    temporal12=tiem12.values.tolist()
    tiem14=train[:14]
    temporal14=tiem14.values.tolist()
    tiem15=train[:15]
    temporal15=tiem15.values.tolist()
    tiem16=train[:16]
    temporal16=tiem16.values.tolist()
    tiem17=train[:17]
    temporal17=tiem17.values.tolist()
    tiem18=train[:18]
    temporal18=tiem18.values.tolist()
    tiem19=train[:19]
    temporal19=tiem19.values.tolist()
    tiem20=train[:20]
    temporal20=tiem20.values.tolist()
    # tiem22=train[22]
    # temporal22=tiem22.values.tolist()
    tiem23=train[:23]
    temporal23=tiem23.values.tolist()
    tiem24=train[:24]
    temporal24=tiem24.values.tolist()
    tiem25=train[:25]
    temporal25=tiem25.values.tolist()
    tiem26=train[:26]
    temporal26=tiem26.values.tolist()
    tiem27=train[:27]
    temporal27=tiem27.values.tolist()
    tiem28=train[:28]
    temporal28=tiem28.values.tolist()
    tiem29=train[:29]
    temporal29=tiem29.values.tolist()
    tiem31=train[:31]
    temporal31=tiem31.values.tolist()

    print("temporal1",temporal1)
    print("temporal2",temporal2)

    x = np.linspace(1611201603,1611201633,num=30)
    print("x:",x)
    temporal1=temporal1[0]
    temporal1=temporal1[2:32]

    temporal2=temporal2[1]
    temporal2=temporal2[2:32]
    print("temporal1",temporal1)
    print("temporal2",temporal2)
    print("len(temporal1)",len(temporal1))
    print("len(temporal2)",len(temporal2))


    plt.stem(x,temporal2,'-.')
    plt.stem(x,temporal1,':')
    plt.show()


if __name__ == "__main__":
    main()
