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
                df_wake.at[2,xx]=filtrado['time'].iloc[0]   #offset of n submeter
                df_type.at[1,xx]=singles[xx]
                df_type.at[2,xx]=filtrado['Type'].iloc[0]   #offset of n submeter
                df_type.at[3,xx]=average[0]        #mean wakeup time of every device
                df_scatter.append(diff)

        xx+=1

    df_mean=df_mean.dropna(axis=1)
    df_stadv=df_stadv.dropna(axis=1)
    tiempos=df_mean[1:]   #Cool! Keep going!
    stdev=df_stadv[1:]
    x=range(len(df_scatter[0]))
    flatList = [ item for elem in df_scatter[0] for item in elem]  #To take the values inside the inner lists to ground level of the main list
    stand0=pd.DataFrame({'x_axis': x, 'y_axis': flatList })

    plt.ylabel('Filtered time interval (min)')
    plt.xlabel('Consecutive ocurrences')
    plt.title('Time interval distribution for meter No.0 between consecutive reception of messages from the meters')
    plt.grid(True)
    plt.plot( 'x_axis', 'y_axis', data=stand0, linestyle='-', marker='o')

    i=0
    line1 = []
    line2 = []
    while i<len(x):
        line1.append(.019)
        line2.append(0.013)
        i+=1
    x1 = range(len(x))
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

    while i<len(temporal[0]):
        t.append(temporal[0][i])
        s.append(temporal2[0][i])
        i+=1

    x = np.arange(len(df_mean.columns)) #OK

    fig = plt.figure()
    ax = fig.add_axes([0.07,0.07,.9,.9])
    ax.bar(x,t,
        yerr=s,
        align='center',
        alpha=0.5,
        ecolor='black',
        capsize=15,
        color=(0.2, 0.4, 0.6, 0.6))
    plt.ylabel('Mean duty cycle (min)')
    plt.xlabel('Meters')
    plt.title('Mean interval period with standard deviation for every meter')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    stand=pd.DataFrame({'x_axis': range(len(s)), 'y_axis': s })

    plt.ylabel('Variance: \u03C3^2')
    plt.xlabel('Devices')
    plt.title('Variance of all the detected meters')
    plt.grid(True)
    plt.tight_layout()
    plt.plot( 'x_axis', 'y_axis', data=stand, linestyle='-', marker='o')

    i=0
    line1 = []
    line2 = []
    while i<len(s):
        line1.append(.141)
        line2.append(0.02)
        i+=1
    x1 = range(len(s))
    x2 = x1
    error1=pd.DataFrame({'x': x1, 'y': line1})
    error2=pd.DataFrame({'x': x1, 'y': line2})
    plt.plot( 'x', 'y', data=error1, linestyle='-', color='r')
    plt.plot( 'x', 'y', data=error2, linestyle='-', color='r')
    plt.show()

    a=df_type[0].iloc[2]
    b=df_type[15].iloc[2]
    c=df_type[11].iloc[2]
    d=df_type[3].iloc[2]

    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    cuatro =[df_type[0].iloc[2],df_type[1].iloc[2],df_type[2].iloc[2],df_type[4].iloc[2],df_type[5].iloc[2],df_type[6].iloc[2],df_type[7].iloc[2],df_type[8].iloc[2],df_type[14].iloc[2],df_type[17].iloc[2],df_type[22].iloc[2],df_type[24].iloc[2],df_type[25].iloc[2],df_type[28].iloc[2],df_type[34].iloc[2]]
    seis = [df_type[15].iloc[2],df_type[10].iloc[2],df_type[12].iloc[2],df_type[15].iloc[2],df_type[16].iloc[2],df_type[23].iloc[2],df_type[26].iloc[2],df_type[33].iloc[2]]
    siete = [df_type[11].iloc[2],df_type[18].iloc[2],df_type[19].iloc[2],df_type[20].iloc[2],df_type[27].iloc[2],df_type[29].iloc[2],df_type[31].iloc[2]]
    tys = [df_type[3].iloc[2],df_type[9].iloc[2]]

    br1 = np.arange(len(cuatro))
    a = len(cuatro)
    b = len(seis)
    br2 = np.arange(len(cuatro),len(cuatro)+len(seis))
    br3 = np.arange(len(cuatro)+len(seis),len(cuatro)+len(seis)+len(siete))
    br4 = np.arange(len(cuatro)+len(seis)+len(siete),len(cuatro)+len(seis)+len(siete)+len(tys))

    plt.bar(br1,cuatro, color = 'r', label='Heat meters')
    plt.bar(br2,seis, color = 'b', label='Warm water meter')
    plt.bar(br3,siete,color='y', label='Water meter')
    plt.bar(br4,tys,color='g',label='Radio converter')
    plt.legend(labels=['Heat meters', 'Warm water meter','Water meter','Radio converter'])

    plt.ylabel('Mean duty cycle (min)')
    plt.xlabel('Meters')
    plt.title('Mean interval period')
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['legend.framealpha'] = None
    plt.rcParams['legend.edgecolor'] = 'inherit'
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("singles:",singles)
    print("df_mean:",df_mean)

if __name__ == "__main__":
    main()
