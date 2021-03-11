def main():

    import csv
    import pandas as pd
    import statistics
    import numpy as np
    import math

    df = pd.read_csv("agustin_thesis(2).csv",low_memory=False)
    df=df.sort_values(['time'], inplace=False, ascending=True)
    ADD=[]
    TYPE=[]
    TYPEN=[]
    x=0
    lim = len(df)
    # print("df.columns1",df.columns)
    while x < lim:
        ADD.append(df.value[x][4:20])   #Checked[4:20]
        TYPE.append(df.value[x][18:20])
        # TYPE[x] = "0x" +TYPE[x]
        x+=1

    df['Address']=ADD #Added column of Address to the sorted list
    df['Type']=TYPE #Added column of TYPE pf meter to sorted list
    # print("df.columns2",df.columns)
    singles=pd.unique(ADD) #OK
    # print("singles:",singles)
    # print("df.Type:",df.Type)

    diff_list = []
    xx=0
    df_mean=pd.DataFrame()
    df_stadv=pd.DataFrame()
    flag=False
    df_wake = pd.DataFrame()
    df_type = pd.DataFrame()

    while xx<len(singles):

        temp=singles[xx]

        filtrado=df.query('Address==@temp',inplace=False)
        filtrado=filtrado.sort_values(['time'], inplace=False, ascending=True)
        filtrado['time']=filtrado['time'].div(1e+12) #[seg]
        print("filtrado.time:",filtrado.time)

        # print("dev:",temp)
        if len(filtrado)==1:

            flag=True
            average=0
        else:
            flag=False
            # print("filtrado:",filtrado.time)
            diferencia=filtrado.time.diff()
            # print("diferenciab:\n",diferencia)
            average=diferencia.mean()
            stadvt=diferencia.std(axis=0)
            stadv=math.sqrt(stadvt)
            # print("stadv:",stadv)

        if flag == False:
                tempx=temp
                df_mean.at[1,xx]=singles[xx]    #only the addresses of the devices
                df_mean.at[2,xx]=average        #mean wakeup time of every device
                df_stadv.at[1,xx]=singles[xx]
                df_stadv.at[2,xx]=stadv
                df_wake.at[1,xx]=singles[xx]
                df_wake.at[2,xx]=filtrado['time'].iloc[0]   #offset of n submeter
                df_type.at[1,xx]=singles[xx]
                df_type.at[2,xx]=filtrado['Type'].iloc[0]   #offset of n submeter
                df_type.at[3,xx]=average        #mean wakeup time of every device

                print("singles[xx]:",singles[xx])
                print("diferencia:",diferencia)
        xx+=1

    # print("df_type:",df_type)

    tiempos=df_mean[1:]   #Cool! Keep going!
    stdev=df_stadv[1:]
    print("diferencia:",diferencia)

    from matplotlib.ticker import FuncFormatter
    import matplotlib.pyplot as plt

    temporal=tiempos.values.tolist()
    temporal2=stdev.values.tolist()
    t = []  #mean times list
    s = []  #Standard Deviation list
    i=0
    #print("length(temporal)",len(temporal[0]))
    #print("temporal:",temporal)
    while i<len(temporal[0]):
        t.append(temporal[0][i])    #
        s.append(temporal2[0][i])
        i+=1

    # print("singles:",singles)
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
    plt.xlabel('Devices')
    plt.title('Mean interval period with standard deviation')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<


    # print("df_type:",df_type)
    #
    # print("df_type[0].iloc[2]:",df_type[1].iloc[2])

    # for element in uniqtypes:
    #     print("element:", element)
    #     print("query:",ttype.query('[2]==@element',inplace=False))

    # ttype[2]=ttype[2].astype(str)
    #
    # cuatros=ttype.query('[2]=="0x04"',inplace=False)
    # cuatros=cuatros.values.tolist()
    # seises=ttype.query([2]=="0x06",inplace=False)
    # seises=seises.values.tolist()
    # sietes=ttype.query([7]=="0x07",inplace=False)
    # sietes=sietes.values.tolist()
    # tys=ttype.query([2]=="0x37",inplace=False)
    # tys=tys.values.tolist()

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

    # print("len(seis):",len(seis))
    br1 = np.arange(len(cuatro))
    a = len(cuatro)
    b = len(seis)
    br2 = np.arange(len(cuatro),len(cuatro)+len(seis))
    br3 = np.arange(len(cuatro)+len(seis),len(cuatro)+len(seis)+len(siete))
    br4 = np.arange(len(cuatro)+len(seis)+len(siete),len(cuatro)+len(seis)+len(siete)+len(tys))

    # print("cuatro:",cuatro)
    # print("br1:",br1)
    # print("seis:",seis)
    # print("br2:",br2)
    # print("siete:",siete)
    # print("br3:",br3)
    # print("tys:",tys)
    # print("br4:",br4)



    # plt.bar(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'],cuatro, label = "04", color = 'r')
    plt.bar(br1,cuatro, color = 'r', label='Heat meters')
    plt.bar(br2,seis, color = 'b', label='Warm water meter')
    plt.bar(br3,siete,color='y', label='Water meter')
    plt.bar(br4,tys,color='g',label='Radio converter')
    plt.legend(labels=['Heat meters', 'Warm water meter','Water meter','Radio converter'])

    plt.ylabel('Mean duty cycle (min)')
    plt.xlabel('Devices')
    plt.title('Mean interval period with standard deviation')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    # df_mean.at[1,xx]=singles[xx]    #only the addresses of the devices
    # df_mean.at[2,xx]=average        #mean wakeup time of every device
    # df_stadv.at[1,xx]=singles[xx]
    # df_stadv.at[2,xx]=stadv
    # df_wake.at[1,xx]=singles[xx]
    # df_wake.at[2,xx]=filtrado['time'].iloc[0]   #offset of n submeter t0

    # DCS=[]#pd.DataFrame(columns='A','B')
    # dutyzkl=10*60*60*24     #=864k /day amount of cycles
    # MESS = 1
    # ON = False
    # begin = False
    #
    # k=1
    # x=1
    # y=2
    # n=1
    # j=2
    #
    # while k<n:
    #     j=x
    #     while j<y
    #         m=math.remainder(j, 2)
    #         if begin==False:
    #             n=k
    #             DCS[i,j] = 0 + offset(k)
    #             begin = [true]
    #         elseif m == 1
    #             DCS[i,j] = DCS[i-1,j+1] + wake(k)
    #         else m == 0
    #             DCS[i,j] = DCS[i,j-1] + MESS
    #         j+=1
    #
    #
    #
    #
    #
    # for k=1:1:length(wake)
    #      for i=1:1:dutyzkl   #24x60 = 1440
    #          for j=x:1:y
    #              m = rem(j,2)
    #              if begin==false
    #                  n=k
    #                  DCS(i,j) = 0 + offset(k)
    #                  begin = [true]
    #              elseif m == 1
    #                  DCS(i,j) = DCS(i-1,j+1) + wake(k)
    #              else m == 0
    #                  DCS(i,j) = DCS(i,j-1) + MESS
    #              end
    #          end
    #      end
    #      x=x+2;
    #      y=y+2
    #      begin=false
    # end
    # #
    # #
    # %------First meter-----
    # clear level;
    # level=true;
    # k=1;
    #     for j=1:1:length(DCS)
    #         xlim1 =DCS(j,k+1)-DCS(j,k);
    #
    #         if j~=length(DCS)
    #         xlim2 = DCS(j+1,k)-DCS(j,k+1)-1;                    %-1 because the last point is not included in the low range
    #         for i=length(level):1:length(level)+xlim1           %+1 porque hay que incluir el 1
    #             level(i,k)=true;
    #         end
    #         for(i=length(level)+xlim1+2:1:length(level)+xlim2)
    #             level(i,k)=false;
    #         end
    #         end
    #
    #     end



if __name__ == "__main__":
    main()
