import pandas as pd
def Output_ksi(D,alpha,Ece,Gama):
    df = pd.read_csv('moment.csv',sep='\s+')
    EcePile =-Ece
    df['Depth']=-df['COORD.COOR2']+EcePile
    df['U.U1'] = -df['U.U1']
    df.rename(columns={'U.U1': 'yp'}, inplace=True)
    df['ysi-yp']=pd.read_csv('spring_info.csv')['y_soil (m)']-df['yp']
    df['P']=df['SF.SF2'].diff()
    df['P_max']=pd.read_csv('spring_info.csv')['pu(kN)']
    df['G_max/s_u'] = 2060*(0.84*Gama*abs(df['Depth']))**0.653/pd.read_csv('spring_info.csv')['Su(kPa)']
    df['(ysi-yp)/D']=df['ysi-yp']/D
    df['ksi_e']=df['P']/df['P_max']/df['G_max/s_u']
    df['ksi_p']=(-2.8*df['ksi_e']+df['(ysi-yp)/D'])/(1.35+0.25*alpha)
    df.to_csv('ksi.csv')