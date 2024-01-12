import pandas as pd
file = pd.read_csv('PassivePile_Lai2020'+str(1)+r'\ksi.csv')['COORD.COOR2']
for i in range(1,84):
    try:
        file =pd.concat([file,pd.read_csv('PassivePile_Lai2020'+str(i)+r'\ksi.csv')['SM.SM1']],axis=1)
    except:
        print(str(i)+'does not exist')
file.to_csv('moments.csv')
print(file)