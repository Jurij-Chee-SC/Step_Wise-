import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
import shutil
import subprocess
import importlib

output_gama = importlib.import_module('3output_gama')


print( 'This version is finished at the 01 Dec 2022 by Jurj Chee Szeje.')
print('-----------------------------------------------------------')
print('Please edit the input.csv file to set the import parameters.')
print('press any key to confirm the input')
#wt = input()
# ----------------input script--------------
D = 1.12
su_Es = 1/121.74
su0 = 1.0
k = 1.44
Gama = 6.0
Ep = 32.3*1000000
Lp = 21.6
Ece = 4.8
Ele_nums = int(Lp/0.2)
Frame_num = 84
JobName = 'PassivePile_Lai2020'
ksi_F = 0.2
alpha = 0.1
Nc_max = 9.14 + 2.8 *alpha
#-------------------------------


print('--------------WARNING------------')
print('In the file of y_matrix.csv, the unit of the soil movement is in metres')
# --------------input soil movement----------------
SoilMvmt_df = pd.read_csv('y_matrix.csv',sep=',')
def Y_yield(df):
    df_y = df.iloc[:,1:].copy()
    for i in range(0,25):
        for h in range(25,len(df_y.columns)):
            df_y.iloc[i,h] = df_y.iloc[i,25]
    for i in range(25,len(df_y)):
        for h in range(i,len(df_y.columns)):
            df_y.iloc[i, h] = df_y.iloc[i,i]
    df_y = pd.concat([df['Depth'],df_y],axis=1)
    return df_y
YYield_df = Y_yield(SoilMvmt_df)
print('the y_matrix.csv has been successfully loaded!')
def spring_info_generator(D_sp, su0_sp, k_sp, Gama_sp, J_sp, Lp_sp, Ece_sp, Ele_nums_sp,Frame_sp):
    Output = []
    for b in range(1, Ele_nums_sp+2):
        Output.append([b,(b-1)/Ele_nums*Lp_sp-Ece_sp])
    Output = pd.DataFrame(Output, columns = ['Node', 'Depth'])
    Output['Su(kPa)'] = Output['Depth']*k_sp+su0_sp
    Output['Nc'] =Nc_max
    Output[['Node','Depth']] = Output[['Depth','Node']]
    Output.columns=['Depth','Node','Su(kPa)','Nc']
    for c in range(0,len(Output)):
        Output['Su(kPa)'][c]=max([Output['Su(kPa)'][c],0])
        #--------------------Define the Nc------------------
        N_p0 = 11.94-(1-alpha)-(11.94-3.22)*(1-(Output['Depth'][c]/14.5/D)**0.6)**1.35
        Output['Nc'][c] = min([ N_p0+Gama*Output['Depth'][c]/Output['Depth'][c]/k,Nc_max])
    Output['pu(kN)']=Lp_sp/Ele_nums_sp*Output['Nc']*Output['Su(kPa)']*D_sp
    Output.loc[Output['pu(kN)'] < 0.001, 'pu(kN)'] = 0.001
    SoilMvmt = pd.concat([SoilMvmt_df.iloc[:, [0, Frame_sp+1]],YYield_df.iloc[:, Frame_sp+1]], axis =1)
    SoilMvmt.columns = ['Depth', 'y_soil (m)', 'y_yield (m)']
    New_Output = pd.merge(Output,SoilMvmt,on='Depth', how='outer')
    New_Output =New_Output.sort_values('Depth')
    New_Output.iloc[:] =New_Output.iloc[:].interpolate().fillna(0)
    New_Output =New_Output[New_Output['Depth'].isin(Output['Depth'])]
    #print(New_Output)
    return New_Output
#spring_info_generator(D, su0, k, Gama, J, Lp, Ece, Ele_nums,5).to_csv('11.csv')
#print(spring_info_generator(D, su0, k, Gama, J, Lp, Ece, Ele_nums))


def curve_generator(Pmax, Ysi, Y_yield, depth, y_P_ini):
    def cap_values(df, threshold):
        df.loc[df['y'] > threshold, 'P'] = df.loc[df['y'] <= threshold, 'P'].max()
        return df
    def func(x, ksi_F, P_u):
        ksi_P, d = x
        s_u = k * d + 0.1
        G_0 = 2060 * ((0.84 * Gama * d) ** 0.653) + 0.1
        P = (2 * ((abs(ksi_P) / ksi_F) ** 0.5) / (1 + (abs(ksi_P) / ksi_F))) * P_u*np.sign(ksi_P)
        y = (D * (2.8 *(2 * ((abs(ksi_P) / ksi_F) ** 0.5) / (1 + (abs(ksi_P)/ ksi_F))) / G_0 * s_u
                 + (1.35+0.25*alpha) *abs(ksi_P)))*np.sign(ksi_P)
        return P, y
        # Cast the parameters to float

    Pmax = float(Pmax)
    Ysi = float(Ysi)
    Y_yield = float(Y_yield)
    depth = float(depth)
    y_P_ini = float(y_P_ini)
    if depth < 0:
        depth = 0

    # Initialize the lists to store the computed P and y values
    P_values = []
    y_values = []

    # Determine the minimum and maximum y values
    y_min = func([-ksi_F, depth], ksi_F, Pmax)[1]
    y_max = func([ksi_F, depth], ksi_F, Pmax)[1]

    # Determine the number of ksi_P values to generate
    num_points = int(abs(y_max - y_min) / 0.001) + 1

    # Generate the ksi_P values
    ksi_P_values = np.linspace(-ksi_F, ksi_F, num_points)

    # Compute the P and y values
    for ksi_P in ksi_P_values:
        P, y = func([ksi_P, depth], ksi_F, Pmax)
        P_values.append(P)
        y_values.append(y)

    # Create the new y values with an interval of 0.001
    y_new = np.arange(min(y_values), max(y_values) + 0.001, 0.001)

    # Interpolate the P values
    P_new = np.interp(y_new, y_values, P_values)

    # Create the dataframe for the backbone curve
    df_backbone = pd.DataFrame({
        'P': P_new,
        'y': y_new
    })

    # Generate the unloading curve
    df_unloading = df_backbone.copy()
    Loading_curve_df=df_backbone.copy()
    if Ysi>y_P_ini:
        if Y_yield<Ysi:
            Loading_curve_df = cap_values(Loading_curve_df,Y_yield)
            P_ini = np.interp(Ysi - y_P_ini, Loading_curve_df['y'], Loading_curve_df['P'])
            df_unloading_scaled = df_backbone * 2
            df_unloading_scaled['P'] += P_ini
            df_unloading_scaled['y'] += Ysi - y_P_ini
            # Filter the dataframe to ensure that the 'y' values are within the range of the backbone curve
            df_unloading_filtered = df_unloading_scaled[
                (df_unloading_scaled['y'] >= min(df_backbone['y'])) & (df_unloading_scaled['y'] <= max(df_backbone['y']))]

            # Create the new y values with an interval of 0.001 for the unloading curve
            y_new_unloading = np.arange(min(df_unloading_filtered['y']), max(df_unloading_filtered['y']) + 0.001, 0.001)

            # Interpolate the 'P' values to match the new 'y' values of the unloading curve
            f_unloading = interp1d(df_unloading_filtered['y'], df_unloading_filtered['P'], kind='cubic',
                                   fill_value='extrapolate')
            P_new_unloading = f_unloading(y_new_unloading)

            # Create the dataframe for the unloading curve with the new 'y' values and interpolated 'P' values
            df_unloading = pd.DataFrame({
                'P': P_new_unloading,
                'y': y_new_unloading
            })
            if Y_yield < Ysi - y_P_ini:
                # Replace the middle part of the loading curve with the corresponding part of the unloading curve
                Loading_curve_df.loc[
                    (Loading_curve_df['y'] > Ysi - y_P_ini - 2 * Y_yield) & (Loading_curve_df['y'] <= Ysi - y_P_ini), 'P'] = \
                    df_unloading.loc[
                        (df_unloading['y'] > Ysi - y_P_ini - 2 * Y_yield) & (df_unloading['y'] <= Ysi - y_P_ini), 'P']

                # Replace the lower part of the loading curve with the corresponding part of the backbone curve plus Y_yield - y_P_ini
                df_backbone_moved = df_backbone.copy()
                df_backbone_moved['y']=df_backbone_moved['y']+Ysi-y_P_ini-Y_yield
                Loading_curve_df.loc[Loading_curve_df['y'] <= Ysi - y_P_ini - 2 * Y_yield, 'P'] = \
                    df_backbone_moved.loc[df_backbone_moved['y'] <= Ysi - y_P_ini - 2 *  Y_yield, 'P']
                Loading_curve_df.loc[Loading_curve_df['y'] <= Ysi - y_P_ini - 2 * Y_yield, 'y']= \
                    Loading_curve_df.loc[Loading_curve_df['y'] <= Ysi - y_P_ini - 2 * Y_yield, 'y']+Ysi-y_P_ini-Y_yield
            else:
                Loading_curve_df.loc[
                    (Loading_curve_df['y'] > y_P_ini - Ysi) & (Loading_curve_df['y'] <= Ysi - y_P_ini), 'P'] = \
                    df_unloading.loc[
                        (df_unloading['y'] > y_P_ini - Ysi) & (df_unloading['y'] <= Ysi - y_P_ini), 'P']
        else:
            P_ini = np.interp(Ysi - y_P_ini, Loading_curve_df['y'], Loading_curve_df['P'])
            df_unloading_scaled = df_backbone * 2
            df_unloading_scaled['P'] += P_ini
            df_unloading_scaled['y'] += Ysi - y_P_ini
            # Filter the dataframe to ensure that the 'y' values are within the range of the backbone curve
            df_unloading_filtered = df_unloading_scaled[
                (df_unloading_scaled['y'] >= min(df_backbone['y'])) & (df_unloading_scaled['y'] <= max(df_backbone['y']))]

            # Create the new y values with an interval of 0.001 for the unloading curve
            y_new_unloading = np.arange(min(df_unloading_filtered['y']), max(df_unloading_filtered['y']) + 0.001, 0.001)

            # Interpolate the 'P' values to match the new 'y' values of the unloading curve
            f_unloading = interp1d(df_unloading_filtered['y'], df_unloading_filtered['P'], kind='cubic',
                                   fill_value='extrapolate')
            P_new_unloading = f_unloading(y_new_unloading)

            # Create the dataframe for the unloading curve with the new 'y' values and interpolated 'P' values
            df_unloading = pd.DataFrame({
                'P': P_new_unloading,
                'y': y_new_unloading
            })
            Loading_curve_df.loc[
                (Loading_curve_df['y'] > y_P_ini - Ysi) & (Loading_curve_df['y'] <= Ysi - y_P_ini), 'P'] = \
                df_unloading.loc[
                    (df_unloading['y'] > y_P_ini - Ysi) & (df_unloading['y'] <= Ysi - y_P_ini), 'P']

    else:
        P_ini = np.interp(Ysi - y_P_ini, Loading_curve_df['y'], Loading_curve_df['P'])
        df_unloading_scaled = df_backbone * 2
        df_unloading_scaled['P'] += P_ini
        df_unloading_scaled['y'] += Ysi - y_P_ini
        # Filter the dataframe to ensure that the 'y' values are within the range of the backbone curve
        df_unloading_filtered = df_unloading_scaled[
            (df_unloading_scaled['y'] >= min(df_backbone['y'])) & (df_unloading_scaled['y'] <= max(df_backbone['y']))]

        # Create the new y values with an interval of 0.001 for the unloading curve
        y_new_unloading = np.arange(min(df_unloading_filtered['y']), max(df_unloading_filtered['y']) + 0.001, 0.001)

        # Interpolate the 'P' values to match the new 'y' values of the unloading curve
        f_unloading = interp1d(df_unloading_filtered['y'], df_unloading_filtered['P'], kind='cubic',
                               fill_value='extrapolate')
        P_new_unloading = f_unloading(y_new_unloading)

        # Create the dataframe for the unloading curve with the new 'y' values and interpolated 'P' values
        df_unloading = pd.DataFrame({
            'P': P_new_unloading,
            'y': y_new_unloading
        })
        Loading_curve_df.loc[
            (Loading_curve_df['y'] > Ysi-y_P_ini) & (Loading_curve_df['y'] <= y_P_ini - Ysi), 'P'] = \
            df_unloading.loc[
                (df_unloading['y'] > Ysi-y_P_ini) & (df_unloading['y'] <=  y_P_ini - Ysi), 'P']


    start_line = pd.DataFrame({
        'P': [-Pmax],
        'y': [-40]
    })

    end_line = pd.DataFrame({
        'P': [Loading_curve_df['P'].iloc[-1]],
        'y': [40]
    })
    Loading_curve_df.dropna(inplace=True)
    Loading_curve_df = pd.concat([start_line,Loading_curve_df,end_line],axis=0)
    Loading_curve_df = Loading_curve_df - Ysi
    Loading_curve_df = Loading_curve_df.applymap("{0:.6f}".format)
    curve_words =Loading_curve_df.to_csv(sep=',', index=False, header=False, line_terminator='\n').strip()+'\n'
    return curve_words


def inp_gen(frame):
    def convert_y_to_words(df):
        output_word = ''
        for i in range(1, len(df) - 1):
            output_word = output_word + r'beam-1.Beam_node.' + str(df['Node'][i]) + ',1,1,' + str(-df['yp'][i]) + '\n'
        return output_word
    if frame>1:
        Previous_result = pd.read_csv(JobName + str(frame - 1) + '\\' + 'ksi.csv')
        stable_inp_line = convert_y_to_words(Previous_result)
        load_inp_line = '*Initial Conditions, TYPE=STRESS,input=stress.csv\n'
    else:
        stable_inp_line =' '
        load_inp_line = ' '

    def node_generator(Lp_node, Ele_nums_node):
        Node_out = ''
        for d in range(1, Ele_nums_node + 2):
            Node_out = Node_out + str(d) + ',0,' + str(-(d - 1) / Ele_nums_node * Lp_node) + '\n'
        return Node_out
    # print(node_generator(Lp,Ele_nums))
    def ele_generator(Ele_nums_ele):
        ele_out = ''
        for e in range(1, Ele_nums_ele + 1):
            ele_out = ele_out + str(e) + ',' + str(e) + ',' + str(e + 1) + '\n'
        return ele_out
    def Load_generator():
        Ld_out = ''
        Ld_out = Ld_out + \
                 '*Step, name=stable, nlgeom=YES\n' + \
                 '*Static\n' + \
                 '100., 1000., 1e-05, 100.\n' + \
                 '*Boundary, op = NEW\n' + \
                 'Set - 2, 2, 2\n' + \
                 stable_inp_line + \
                 '*Restart, write, frequency=0\n' + \
                 '*Output, field\n' + \
                 '*Node Output\n' + \
                 'CF, RF, RM, U, COORD\n' + \
                 '*Element Output, directions=YES\n' + \
                 'LE, NFORCSO, PE, PEEQ, PEMAG, S, SF\n' + \
                 '*Contact Output\n' + \
                 'CDISP, CSTRESS\n' + \
                 '*Output, history, variable=PRESELECT\n'+'*End step\n'+ \
                 '*Step, name=load, nlgeom=YES\n' + \
                 '*Static\n' + \
                 '100., 1000., 1e-05, 100.\n' + \
                 '*Boundary, op = NEW\n' + \
                 'Set - 2, 2, 2\n' + \
                 '*Restart, write, frequency=0\n' + \
                 '*Output, field\n' + \
                 '*Node Output\n' + \
                 'CF, RF, RM, U, COORD\n' + \
                 '*Element Output, directions=YES\n' + \
                 'LE, NFORCSO, PE, PEEQ, PEMAG, S, SF\n' + \
                 '*Contact Output\n' + \
                 'CDISP, CSTRESS\n' + \
                 '*Output, history, variable=PRESELECT\n'+'*End step\n'
        return Ld_out
    def spring_script_Gen(Serial_num, spring_num):
        Serial_num = float(Serial_num)
        if frame==1:
            y_P_ini=0
        else:
            Previous_result = pd.read_csv(JobName + str(frame - 1) + '\\' + 'ksi.csv')
            y_P_ini=Previous_result['yp'][int(Serial_num) - 1]
        Pu = Info['pu(kN)'][int(Serial_num) - 1]
        if Pu<0.1: Pu =0.0
        Y = Info['y_soil (m)'][int(Serial_num) - 1]
        YYield = Info['y_yield (m)'][int(Serial_num) - 1]
        Script = \
            '*Spring, elset=Springs/Dashpots-' + str(int(spring_num)) + '-spring, nonlinear\n' \
            + '1\n' \
            + curve_generator(Pu, Y,YYield, Info['Depth'][int(Serial_num) - 1],y_P_ini) \
            + '*Element, type=Spring1, elset=Springs/Dashpots-' + str(int(spring_num)) + '-spring\n' \
            + str(int(Serial_num + 1000)) + ',' + str(int(Serial_num)) + '\n'
        return Script
    Info = pd.read_csv(JobName+str(frame)+'\\'+'spring_info.csv')

    #--generate spring inp
    Inp_spring_txt = ''
    for z in range(1,Ele_nums+2):
        Inp_spring_txt = Inp_spring_txt + spring_script_Gen(z, z)
    #-------------
    #--generate nodes
    #print(Load_generator(Delta))
    Inp_txt =  \
        '*Heading\n'+\
        '*Preprint, echo=NO, model=NO, history=NO, contact=NO\n' +\
        '*Part, name=beam\n' +\
        '*End Part\n' +\
        '*Assembly, name=Assembly\n' +\
        '*Instance, name=beam-1, part=beam\n'+\
        '          0.,        0,           0.\n'+\
        '*Node\n'+ \
        node_generator(Lp, Ele_nums) +\
        '*Element, type=B21\n' +\
        ele_generator(Ele_nums) +\
        '*Nset, nset=Beam_node, generate\n'+\
        '1,'+str(Ele_nums+1)+',1\n'+\
        '*Elset, elset=Beam_ele, generate\n'+\
        '1,'+str(Ele_nums)+',1\n'+\
        '*Beam Section, elset=Beam_ele, material=beam, temperature=GRADIENTS, section=CIRC\n' +\
        str(D/2)+'\n'+\
        '0., 0., -1.0\n'+ \
        Inp_spring_txt +\
        '*End Instance\n' +\
        '*Nset, nset=RP, instance=beam-1\n'+\
        '1,\n'+\
        '*Nset, nset=Set-2, instance=beam-1\n' +\
        str(Ele_nums+1)+',\n'+\
        '*End Assembly\n'+\
        '*Material, name=beam\n'+\
        '*Elastic\n'+\
        str(Ep)+', 0.2\n'+\
        '*Boundary\n'+\
        'Set-2, 2, 2\n'+\
        load_inp_line+\
        Load_generator()
    return Inp_txt

# This is the preparation for the computatiion
for i in range(1,Frame_num+1):
    try:
        os.mkdir(JobName+str(i))
    except:
        D=1.12
    spring_info_generator(D, su0, k, Gama, 1.5, Lp, Ece, Ele_nums, i).to_csv(JobName+str(i) + '\\' + 'spring_info.csv')
def copy_files_to_folders():
    # Define the source file paths
    source_file1 = '1Moment.py'
    source_file2 = '2Moment_ploting.py'
    # Loop over the folder names
    for i in range(1, 85):
        folder_name = JobName+str(i)

        # Copy the source files to the destination folder
        shutil.copy(source_file1, folder_name)
        shutil.copy(source_file2, folder_name)
copy_files_to_folders()


# Starting the iteration
for frame in range(1,Frame_num+1):

    with open(JobName+str(frame)+'\\'+JobName+'.inp', 'w') as g:
        g.write(inp_gen(frame))
    os.system('copy ' + JobName +str(frame-1)+ '\\stress.csv '+JobName+str(frame))
    os.chdir(JobName+str(frame))
    os.system('del'+JobName+r'.com')
    os.system('del'+JobName+r'.dat')
    os.system('del'+JobName+r'.msg')
    os.system('del'+JobName+'.prt')
    os.system('del'+JobName+'.sim')
    os.system('del'+JobName+'.sta')
    os.system('del'+JobName+'.odb')
    os.system('abq6141 double=both job='+JobName+' cpus=4 int')
    os.system('abq6141 CAE noGUI=1Moment.py')
    subprocess.call(['python', r'2Moment_ploting.py'])
    output_gama.Output_ksi(D=D,alpha=alpha,Ece=Ece,Gama=Gama)
    os.chdir(os.path.pardir)

'''
for frame in range(1,Frame_num+1):
    with open(JobName+str(frame)+'\\'+JobName+'.inp', 'w') as g:
        g.write(inp_gen(frame))
    os.chdir(JobName+str(frame))
    os.system('del'+JobName+r'.com')
    os.system('del'+JobName+r'.dat')
    os.system('del'+JobName+r'.msg')
    os.system('del'+JobName+'.prt')
    os.system('del'+JobName+'.sim')
    os.system('del'+JobName+'.sta')
    os.system('del'+JobName+'.odb')
    os.system('abq6141 double=both job='+JobName+' cpus=6 int')
    os.system('abq6141 CAE noGUI=1Moment.py')
    subprocess.call(['python', r'2Moment_ploting.py'])
    output_gama.Output_ksi(D=D,alpha=alpha,Ece=Ece,Gama=Gama)
    os.chdir(os.path.pardir)
'''
print('Inp generating finished!')

