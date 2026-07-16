# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:28:24 2021

@author: IGB
"""
#current season parameters
import pickle
import time
import ClassifyingData
from datetime import datetime
import numpy as np
import pandas as pd
import math
import os
import sys



# Field='Kelly'
# Field='Links'
# Field='S Koinzan'
Field='N Koinzan'
# ClassifyingData.run(Year,Field)
# print('Finished ClassifyingData')
#importing input files

from datetime import datetime
start_time = datetime.now()

# 'N Koinzan site1 (site N1)'
# longitude = -98.224144
# latitude = 42.015928

# site N2
# longitude = -88.415
# latitude = 40.595

# site N3
# longitude = -96.877
# latitude = 46.321

# site N4
# longitude = -94.6686
# latitude = 42.6816

# lon,lat
# B
# longitude = -88.455
# latitude = 40.807

# longitude = -95.7
# latitude = 39.561

# A
# longitude = -98.217
# latitude = 40.616

# longitude = -113.828
# latitude = 42.717

# C
# longitude = -87.6
# latitude = 41.5

# longitude = -85.6
# latitude = 42.5

# longitude = -97.363613
# latitude = 40.760541

longitude = -112.265
latitude = 41.735


calibration= 0 #is calibration included? input by users 1: need 0: no need (calibrated)
calibrated = abs(calibration-1)

real_irrigation= 0 #is real irrigation data included in application?

if real_irrigation== 1:
    ir_date = ['25-May-2024', '29-May-2024', '10-Jun-2024']  # real irrigation date input by users
    ir_depth = [10.0, 10.0, 10.0]  # real irrigation depth (mm) input by users

water_cost_per_ha_per_mm = 2.0
yield_price_per_ha_per_mm = 0.20

weight_index = 0.7

import Maize.Extract_tif
dir_croptype = './data/CropAT_US/CropType'
corn_year = Maize.Extract_tif.extract_crop_type_years(longitude, latitude, dir_croptype)
print('corn_year  '+ str(corn_year))

crop_type='Maize'
start_month = 4
start_day = 1
if corn_year is not None and corn_year != '' and corn_year != [] and corn_year != {}:
    start_year = corn_year
else:
    start_year = 2019
    #calibration = 0

end_month = 9
end_day = 10
end_year = start_year #here calibration is based on single year data

start_month1 = 3 # input by users
start_day1 = 1 # input by users
start_year1 = 2024
end_month1 = 7
end_day1 = 15
end_year1 = 2024

start_month2 = 7
start_day2 = 16
start_year2 = start_year1

end_year2 = start_year1
end_day2 = 1
#end_month2 = 11
if start_month2 < 6:
    end_month2 = start_month2 + 7
else:
    end_month2 = 12
    end_day2 = 1



#写一个start_day_seq2 to file_number 的函数
def list_folders_as_dates(folder_path):
    folder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    date_folders = []
    for name in folder_names:
        try:
            # 尝试将文件夹名转换为日期对象
            folder_date = datetime.strptime(name, '%Y%m%d')
            date_folders.append(folder_date)
        except ValueError:
            # 如果文件夹名不符合日期格式，忽略该文件夹
            continue
    return date_folders

def find_closest_folder(folder_path, target_date_str):
    target_date = datetime.strptime(target_date_str, '%Y%m%d')
    date_folders = list_folders_as_dates(folder_path)

    # 过滤出所有早于目标日期的文件夹
    previous_folders = [date for date in date_folders if date < target_date]

    if not previous_folders:
        return None  # 如果没有早于目标日期的文件夹，返回None

    # 找到最接近给定日期的文件夹
    closest_folder = max(previous_folders)
    closest_date_diff = (target_date - closest_folder).days

    # 如果最近的文件夹日期与目标日期的差小于等于4天，尝试找第二近的日期
    if closest_date_diff <= 4:
        # 移除最近的日期
        previous_folders.remove(closest_folder)
        if previous_folders:
            second_closest_folder = max(previous_folders)
            return second_closest_folder.strftime('%Y%m%d')
        else:
            return None  # 如果没有第二近的日期，返回None

    return closest_folder.strftime('%Y%m%d')

target_date_str = datetime(start_year2,start_month2, start_day2).strftime('%Y%m%d')
file_number = find_closest_folder('../../../PUB/S2S/V2023-07/Operational', target_date_str)
print("most recent s2s forecast nc folder", file_number)


with open("./Maize/start_day.pkl", "wb") as f:
    pickle.dump(start_day, f)
with open("./Maize/start_month.pkl", "wb") as f:
    pickle.dump(start_month, f)
with open("./Maize/start_year.pkl", "wb") as f:
    pickle.dump(start_year, f)
with open("./Maize/end_day.pkl", "wb") as f:
    pickle.dump(end_day, f)
with open("./Maize/end_month.pkl", "wb") as f:
    pickle.dump(end_month, f)
with open("./Maize/end_year.pkl", "wb") as f:
    pickle.dump(end_year, f)

crop_initial_pars = [0.024, 11.35, 0.5, 0.808]

lai_m_std=0.05#1%

planting_day_seq=datetime(start_year, start_month, start_day).timetuple()[7]
harvest_day_seq=datetime(end_year, end_month, end_day).timetuple()[7]


#go into Maize folder
os.chdir('./'+crop_type)
directory = os.getcwd()
sys.path.insert(1, directory)


import main
import numpy as np
import Read
import ChangeWeather
import Calibration
import ChangeIrrigation
import ChangeSwap
import Extract_tif
import pandas as pd
import ForecastStep
import use_s2s
import real_ir_update

start_month_cal=start_month
start_day_cal=start_day
start_year_cal=start_year
end_month_cal=end_month
end_day_cal=end_day
end_year_cal=end_year


year = start_year_cal

polaris_path = '../data/polaris'
dtw_path = '../data/dtw'
dem_path = '../data/dem'
tiledrain_path = '../data/tiledrain'
lai_path = '../data/lai_' + str(year)
era5_path = '../data/era5_' + str(year)

if calibration == 1:
    #Use MODIS LAI as measurments
    lai_modis = Extract_tif.extract_and_interpolate_lai(longitude, latitude, lai_path)['Value'].to_list()

    pd.DataFrame(lai_modis, columns=['lai_modis']).to_csv('lai_modis.csv')

    with open('lai_measurements_modis.pkl', "wb") as f:
        pickle.dump(lai_modis, f)

    #Use ERA5 meteo data

    ta = Extract_tif.extract_era_temperature_2m(longitude, latitude, os.path.join(era5_path ,'temperature_2m') , year)['Value'].to_list()
    tmin = Extract_tif.extract_era_temperature_2m_min(longitude, latitude, os.path.join(era5_path ,'temperature_2m_min') , year)['Value'].to_list()
    tmax = Extract_tif.extract_era_temperature_2m_max(longitude, latitude, os.path.join(era5_path ,'temperature_2m_max') , year)['Value'].to_list()
    tdew = Extract_tif.extract_era_dewpoint_temperature_2m(longitude, latitude, os.path.join(era5_path ,'dewpoint_temperature_2m'), year)['Value'].to_list()

    # surface_solar_radiation_downwards_sum
    # total_precipitation_sum
    # potential_evaporation_sum
    rad = Extract_tif.extract_era_surface_solar_radiation_downwards_sum(longitude, latitude, os.path.join(era5_path ,'surface_solar_radiation_downwards_sum'), year)['Value'].to_list()
    prec = Extract_tif.extract_era_total_precipitation_sum(longitude, latitude, os.path.join(era5_path ,'total_precipitation_sum'), year)['Value'].to_list()
    pet = Extract_tif.extract_era_potential_evaporation_sum(longitude, latitude, os.path.join(era5_path ,'potential_evaporation_sum'), year)['Value'].to_list()

    wind_u = Extract_tif.extract_era_u_component_of_wind_10m(longitude, latitude, os.path.join(era5_path ,'u_component_of_wind_10m'), year)['Value'].to_list()
    wind_v = Extract_tif.extract_era_v_component_of_wind_10m(longitude, latitude, os.path.join(era5_path ,'v_component_of_wind_10m'), year)['Value'].to_list()

    humd = [Extract_tif.calculate_relative_humidity(ta_d, tdew_d) for ta_d, tdew_d in zip(ta, tdew)]

    #Year	DOY	Solar	T-max	T-min	RelHum	Precip	ET

    df_era = pd.DataFrame(list(zip(rad, tmax, tmin, humd, prec, pet,wind_u,wind_v)), columns=['Solar','T-max','T-min','RelHum','Precip','ET','wind_u','wind_v'])
    df_era['WindSpeed'] = np.sqrt(df_era['wind_u']**2 + df_era['wind_v']**2)

    df_era['Year'] = year

    # Create the 'DOY' column with values from 1 to 364
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        df_era['DOY'] = range(1, 366)
    else:
        df_era['DOY'] = range(1, 365)

    df_era['Date'] = pd.to_datetime(df_era['Year'].astype(str) + df_era['DOY'].astype(str), format='%Y%j')
    df_era['Date'] = df_era['Date'].dt.strftime('%m/%d/%Y')
    df_era = df_era[['Date', 'Year', 'DOY', 'Solar','T-max','T-min','RelHum','Precip','ET','WindSpeed']]
    df_era.to_csv('./df_era.csv')

    df_era.to_excel('../data/weather_era.xlsx')

    # df_weather = pd.read_excel('data/Weather.xlsx')
    # df_weather['Solar']=df_weather['Solar']*1000#MJ/m2 to KJ/m2
    # t_mean=(df_weather['T-max']+df_weather['T-min'])/2
    # SVP=610.78*np.power(math.exp(1),(t_mean/(t_mean+238.3)*17.2694))/1000
    # df_weather['RelHum']=SVP*(1-df_weather['RelHum']/100)
    # wind_speed=pd.read_excel('data/WindSpeed.xlsx')
    # df_weather['WindSpeed']=wind_speed['Wind Speed']

    df_weather = pd.read_excel('../data/weather_era.xlsx')

    t_mean=(df_weather['T-max']+df_weather['T-min'])/2
    SVP=610.78*np.power(math.exp(1),(t_mean/(t_mean+238.3)*17.2694))/1000
    df_weather['RelHum']=SVP*(1-df_weather['RelHum']/100)

    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%m/%d/%Y')
    weather=df_weather
    weather["WindSpeed"][np.isnan(weather["WindSpeed"])]=3.5
    weather["month"]=weather['Date'].dt.month
    weather["day"]=weather['Date'].dt.day
    weather["year"]=weather['Date'].dt.year
    weather["station"]= "'Weather'"
    weather=weather.drop(['DOY', 'Unnamed: 0', 'Date','Year'], axis=1)
    titles=["station","day","month","year","Solar","T-min","T-max"\
    ,"RelHum","WindSpeed","Precip","ET"]
    weather=weather.reindex(columns=titles)
    weather.to_csv('./weather_era_out.csv')

    weather=weather.values.tolist()
    temp=[(start_year_cal),(end_year_cal)]

    for i in range(len(weather)):
        weather[i][7]=max(0.0,weather[i][7])
        weather[i][8]=max(0.0,weather[i][8])
        weather[i][9]=max(0.0,weather[i][9])
        weather[i][10] = max(0.0, weather[i][10])
        weather[i][5]=min(weather[i][5],weather[i][6]*0.95)

    #change weather data
    for i in range(end_year_cal-start_year_cal+1):
        ChangeWeather.change_weather(start_day_seq=1, Year=temp[i], data=weather)


#replace POLARIS data
polaris_data = Extract_tif.extract_polaris(longitude, latitude, polaris_path)
df_polaris = Extract_tif.process_polaris_data(polaris_data)
df_polaris.to_csv('./df_polaris_soil_hydraulic.csv')
list_polaris =df_polaris.values.tolist()

ChangeSwap.change_soilhydraulic(list_polaris) #SwapOriginal.swp to Swap1.swp

crp_file_path = 'gmaized.crp'
swp_file_path = 'Swap1.swp'
#end_file_path = 'result_p1.end'
#swap_executable = 'swap_test'

def use_crp_cali_pars(crp_file_path,crop_initial_pars):
    rgrlai_cali = round(crop_initial_pars[0],3)
    tbase_cali = round(crop_initial_pars[1], 3)
    eff_cali = round(crop_initial_pars[2], 3)
    cvl_cali = round(crop_initial_pars[3], 3)

    with open(crp_file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if "RGRLAI" in line and 'Maximum relative increase in LAI' in line:
            lines[i] = f"  RGRLAI = {rgrlai_cali:.3f}    ! Maximum relative increase in LAI [0..1 m2/m2/d, R]****************************FOR CALIBRATION - LAI RELATED\n"
            break
    with open(crp_file_path, 'w') as file:
        file.writelines(lines)

    with open(crp_file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if "TBASE" in line and 'Lower threshold temperature for ageing of leaves' in line:
            lines[i] = f"  TBASE  =    {tbase_cali:.3f} ! Lower threshold temperature for ageing of leaves ,[-10..30 C, R]**************************FOR CALIBRATION - LAI RELATED\n"
            break
    with open(crp_file_path, 'w') as file:
        file.writelines(lines)

    with open(crp_file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if "EFF" in line and 'Light use efficiency for real leaf' in line:
            lines[i] = f"  EFF    =    {eff_cali:.3f}  ! Light use efficiency for real leaf [0..10 kg CO2 /J adsorbed), R]**************************FOR CALIBRATION - LAI RELATED\n"
            break
    with open(crp_file_path, 'w') as file:
        file.writelines(lines)

    with open(crp_file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if "CVL" in line and 'efficiency of conversion into leaves' in line:
            lines[i] = f"  CVL    =   {cvl_cali:.3f}  ! efficiency of conversion into leaves [kg kg-1]**************************FOR CALIBRATION - LAI RELATED\n"
            break
    with open(crp_file_path, 'w') as file:
        file.writelines(lines)

#running
#calibration

# soil_initial_pars=np.array([])
# with open("Swap "+Field+".swp") as f:
#     line=f.readline()
#     while line[:17]!='  ISOILLAY1  ORES':
#         line=f.readline()
#     line=f.readline()
#     while line[:5]!='* ---':
#         temp=np.fromstring(line,dtype=float,sep='    ')
#         soil_initial_pars=np.append(soil_initial_pars,temp[1:6])
#         line=f.readline()

#crop_initial_pars=Read.cropars('initial')

# def modify_RET_lat(lat_value,ademvalue,meteo_etref):
#     for swp_file_path in ['SwapOriginal.swp','Swap1.swp','swap.swp']:
#         with open(swp_file_path, 'r') as file:
#             lines = file.readlines()
#         for i, line in enumerate(lines):
#             if "LAT    =" in line and 'Latitude of meteo station' in line:
#                 lines[i] = f"  LAT    =   {lat_value}       ! Latitude of meteo station, [-60..60 degrees, R, North = +]\n"
#                 break
#         with open(swp_file_path, 'w') as file:
#             file.writelines(lines)
#
#         with open(swp_file_path, 'r') as file:
#             lines = file.readlines()
#         for i, line in enumerate(lines):
#             if "ALT    =" in line and 'Altitude of meteo station' in line:
#                 lines[i] = f"  ALT    =   {ademvalue}       ! Altitude of meteo station, [-400..3000 m, R]\n"
#                 break
#         with open(swp_file_path, 'w') as file:
#             file.writelines(lines)
#
#         with open(swp_file_path, 'r') as file:
#             lines = file.readlines()
#         for i, line in enumerate(lines):
#             if "SWETR  =" in line and 'Switch, use reference ET values of meteo file' in line:
#                 lines[i] = f"  SWETR  =  {meteo_etref}           ! Switch, use reference ET values of meteo file [Y=1, N=0]\n"
#                 break
#         with open(swp_file_path, 'w') as file:
#             file.writelines(lines)

#replace DEM and GWT data
demvalue = Extract_tif.extract_tif_single_value(longitude, latitude, dem_path)

dtwvalue = Extract_tif.extract_tif_single_value(longitude, latitude, dtw_path)

tiledrainvalue = Extract_tif.extract_tif_single_value(longitude, latitude, tiledrain_path)


def modify_RET_lat(lat_value, dem_value, meteo_etref):
    swp_files = ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']
    for swp_file_path in swp_files:
        if os.path.exists(swp_file_path):
            with open(swp_file_path, 'r') as file:
                lines = file.readlines()

            modified = False
            for i, line in enumerate(lines):
                if "LAT    =" in line and 'Latitude of meteo station' in line:
                    lines[i] = f"  LAT    =   {lat_value}       ! Latitude of meteo station, [-60..60 degrees, R, North = +]\n"
                    modified = True
                elif "ALT    =" in line and 'Altitude of meteo station' in line:
                    lines[i] = f"  ALT    =   {round(float(dem_value),1)}       ! Altitude of meteo station, [-400..3000 m, R]\n"
                    modified = True
                elif "SWETR  =" in line and 'Switch, use reference ET values of meteo file' in line:
                    lines[i] = f"  SWETR  =  {meteo_etref}           ! Switch, use reference ET values of meteo file [Y=1, N=0]\n"
                    modified = True

            if modified:
                with open(swp_file_path, 'w') as file:
                    file.writelines(lines)
        else:
            print(f"File {swp_file_path} not found.")


modify_RET_lat(latitude, demvalue,0 )

def modify_dtw_tiledrain(dtw_value,tiledrain_value):
    if dtw_value >= 0:
        dtw_value2 = min((-100.0) * dtw_value, -100.0)
    elif dtw_value < 0:
        dtw_value2 = -100.0
    else:
        dtw_value2 = -200.0

    for swp_file_path in ['SwapOriginal.swp','Swap1.swp','swap.swp']:
        with open(swp_file_path, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if "GWLI   =" in line and 'Initial groundwater level, [-10000..100 cm, R]' in line:
                lines[i] = f"  GWLI   = {round(float(dtw_value2),1)}  ! Initial groundwater level, [-10000..100 cm, R]\n"
                break
        with open(swp_file_path, 'w') as file:
            file.writelines(lines)

        with open(swp_file_path, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if "  01-jan-2010     " in line:
                lines[i] = f"  01-jan-2010     {round(float(dtw_value2),1)}\n"
                break
        with open(swp_file_path, 'w') as file:
            file.writelines(lines)

        with open(swp_file_path, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if "  31-dec-2030     " in line:
                lines[i] = f"  31-dec-2030     {round(float(dtw_value2),1)}\n"
                break
        with open(swp_file_path, 'w') as file:
            file.writelines(lines)


        with open(swp_file_path, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if "SWDRA =" in line and 'Switch, simulation of lateral drainage' in line:
                lines[i] = f"  SWDRA = {tiledrain_value}  ! Switch, simulation of lateral drainage:\n"
                break
        with open(swp_file_path, 'w') as file:
            file.writelines(lines)

modify_dtw_tiledrain(dtwvalue,tiledrainvalue)


crop_initial_pars = [0.024, 11.35, 0.5, 0.808] #default parameters


if calibration == 1:
    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
        real_ir_update.modify_irrigation_swp(swp_file_path, 0)

    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 1)
        real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)

    #ChangeIrrigation.irrigation(i, start_year_cal, irrigation_cal, start_day_cal)
    # soil_initial_pars, crop_initial_pars = Calibration.calibration(soil_initial_pars, crop_initial_pars,
    #                                                                start_month_cal, start_day_cal, start_year_cal,
    #                                                                end_month_cal, end_day_cal, end_year_cal)
    # crop_initial_pars = Calibration.calibration(crop_initial_pars, start_month_cal, start_day_cal,start_year_cal, end_month_cal, end_day_cal, end_year_cal)

    _, crop_initial_pars_cali, best_value = Calibration.calibration(crop_initial_pars,start_month_cal, start_day_cal, start_year_cal,end_month_cal, end_day_cal, end_year_cal)

    pd.DataFrame(crop_initial_pars_cali, columns=["Parameters"]).to_csv('crop_initial_pars_cali.csv')
    pd.DataFrame([[best_value]], columns=["cali_best_obj_value"]).to_csv('cali_best_obj_value.csv')
    calibrated = 1
    #soil_initial_pars, crop_initial_pars = calibration(shp_lb,shp_ub,crop_initial_pars, start_month, start_day, start_year, end_month, end_day, end_year):


from datetime import datetime

# update calibrated parameters
if (calibration == 1) & (pd.read_csv('cali_best_obj_value.csv')['cali_best_obj_value'].to_list()[0] < 0.6):
    try:
        crop_initial_pars_cali = pd.read_csv('crop_initial_pars_cali.csv')['Parameters'].to_list()
    except ValueError as e:
        print("Error crop para:", e)
elif (calibrated == 1) & (pd.read_csv('cali_best_obj_value.csv')['cali_best_obj_value'].to_list()[0] < 0.6):
    try:
        crop_initial_pars_cali = pd.read_csv('crop_initial_pars_cali.csv')['Parameters'].to_list()
    except ValueError as e:
        print("Error crop para:", e)
else:
    crop_initial_pars_cali = crop_initial_pars

for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
    use_crp_cali_pars(crp_file_path, [0.024, 11.35, 0.5, 0.808])


start_day_seq1 =datetime(start_year1, start_month1, start_day1).timetuple()[7]
end_day_seq1 =datetime(end_year1, end_month1, end_day1).timetuple()[7]

start_day_seq2 =datetime(start_year2, start_month2, start_day2).timetuple()[7]
end_day_seq2 =datetime(end_year2, end_month2, end_day2).timetuple()[7]


if real_irrigation == 1:
    # Modify SWAP's default irrigation mode if the user has entered irrigation information prior to the current date
    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 1)
        real_ir_update.modify_schedule_irrigation_date(crp_file_path, start_day2, start_month2)
    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
        real_ir_update.modify_irrigation_swp(swp_file_path, 1)
        real_ir_update.update_swp_file(swp_file_path, ir_date, ir_depth)

else:
    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 0)
        # real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)
    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:

        real_ir_update.modify_irrigation_swp(swp_file_path, 1)



seq_s2s_start = datetime.strptime(file_number, '%Y%m%d').timetuple().tm_yday + 2
print('s2s start day')
print(seq_s2s_start)

modify_RET_lat(latitude, demvalue,0 )
modify_dtw_tiledrain(dtwvalue,tiledrainvalue)

# if real_irrigation== 1:
#     ir_date = ['25-May-2024', '29-May-2024', '10-Jun-2024']  # real irrigation date input by users
#     ir_depth = [10.0, 20.0, 20.0]  # real irrigation depth (mm) input by users

use_s2s.gridmet_processing(start_year1,1, end_day_seq1, file_number, latitude, longitude, start_day_seq2)
use_s2s.s2s_processing(start_year1, end_day_seq2, file_number, latitude, longitude,start_day_seq2)


import pandas as pd
from datetime import datetime, timedelta

# 初始日期
start_date = datetime(start_year2, start_month2, start_day2)

# 定义列名
columns = ['Date', 'Daynr', 'Daycrp', 'DVS', 'TSUM', 'LAIpot', 'LAI', 'Height', 'CrpFac', 'RootdPot',
           'Rootd', 'PWLV', 'WLV', 'PWST', 'WST', 'PWRT', 'WRT', 'CPWDM', 'CWDM', 'CPWSO', 'CWSO',
           'PGRASSDM', 'GRASSDM', 'PMOWDM', 'MOWDM', 'PGRAZDM', 'GRAZDM', 'DWLVCROP', 'DWLVSOIL',
           'DWST', 'DWRT', 'DWSO', 'HarLosOrm']

# 函数：从文件中读取数据并检查列存在性
# def read_data(file_path, required_columns):
#     data = []
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             if 'Date' not in line:
#                 values = line.strip().split(',')
#                 if len(values) == len(columns):
#                     data.append(values)
#                 else:
#                     print(f"Skipping malformed line: {line}")
#     df = pd.DataFrame(data, columns=columns)
#     missing_cols = set(required_columns) - set(df.columns)
#     if missing_cols:
#         raise KeyError(f"Missing columns in data: {missing_cols}")
#     return df

def read_data(file_path, required_columns):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[7:]  # 跳过前7行
        for line in lines:
            values = line.strip().split(',')
            if len(values) == len(columns):  # 确保行数据长度与列数一致
                data.append(values)
            else:
                print(f"Skipping malformed line: {line}")
    df = pd.DataFrame(data, columns=columns)
    df = df[required_columns]
    return df

# 函数：检查并转换为数值类型
def to_numeric(df, cols):
    for col in cols:
        df = df[df[col].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
        df[col] = pd.to_numeric(df[col])
    return df


if real_irrigation == 1 and calibrated == 1:
    ir_date2 = ir_date.copy()
    ir_depth2 = ir_depth.copy()

    dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)]

    first_date = True
    results_all_dates = []
    all_ensemble_results = []

    # 计算 file_number 的组合
    target_date_str = datetime(start_year2, start_month2, start_day2).strftime('%Y%m%d')
    file_number = find_closest_folder('../../../../PUB/S2S/V2023-07/Operational', target_date_str)
    date_format = '%Y%m%d'
    base_date = datetime.strptime(file_number, date_format)
    file_numbers = [
        file_number,
        (base_date - timedelta(days=5)).strftime(date_format),
        (base_date - timedelta(days=10)).strftime(date_format)
    ]

    # 定义sf_time和exp_n的组合
    sf_time_exp_n_combinations = [('06', '02'), ('06', '00'), ('00', '00')]

    for current_date in dates_to_evaluate:
        date_t = current_date.strftime('%d-%b-%Y')
        doy = current_date.timetuple().tm_yday

        # 初始化用于存储每个预测场景结果的列表
        ensemble_results = []


        # 遍历9种预测场景
        for file_num_ens in file_numbers:
            for sf_time, exp_n in sf_time_exp_n_combinations:
                results = []

                use_s2s.s2s_processing2(start_year1, end_day_seq2, file_num_ens, sf_time, exp_n, latitude, longitude,start_day_seq2)

                for ir in [0, 10, 15, 20, 25, 30, 40, 60]:
                    temp_ir_date2 = ir_date2.copy()
                    temp_ir_depth2 = ir_depth2.copy()
                    # 附加 results_all_dates 中的灌溉日期和灌溉量
                    for result in results_all_dates:
                        result_date = result[0]  # 从 result 中提取日期
                        result_ir = result[1]  # 从 result 中提取灌溉量

                        # 将结果附加到 temp_ir_date2 和 temp_ir_depth2
                        temp_ir_date2.append(result_date)
                        temp_ir_depth2.append(result_ir)

                    temp_ir_date2.append(date_t)
                    temp_ir_depth2.append(ir)

                    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
                        real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)

                    ForecastStep.run_sub1(start_day_seq1, start_year1, doy + 7, start_year2, start_day_seq2,
                                          start_year2, 'gmaized.crp', 'swap.swp', divide=0)

                    df = read_data('result_forec.crp', ['Daynr', 'DVS', 'CWDM', 'CWSO'])
                    df = df.dropna(subset=['Daynr', 'CWDM', 'CWSO', 'DVS'])
                    df = to_numeric(df, ['CWDM', 'CWSO', 'Daynr', 'DVS'])

                    cwdm_value = df['CWDM'].iloc[-1]
                    cwso_value = df['CWSO'].iloc[-1]
                    doy_max = df['Daynr'].iloc[-1]

                    # Determine coefficient based on DVS value
                    dvs_value = df['DVS'].iloc[-1]

                    if ir == 0:
                        target_value = 0
                        cwdm_ir0 = cwdm_value
                    else:
                        target_value = (cwdm_value - cwdm_ir0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index

                    #results.append((date_t, ir, cwdm_value, cwso_value, target_value))
                    results.append((date_t, ir, cwdm_value, cwso_value, target_value, file_num_ens, sf_time, exp_n))

                # 找出当前场景下的最优灌溉策略
                results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value',
                                                            'file_num_ens', 'sf_time', 'exp_n'])
                max_target_row = results_df.loc[results_df['target_value'].idxmax()]

                ensemble_results.append(max_target_row)

        all_ensemble_results.append(ensemble_results)
        # 计算9种预测场景下最优灌溉量的均值
        ensemble_results_df = pd.DataFrame(ensemble_results)

        mean_ir = ensemble_results_df['ir'].mean()
        mean_target_value = ensemble_results_df['target_value'].mean()

        # 保存当前日期的最优结果
        results_all_dates.append((date_t, mean_ir, mean_target_value))

        if first_date:
            pd.DataFrame([results_all_dates[-1]], columns=['date_t', 'mean_ir', 'mean_target_value']).to_csv(
                'day_scheduled.csv', index=False)
            first_date = False
        else:
            pd.DataFrame([results_all_dates[-1]], columns=['date_t', 'mean_ir', 'mean_target_value']).to_csv(
                'day_scheduled.csv', mode='a', header=False, index=False)

        # 检查生长发育阶段
        current_dvs = float(df['DVS'].iloc[-1])
        if (current_dvs >= 1.99) & (doy + 4 >= float(doy_max)):
        #if current_dvs >= 1.99:
            break

    # 循环结束后，将所有结果合并到一个 DataFrame 中
    all_ensemble_results_df = pd.concat([pd.DataFrame(res) for res in all_ensemble_results], ignore_index=True)

    # 导出所有结果到一个 CSV 文件
    all_ensemble_results_df.to_csv('all_day_ir_var_results.csv', index=False)

elif real_irrigation == 0 and calibrated == 1:
    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 1)
        real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)
    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:

        real_ir_update.modify_irrigation_swp(swp_file_path, 0)

    ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq1, start_year1, start_day_seq2, start_year2,
                          'gmaized.crp', 'swap.swp', divide=0)

    ir_date = []
    ir_depth = []

    with open('result_forec.irg', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'mais            ,' in line:
                values = line.strip().split(',')
                date_str = values[1].strip()
                irrigation = float(values[4].strip()) * 10

                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d-%b-%Y')
                doy = date_obj.timetuple().tm_yday
                if doy < start_day_seq2 -1:
                    ir_date.append(formatted_date)
                    ir_depth.append(irrigation)

    ir_date2 = ir_date.copy()
    ir_depth2 = ir_depth.copy()

    start_date = datetime(start_year2, start_month2, start_day2)
    dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)]

    first_date = True
    results_all_dates = []
    all_ensemble_results = []


    # 计算 file_number 的组合
    target_date_str = datetime(start_year2, start_month2, start_day2).strftime('%Y%m%d')
    file_number = find_closest_folder('../../../../PUB/S2S/V2023-07/Operational', target_date_str)
    date_format = '%Y%m%d'
    base_date = datetime.strptime(file_number, date_format)
    file_numbers = [
        file_number,
        (base_date - timedelta(days=5)).strftime(date_format),
        (base_date - timedelta(days=10)).strftime(date_format)
    ]
    print(file_numbers)

    # 定义sf_time和exp_n的组合
    sf_time_exp_n_combinations = [('06', '02'), ('06', '00'), ('00', '00')]

    for current_date in dates_to_evaluate:
        ensemble_results = []
        date_t = current_date.strftime('%d-%b-%Y')
        doy = current_date.timetuple().tm_yday


        # 遍历9种预测场景
        for file_num_ens in file_numbers:
            for sf_time, exp_n in sf_time_exp_n_combinations:

                use_s2s.s2s_processing2(start_year1, end_day_seq2, file_num_ens, sf_time, exp_n, latitude, longitude,start_day_seq2)

                for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
                    real_ir_update.modify_irrigation_crp(crp_file_path, 0)

                for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
                    real_ir_update.modify_irrigation_swp(swp_file_path, 1)


                results = []

                for ir in [0, 10, 15, 20, 25, 30, 40, 60]:
                    temp_ir_date2 = ir_date2.copy()
                    temp_ir_depth2 = ir_depth2.copy()
                    # 附加 results_all_dates 中的灌溉日期和灌溉量
                    for result in results_all_dates:
                        result_date = result[0]  # 从 result 中提取日期
                        result_ir = result[1]  # 从 result 中提取灌溉量
                        # 将结果附加到 temp_ir_date2 和 temp_ir_depth2
                        temp_ir_date2.append(result_date)
                        temp_ir_depth2.append(result_ir)

                    temp_ir_date2.append(date_t)
                    temp_ir_depth2.append(ir)

                    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
                        real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)

                    ForecastStep.run_sub1(start_day_seq1, start_year1, doy + 7, start_year2, start_day_seq2,
                                          start_year2, 'gmaized.crp', 'swap.swp', divide=0)

                    df = read_data('result_forec.crp', ['Daynr', 'DVS', 'CWDM', 'CWSO'])
                    df = df.dropna(subset=['Daynr', 'CWDM', 'CWSO', 'DVS'])
                    df = to_numeric(df, ['CWDM', 'CWSO', 'Daynr', 'DVS'])

                    cwdm_value = df['CWDM'].iloc[-1]
                    cwso_value = df['CWSO'].iloc[-1]
                    doy_max = df['Daynr'].iloc[-1]

                    # Determine coefficient based on DVS value
                    dvs_value = df['DVS'].iloc[-1]

                    if ir == 0:
                        target_value = 0
                        cwdm_ir0 = cwdm_value
                    else:
                        target_value = (cwdm_value - cwdm_ir0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index

                    #results.append((date_t, ir, cwdm_value, cwso_value, target_value))
                    results.append((date_t, ir, cwdm_value, cwso_value, target_value, file_num_ens, sf_time, exp_n))

                results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value',
                                                            'file_num_ens', 'sf_time', 'exp_n'])
                #results_df.to_csv('day_scheduled00.csv', mode='a', header=False, index=False)

                max_target_row = results_df.loc[results_df['target_value'].idxmax()]

                ensemble_results.append(max_target_row)

        all_ensemble_results.append(ensemble_results)
        # 计算9种预测场景下最优灌溉量的均值
        ensemble_results_df = pd.DataFrame(ensemble_results)

        mean_ir = ensemble_results_df['ir'].mean()
        mean_target_value = ensemble_results_df['target_value'].mean()

        # 保存当前日期的最优结果
        results_all_dates.append((date_t, mean_ir, mean_target_value))

        if first_date:
            pd.DataFrame([results_all_dates[-1]], columns=['date_t', 'mean_ir', 'mean_target_value']).to_csv(
                'day_scheduled.csv', index=False)
            first_date = False
        else:
            pd.DataFrame([results_all_dates[-1]], columns=['date_t', 'mean_ir', 'mean_target_value']).to_csv(
                'day_scheduled.csv', mode='a', header=False, index=False)

        # 检查生长发育阶段
        current_dvs = float(df['DVS'].iloc[-1])
        if (current_dvs >= 1.99) & (doy + 4 >= float(doy_max)):
            break

    # 循环结束后，将所有结果合并到一个 DataFrame 中
    all_ensemble_results_df = pd.concat([pd.DataFrame(res) for res in all_ensemble_results], ignore_index=True)

    # 导出所有结果到一个 CSV 文件
    all_ensemble_results_df.to_csv('all_day_ir_var_results.csv', index=False)

if calibrated == 1:
    temp_ir_date2 = ir_date2.copy()
    temp_ir_depth2 = ir_depth2.copy()

    # 附加 results_all_dates 中的灌溉日期和灌溉量
    for result in results_all_dates:
        result_date = result[0]  # 从 result 中提取日期
        result_ir = result[1]  # 从 result 中提取灌溉量

        # 将结果附加到 temp_ir_date2 和 temp_ir_depth2
        temp_ir_date2.append(result_date)
        temp_ir_depth2.append(result_ir)

    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 0)

    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:

        real_ir_update.modify_irrigation_swp(swp_file_path, 1)

        real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)

    use_s2s.s2s_processing(start_year1, end_day_seq2, file_number, latitude, longitude,start_day_seq2)

    # 运行子过程
    ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq2, start_year2, start_day_seq2, start_year2,'gmaized.crp', 'swap.swp', divide=0)











# if calibrated == 1:
#     temp_ir_date2 = ir_date2.copy()
#     temp_ir_depth2 = ir_depth2.copy()
#
#     for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#         real_ir_update.modify_irrigation_crp(crp_file_path, 0)
#         real_ir_update.modify_irrigation_swp(swp_file_path, 1)
#
#         real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)
#
#     use_s2s.s2s_processing(start_year1, end_day_seq2, file_number, latitude, longitude)
#     ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq2, start_year2, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)






# # 初始数据准备和主循环
# if real_irrigation == 1 and calibrated == 1:
#     ir_date2 = ir_date.copy()
#     ir_depth2 = ir_depth.copy()
#
#     dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)]
#
#     first_date = True
#     results_all_dates = []
#
#     for current_date in dates_to_evaluate:
#         results = []
#         date_t = current_date.strftime('%d-%b-%Y')
#         doy = current_date.timetuple().tm_yday
#
#         for ir in [0, 10, 15, 20, 25, 30, 40, 60]:
#             temp_ir_date2 = ir_date2.copy()
#             temp_ir_depth2 = ir_depth2.copy()
#             temp_ir_date2.append(date_t)
#             temp_ir_depth2.append(ir)
#
#             for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#                 real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)
#
#             ForecastStep.run_sub1(start_day_seq1, start_year1, doy + 4, start_year2, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)
#
#             # 读取并检查数据
#             df = read_data('result_forec.crp', ['Daynr', 'CWDM', 'CWSO'])
#
#             df = df.dropna(subset=['Daynr', 'CWDM', 'CWSO'])
#             df = to_numeric(df, ['CWDM', 'CWSO', 'Daynr'])
#
#             cwdm_value = df['CWDM'].iloc[-1]
#             cwso_value = df['CWSO'].iloc[-1]
#             cwdm_value_0 = df['CWDM'].iloc[-4]
#             cwso_value_0 = df['CWSO'].iloc[-4]
#             doy_max = df['Daynr'].iloc[-1]
#
#             target_value = max(((cwso_value - cwso_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index),
#                                (0.3 * (cwdm_value - cwdm_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index))
#
#             results.append((date_t, ir, cwdm_value, cwso_value, target_value))
#
#         results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value'])
#         max_target_row = results_df.loc[results_df['target_value'].idxmax()]
#
#         ir_date2.append(max_target_row['date_t'])
#         ir_depth2.append(max_target_row['ir'])
#
#         results_all_dates.append(max_target_row)
#
#         if first_date:
#             max_target_row.to_frame().T.to_csv('day_scheduled.csv', index=False)
#             first_date = False
#         else:
#             max_target_row.to_frame().T.to_csv('day_scheduled.csv', mode='a', header=False, index=False)
#
#         current_dvs = float(df['DVS'].iloc[-1])
#         if current_dvs >= 1.99:
#             break

# elif real_irrigation == 0 and calibrated == 1:
#     for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#         real_ir_update.modify_irrigation_crp(crp_file_path, 1)
#         real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)
#         real_ir_update.modify_irrigation_swp(swp_file_path, 0)
#
#     ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq1, start_year1, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)
#
#     ir_date = []
#     ir_depth = []
#
#     with open('result_forec.irg', 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             if 'mais            ,' in line:
#                 values = line.strip().split(',')
#                 date_str = values[1].strip()
#                 irrigation = float(values[4].strip()) * 10
#
#                 date_obj = datetime.strptime(date_str, '%Y-%m-%d')
#                 formatted_date = date_obj.strftime('%d-%b-%Y')
#                 doy = date_obj.timetuple().tm_yday
#                 if doy <= start_day_seq2:
#                     ir_date.append(formatted_date)
#                     ir_depth.append(irrigation)
#
#     ir_date2 = ir_date.copy()
#     ir_depth2 = ir_depth.copy()
#
#     start_date = datetime(start_year2, start_month2, start_day2)
#     dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)]
#
#     first_date = True
#     results_all_dates = []
#
#     for current_date in dates_to_evaluate:
#         results = []
#         date_t = current_date.strftime('%d-%b-%Y')
#         doy = current_date.timetuple().tm_yday
#
#         for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#             real_ir_update.modify_irrigation_crp(crp_file_path, 0)
#             real_ir_update.modify_irrigation_swp(swp_file_path, 1)
#
#         for ir in [0, 10, 15, 20, 25, 30, 40, 60]:
#             temp_ir_date2 = ir_date2.copy()
#             temp_ir_depth2 = ir_depth2.copy()
#             temp_ir_date2.append(date_t)
#             temp_ir_depth2.append(ir)
#
#             for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#                 real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)
#
#             ForecastStep.run_sub1(start_day_seq1, start_year1, doy + 4, start_year2, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)
#
#             # 读取并检查数据
#             df = read_data('result_forec.crp', ['Daynr', 'CWDM', 'CWSO'])
#
#             df = df.dropna(subset=['Daynr', 'CWDM', 'CWSO'])
#             df = to_numeric(df, ['CWDM', 'CWSO', 'Daynr'])
#
#             cwdm_value = df['CWDM'].iloc[-1]
#             cwso_value = df['CWSO'].iloc[-1]
#             cwdm_value_0 = df['CWDM'].iloc[-4]
#             cwso_value_0 = df['CWSO'].iloc[-4]
#             doy_max = df['Daynr'].iloc[-1]
#
#             target_value = max(((cwso_value - cwso_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index),
#                                (0.3 * (cwdm_value - cwdm_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index))
#
#             results.append((date_t, ir, cwdm_value, cwso_value, target_value))
#
#         results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value'])
#         max_target_row = results_df.loc[results_df['target_value'].idxmax()]
#
#         ir_date2.append(max_target_row['date_t'])
#         ir_depth2.append(max_target_row['ir'])
#
#         results_all_dates.append(max_target_row)
#
#         if first_date:
#             max_target_row.to_frame().T.to_csv('day_scheduled.csv', index=False)
#             first_date = False
#         else:
#             max_target_row.to_frame().T.to_csv('day_scheduled.csv', mode='a', header=False, index=False)
#
#         current_dvs = float(df['DVS'].iloc[-1])
#         if current_dvs >= 1.99:
#             break


# if calibrated == 1:
#     temp_ir_date2 = ir_date2.copy()
#     temp_ir_depth2 = ir_depth2.copy()
#
#     for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#         real_ir_update.modify_irrigation_crp(crp_file_path, 0)
#         real_ir_update.modify_irrigation_swp(swp_file_path, 1)
#
#         real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)
#
#     ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq2, start_year2, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)













#
#
#
#
#
#
#
#
#
#
#
#
# from datetime import datetime, timedelta
# # 初始日期
# start_date = datetime(start_year2, start_month2, start_day2)
#
# if (real_irrigation == 1) & (calibrated == 1):
#     # 初始化ir_date2和ir_depth2
#     ir_date2 = ir_date.copy()
#     ir_depth2 = ir_depth.copy()
#
#     # 获取200天内的所有日期
#     dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)]
#
#     first_date = True
#
#     for current_date in dates_to_evaluate:
#         results = []
#         date_t = current_date.strftime('%d-%b-%Y')
#         doy = current_date.timetuple().tm_yday
#
#         for ir in [0, 10, 15, 20, 25, 30, 40, 60]:
#             temp_ir_date2 = ir_date2.copy()
#             temp_ir_depth2 = ir_depth2.copy()
#             temp_ir_date2.append(date_t)
#             temp_ir_depth2.append(ir)
#
#             for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#                 real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)
#
#             ForecastStep.run_sub1(start_day_seq1, start_year1,
#                                   doy+4, #start_day_seq2 + 4,
#                                   start_year2, start_day_seq2, start_year2,
#                                   'gmaized.crp', 'swap.swp', divide=0)  # Swap1.swp to swap.swp & 使用实际日期
#
#
#             # 定义列名
#             columns = ['Date', 'Daynr', 'Daycrp', 'DVS', 'TSUM', 'LAIpot', 'LAI', 'Height', 'CrpFac', 'RootdPot',
#                        'Rootd',
#                        'PWLV', 'WLV', 'PWST', 'WST', 'PWRT', 'WRT', 'CPWDM', 'CWDM', 'CPWSO', 'CWSO', 'PGRASSDM',
#                        'GRASSDM',
#                        'PMOWDM', 'MOWDM', 'PGRAZDM', 'GRAZDM', 'DWLVCROP', 'DWLVSOIL', 'DWST', 'DWRT', 'DWSO',
#                        'HarLosOrm']
#
#             # 读取数据文件
#             data = []
#             with open('result_forec.crp', 'r') as file:
#                 lines = file.readlines()
#                 for line in lines:
#                     # 忽略前几行的元数据，只处理数据行
#                     if 'Date' not in line:
#                         values = line.strip().split(',')
#                         if len(values) == len(columns):  # 确保每行有正确数量的字段
#                             data.append(values)
#                         else:
#                             print(f"Skipping malformed line: {line}")
#
#             # 打印读取的数据以调试
#             print("Data preview (first 5 rows):", data[:5])
#
#             # 转换为DataFrame
#             try:
#                 df = pd.DataFrame(data, columns=columns)
#             except ValueError as e:
#                 print("Error creating DataFrame:", e)
#                 for row in data[:5]:  # 打印前5行用于调试
#                     print(row)
#                 exit(1)
#
#             # 去除包含非数值字符和空值的行
#             df = df.dropna(subset=['Daynr', 'CWDM', 'CWSO'])
#             df = df[df['CWDM'].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
#             df = df[df['CWSO'].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
#             df = df[df['Daynr'].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
#
#             # 转换为数值类型
#             df['CWDM'] = pd.to_numeric(df['CWDM'])
#             df['CWSO'] = pd.to_numeric(df['CWSO'])
#             df['Daynr'] = pd.to_numeric(df['Daynr'])
#
#             cwdm_value = df['CWDM'].iloc[-1]
#             cwso_value = df['CWSO'].iloc[-1]
#
#             cwdm_value_0 = df['CWDM'].iloc[-4]
#             cwso_value_0 = df['CWSO'].iloc[-4]
#
#             doy_max = df['Daynr'].iloc[-1]
#
#             print("cwdm_value:", cwdm_value)
#             print("cwso_value:", cwso_value)
#             print("cwdm_value_0:", cwdm_value_0)
#             print("cwso_value_0:", cwso_value_0)
#             print("doy_max:", doy_max)
#
#
#             target_value = max(((cwso_value - cwso_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index),
#                                (0.3 * (cwdm_value - cwdm_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index))
#
#             results.append((date_t, ir, cwdm_value, cwso_value, target_value))
#
#         # 找出 results 中 target_value 最大的行
#         results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value'])
#         max_target_row = results_df.loc[results_df['target_value'].idxmax()]
#
#         # 更新ir_date2和ir_depth2
#         ir_date2.append(max_target_row['date_t'])
#         ir_depth2.append(max_target_row['ir'])
#
#
#         # 保存到相应命名的CSV文件中
#         csv_file_path = 'day_scheduled.csv'
#         if first_date:
#             max_target_row.to_frame().T.to_csv(csv_file_path, index=False)
#             first_date = False
#         else:
#             max_target_row.to_frame().T.to_csv(csv_file_path, mode='a', header=False, index=False)
#
#         print(f'Saved the row with the highest target_value to {csv_file_path}')
#
#         # 检查 DVS 是否达到2.0
#         current_dvs = float(df['DVS'].iloc[-1])
#         if current_dvs >= 1.99:
#             print("DVS euqals 2.0")
#             break
#
# elif (real_irrigation == 0) & (calibrated == 1):
#     for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#         real_ir_update.modify_irrigation_crp(crp_file_path, 1)
#         real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)
#         real_ir_update.modify_irrigation_swp(swp_file_path, 0)
#
#     ForecastStep.run_sub1(start_day_seq1, start_year1,
#                           end_day_seq1,
#                           start_year1, start_day_seq2, start_year2,
#                           'gmaized.crp', 'swap.swp', divide=0)  # Swap1.swp to swap.swp & 使用实际日期
#     # 初始化列表
#     ir_date = []
#     ir_depth = []
#
#     # 读取数据文件
#     with open('result_forec.irg', 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             # 忽略前几行的元数据，只处理数据行
#             if 'mais            ,' in line:
#                 values = line.strip().split(',')
#                 date_str = values[1].strip()
#                 irrigation = float(values[4].strip())*10
#
#                 # 转换日期格式
#                 date_obj = datetime.strptime(date_str, '%Y-%m-%d')
#                 formatted_date = date_obj.strftime('%d-%b-%Y')
#                 doy = date_obj.timetuple().tm_yday
#                 if doy <= start_day_seq2:
#                     ir_date.append(formatted_date)
#                     ir_depth.append(irrigation)
#
#     ir_date2 = ir_date.copy()
#     ir_depth2 = ir_depth.copy()
#
#     # 获取200天内的所有日期
#     start_date = datetime(start_year2, start_month2, start_day2)
#     dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)]
#
#     first_date = True
#
#     for current_date in dates_to_evaluate:
#         results = []
#         date_t = current_date.strftime('%d-%b-%Y')
#         doy = current_date.timetuple().tm_yday
#
#         for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#             real_ir_update.modify_irrigation_crp(crp_file_path, 0)
#             #real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)
#             real_ir_update.modify_irrigation_swp(swp_file_path, 1)
#
#         for ir in [0, 10, 15, 20, 25, 30, 40, 60]:
#             temp_ir_date2 = ir_date2.copy()
#             temp_ir_depth2 = ir_depth2.copy()
#             temp_ir_date2.append(date_t)
#             temp_ir_depth2.append(ir)
#
#             for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#                 real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)
#
#             ForecastStep.run_sub1(start_day_seq1, start_year1,
#                                   doy + 4,  # 用 DOY 代替 start_day_seq2 + 4
#                                   start_year2, start_day_seq2, start_year2,
#                                   'gmaized.crp', 'swap.swp', divide=0)  # Swap1.swp to swap.swp & 使用实际日期
#
#             # 定义列名
#             columns = ['Date', 'Daynr', 'Daycrp', 'DVS', 'TSUM', 'LAIpot', 'LAI', 'Height', 'CrpFac', 'RootdPot',
#                        'Rootd',
#                        'PWLV', 'WLV', 'PWST', 'WST', 'PWRT', 'WRT', 'CPWDM', 'CWDM', 'CPWSO', 'CWSO', 'PGRASSDM',
#                        'GRASSDM',
#                        'PMOWDM', 'MOWDM', 'PGRAZDM', 'GRAZDM', 'DWLVCROP', 'DWLVSOIL', 'DWST', 'DWRT', 'DWSO',
#                        'HarLosOrm']
#
#             # 读取数据文件
#             data = []
#             with open('result_forec.crp', 'r') as file:
#                 lines = file.readlines()
#                 for line in lines:
#                     # 忽略前几行的元数据，只处理数据行
#                     if 'Date' not in line:
#                         values = line.strip().split(',')
#                         if len(values) == len(columns):  # 确保每行有正确数量的字段
#                             data.append(values)
#                         else:
#                             print(f"Skipping malformed line: {line}")
#
#             # 打印读取的数据以调试
#             print("Data preview (first 5 rows):", data[:5])
#
#             # 转换为DataFrame
#             try:
#                 df = pd.DataFrame(data, columns=columns)
#             except ValueError as e:
#                 print("Error creating DataFrame:", e)
#                 for row in data[:5]:  # 打印前5行用于调试
#                     print(row)
#                 exit(1)
#
#             # 去除包含非数值字符和空值的行
#             df = df.dropna(subset=['Daynr', 'CWDM', 'CWSO'])
#             df = df[df['CWDM'].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
#             df = df[df['CWSO'].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
#             df = df[df['Daynr'].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
#
#             # 转换为数值类型
#             df['CWDM'] = pd.to_numeric(df['CWDM'])
#             df['CWSO'] = pd.to_numeric(df['CWSO'])
#             df['Daynr'] = pd.to_numeric(df['Daynr'])
#
#             cwdm_value = df['CWDM'].iloc[-1]
#             cwso_value = df['CWSO'].iloc[-1]
#
#             cwdm_value_0 = df['CWDM'].iloc[-4]
#             cwso_value_0 = df['CWSO'].iloc[-4]
#
#             doy_max = df['Daynr'].iloc[-1]
#
#             print("cwdm_value:", cwdm_value)
#             print("cwso_value:", cwso_value)
#             print("cwdm_value_0:", cwdm_value_0)
#             print("cwso_value_0:", cwso_value_0)
#             print("doy_max:", doy_max)
#
#             target_value = max(((cwso_value - cwso_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index),
#                                (0.3 * (cwdm_value - cwdm_value_0) * yield_price_per_ha_per_mm - ir * water_cost_per_ha_per_mm * weight_index))
#
#
#             results.append((date_t, ir, cwdm_value, cwso_value, target_value))
#
#         # 找出 results 中 target_value 最大的行
#         results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value'])
#         max_target_row = results_df.loc[results_df['target_value'].idxmax()]
#
#         # 更新ir_date2和ir_depth2
#         ir_date2.append(max_target_row['date_t'])
#         ir_depth2.append(max_target_row['ir'])
#
#         # 保存到相应命名的CSV文件中
#         csv_file_path = 'day_scheduled.csv'
#         if first_date:
#             max_target_row.to_frame().T.to_csv(csv_file_path, index=False)
#             first_date = False
#         else:
#             max_target_row.to_frame().T.to_csv(csv_file_path, mode='a', header=False, index=False)
#
#         print(f'Saved the row with the highest target_value to {csv_file_path}')
#
#         # 检查 DVS 是否达到2.0
#         current_dvs = float(df['DVS'].iloc[-1])
#         if current_dvs >= 1.99:
#             print("DVS equals 2.0, stopping simulation.")
#             break
#
# if calibrated == 1:
#     temp_ir_date2 = ir_date2.copy()
#     temp_ir_depth2 = ir_depth2.copy()
#
#     for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
#         real_ir_update.modify_irrigation_crp(crp_file_path, 0)
#         real_ir_update.modify_irrigation_swp(swp_file_path, 1)
#
#         real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)
#
#     ForecastStep.run_sub1(start_day_seq1, start_year1,
#                           # doy_max, #doy_max read max doy to the end . # doy + 4,  # 用 DOY 代替 start_day_seq2 + 4
#                           end_day_seq2,
#                           start_year2, start_day_seq2, start_year2,
#                           'gmaized.crp', 'swap.swp', divide=0)






# ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq2, end_year2,start_day_seq2, start_year2,
#                       'gmaized.crp','swap.swp',divide =0) #Swap1.swp to swap.swp & 使用实际日期
#
# real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)






# Step 2: 修改 gmaized.crp 文件 #改为scheduled irrigation
# modify_irrigation_crp(crp_file_path, 1)
# modify_irrigation_swp(swp_file_path, 0)
# modify_initsm_swp(swp_file_path, 3)

# Step 3: 重新启动或继续运行模型 (不用了 合并到run_sub1了)
#ForecastStep.run_sub2(start_day_seq2, start_year2, end_day_seq2, end_year2, start_day_seq1,'gmaized.crp','swap.swp','result_p1.end')


# with open("Swap "+Field+".swp") as f:
#     with open("Swap1.swp","w") as f1:
#         line=f.readline()
#         while line!='':
#             f1.write(line)
#             line=f.readline()
# with open("Swap1.swp") as f:
#     with open("SwapOriginal.swp","w") as f1:
#         line=f.readline()
#         while line!='':
#             f1.write(line)
#             line=f.readline()
# with open("SwapOriginal.swp") as f:
#     with open("Swap1.swp","w") as f1:
#         line=f.readline()
#         f1.write(line)
#         while line[0:17]!='* End of the main':
#             line=f.readline()
#             f1.write(line)

#
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from datetime import datetime
#
# start_day_seq1 =datetime(start_year1, start_month1, start_day1).timetuple()[7]
# end_day_seq1 =datetime(end_year1, end_month1, end_day1).timetuple()[7]
#
# start_day_seq2 =datetime(start_year2, start_month2, start_day2).timetuple()[7]
# end_day_seq2 =datetime(end_year2, end_month2, end_day2).timetuple()[7]
#
# seq_s2s_start = datetime.strptime('20240416', '%Y%m%d').timetuple().tm_yday + 2
#
# file_path = 'result_forec.inc'
# # 读取文件，跳过前6行
# df = pd.read_csv(file_path, skiprows=6)
# df.columns = df.columns.str.strip()
#
# # 绘制柱状图
# fig, ax = plt.subplots(figsize=(14, 4))  # 设置图形的尺寸
# ax.bar(df['Day'], df['Rain'], width=0.8, label='Prec (cm/day)', alpha=0.6)  # 画出Rain数据的柱状图
# ax.bar(df['Day'], df['Irrig'], width=0.8, label='Irrig (cm/day)', alpha=0.95)  # 画出Irrig数据的柱状图
#
# day_line = start_day_seq2
# # 在图上标出红色竖线并标注
# ax.axvline(day_line, color='red', linestyle='--', lw=2)  # 绘制红色竖线
# ax.text(60, ax.get_ylim()[1]*0.9, 'actual weather', color='black', horizontalalignment='left')  # 在线条右侧添加文字
# ax.text(day_line +30, ax.get_ylim()[1]*0.9, 'S2S forecast', color='black', horizontalalignment='left')  # 在线条右侧添加文字
# ax.text(day_line, ax.get_ylim()[1]*0.9, 'current date', color='black', horizontalalignment='center')  # 在线条右侧添加文字
# ax.text(200 , ax.get_ylim()[1]*0.8, 'total precipitation:' + str(round(df.Rain.sum(),1)) + 'cm', color='blue', horizontalalignment='left')
# ax.text(200 , ax.get_ylim()[1]*0.7, 'total future precipitation:' + str(round(df[df.Day>day_line].Rain.sum(),1)) + 'cm', color='blue', horizontalalignment='left')
# ax.text(200 , ax.get_ylim()[1]*0.6, 'total future irrigation demand:' + str(round(df[df.Day>day_line].Irrig.sum(),1)) + 'cm', color='brown', horizontalalignment='left')
#
# # 设置x轴日期格式
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
# plt.xticks(rotation=90)  # 旋转x轴标签，避免重叠
#
# # 添加图表元素
# plt.title('Precipitation and Irrigation Over Time - Forecast')  # 标题
# plt.xlabel('DOY')  # X轴标签
# plt.ylabel('cm/day')  # Y轴标签
# plt.legend()  # 添加图例
# plt.savefig('../output/rain_future_irrig.png',bbox_inches='tight')
#
#
#
# file_path = 'result_forec.crp'
#
# # 读取文件，跳过前6行
# df = pd.read_csv(file_path, skiprows=7)
# df.columns = df.columns.str.strip()
#
# fig, ax1 = plt.subplots(figsize=(14, 4))
#
# # 绘制CWDM数据与左侧Y轴
# ax1.bar(df['Daynr'], df['CWDM'], width=1, label='CWDM_simulated', alpha=0.3, color='blue')
# ax1.set_xlabel('DOY')
# ax1.set_ylabel('CWDM (kg/ha)', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
#
# day_line = start_day_seq2
# # 在图上标出红色竖线并标注
# ax1.axvline(day_line, color='red', linestyle='--', lw=2)  # 绘制红色竖线
# ax1.text(day_line - 30, ax1.get_ylim()[1]*0.7, 'actual weather', color='black', horizontalalignment='center')  # 在线条右侧添加文字
# ax1.text(day_line+30, ax1.get_ylim()[1]*0.7, 'S2S forecast', color='black', horizontalalignment='center')  # 在线条右侧添加文字
# ax1.text(day_line, ax1.get_ylim()[1]*0.9, 'current date', color='black', horizontalalignment='center')  # 在线条右侧添加文字
#
# # 创建与ax1共享X轴的第二个Y轴
# ax2 = ax1.twinx()
# ax2.plot(df['Daynr'].to_numpy().flatten(), df['LAI'].to_numpy().flatten(), label='LAI_simulated', color='darkgreen', marker='', linestyle='-')
# #ax2.plot(modlai['Daynr'][92:242].to_numpy().flatten(), modlai['LAI_MODIS'][92:242].to_numpy().flatten(), label='LAI_MODIS', color='orange', marker='', linestyle='-')
# #ax2.plot(modlai['Daynr'].to_numpy().flatten(),modlai['LAI_MODIS'].to_numpy().flatten(), label='LAI_MODIS_pkl', color='darkgreen', marker='', linestyle='-')
# ax2.tick_params(axis='y', labelcolor='darkgreen')
# ax2.set_ylabel('LAI', color='darkgreen')
#
# # 设置x轴显示间隔为7
# #ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', nbins=7))
# ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
#
#
# # 添加图例
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines + lines2, labels + labels2, loc='upper left')
# plt.title('LAI and Cumulative dry matter (CWDM) - Forecast')
# plt.xticks(rotation=90)
# # 可选：保存图像
# plt.savefig('../output/LAI_and_CWDM.png',bbox_inches='tight')
#


soil_initial_pars=np.array([])
with open("Swap "+Field+".swp") as f:
    line=f.readline()
    while line[:17]!='  ISOILLAY1  ORES':
        line=f.readline()
    line=f.readline()
    while line[:5]!='* ---':
        temp=np.fromstring(line,dtype=float,sep='    ')
        soil_initial_pars=np.append(soil_initial_pars,temp[1:6])
        line=f.readline()

crop_initial_pars=Read.cropars('initial')


root_zone_depth=60
threshold=0.5
alternative_irrigation=0.5

#assimilation+optmization (should be modified by multi-irrigation-scenario ensemble forecast)

# irrigation_schedule=main.run(start_month,start_day,start_year,end_month,end_day,end_year,lai_m_std,vwc_m_std,depths_vwc,\
#         yield_price,water_cost,soil_initial_pars,crop_initial_pars,root_zone_depth,threshold,alternative_irrigation,weather,\
#             planting_day_seq,harvest_day_seq)


end_time = datetime.now()
print(end_time-start_time)

os.chdir('..')
import analysis
#analysis.run_analysis(crop_type)
print("finished!!!")