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

Field='N Koinzan' # Name of the location of the previous site scale study (not used in the simulation actually)

from datetime import datetime
start_time = datetime.now()

longitude = -112.265
#location input from GIS-based Position-Acquisition Function from DAWN
latitude = 41.735
#location input from GIS-based Position-Acquisition Function from DAWN

calibration= 0
#is calibration included? 1: needed (first time to use the model at a new site);  0: no need (already calibrated, and updated crop parameters will be used)
calibrated = abs(calibration-1)
#flag of if the calibration is finished

real_irrigation= 0
#is real irrigation data (user-recorded irrigation events by today in this year) available (defined by users)?

if real_irrigation== 1:
    # the user has the irrigation dates and depths that already applied by today
    ir_date = ['25-May-2024', '29-May-2024', '07-Jul-2024']
    # real irrigation date input by users
    ir_depth = [10.0, 10.0, 10.0]  # real irrigation depth (mm) input by users

water_cost_per_ha_per_mm = 2.0
# water price, 2.0 USD/ha*mm (that is 2.0 USD/10 m^3 or 0.2 USD/m^3)

yield_price_per_kg = 0.20
# corn price per kg

cropland_area_current_ha = 10.0
# How much maize field area does the user have on the day he uses this model (in a similar area).

water_permit_left = 3000.0
# How much water permit is left for the user on the day they use the model. unit: m^3

weight_index = 0.7
# used to adjust risk the tolerance/irrigation preference and model uncertainty (it is not user input)

import Maize.Extract_tif

#calibration: find previous corn-planted year and use data of the year (using historical year's data)
dir_croptype = './data/CropAT_US/CropType'
#path of annual crop type map data
corn_year = Maize.Extract_tif.extract_crop_type_years(longitude, latitude, dir_croptype)
#find the last corn-planted year in 2015-2019
print('corn_year  '+ str(corn_year))

crop_type='Maize'
start_month = 4
# month of the start date of the calibration year
start_day = 1
# day of the start date of the calibration year

if corn_year is not None and corn_year != '' and corn_year != [] and corn_year != {}:
    #if corn-planted year is found, then use the year's data. if not, use the 2019 data
    start_year = corn_year
else:
    start_year = 2019

end_month = 9
# month of the end date of the calibration year
end_day = 10
# day of the end date of the calibration year
end_year = start_year
#here calibration is based on single year data , thus start year equals end year

# this year: real-time model application (using this year's data)
start_month1 = 3
# input by users - sowing date, from 'planting date entering in the interface'
start_day1 = 1
# input by users - from 'planting date entering in the interface'
start_year1 = 2024
# input by users - from 'planting date entering in the interface'

end_month1 = 8
# The date the user used the model. Usually defined as ‘today’, it can be obtained from system time.
end_day1 = 9
# The date the user used the model. Usually defined as ‘today’, it can be obtained from system time.
end_year1 = 2024
# The date the user used the model. Usually defined as ‘today’, it can be obtained from system time.

start_month2 = 8
# The first date on which the model starts irrigation scheduling. Typically defined as ‘tomorrow’ (the day after end_day1).
start_day2 = 10
# The first date on which the model starts irrigation scheduling. Typically defined as ‘tomorrow’ (the day after end_day1).
start_year2 = start_year1

end_year2 = start_year1
end_day2 = 1
# Which date does the model simulation end?
if start_month2 < 6:
    # The model simulation must end by the 1st of the month six months after the current date or by the 1st of December.
    end_month2 = start_month2 + 7
else:
    end_month2 = 12
    end_day2 = 1


def list_folders_as_dates(folder_path):
    # Function to get the file names in a folder
    folder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    date_folders = []
    for name in folder_names:
        try:
            folder_date = datetime.strptime(name, '%Y%m%d')
            date_folders.append(folder_date)
        except ValueError:
            continue
    return date_folders

def find_closest_folder(folder_path, target_date_str):
    # A function that determines the closest previous climate forecast file to be used based on the current date.
    # The climate forecast for each date provides a forecast for approximately 6 months ahead and is updated every 5 days.
    target_date = datetime.strptime(target_date_str, '%Y%m%d')
    date_folders = list_folders_as_dates(folder_path)

    # Filter out all folders earlier than the target date
    previous_folders = [date for date in date_folders if date < target_date]

    if not previous_folders:
        return None
        # Returns None if there are no folders earlier than the target date

    # Find the folder closest to the given date
    closest_folder = max(previous_folders)
    closest_date_diff = (target_date - closest_folder).days

    # If the difference between the nearest folder date and the target date is less than or equal to 4 days, try to find the second nearest date
    if closest_date_diff <= 4:
        previous_folders.remove(closest_folder)
        if previous_folders:
            second_closest_folder = max(previous_folders)
            # Find seasonal forecast data for the second nearest date
            return second_closest_folder.strftime('%Y%m%d')
        else:
            return None
            # If there is no second nearest date, return None

    return closest_folder.strftime('%Y%m%d')
    # Returns the date of the climate forecast data file used

target_date_str = datetime(start_year2,start_month2, start_day2).strftime('%Y%m%d')
# First date to start irrigation scheduling (usually tomorrow)
file_number = find_closest_folder('../../../PUB/S2S/V2023-07/Operational', target_date_str)
# Find out which day's climate forecast data to use
print("most recent s2s forecast nc folder", file_number)

# Save the calibration date to .pkl files
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

# define initial crop parameters: RGRLAI, TBASE, EFF, CVL
crop_initial_pars = [0.024, 11.35, 0.5, 0.808]

#lai_m_std=0.05

planting_day_seq=datetime(start_year, start_month, start_day).timetuple()[7]
# Get the planting date of the calibration year
harvest_day_seq=datetime(end_year, end_month, end_day).timetuple()[7]
# Get the harvest date of the calibration year


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

# Get the start and end dates of the calibration
start_month_cal=start_month
start_day_cal=start_day
start_year_cal=start_year
end_month_cal=end_month
end_day_cal=end_day
end_year_cal=end_year

year = start_year_cal

# Various paths for driving data
polaris_path = '../data/polaris'
dtw_path = '../data/dtw'
dem_path = '../data/dem'
tiledrain_path = '../data/tiledrain'
lai_path = '../data/lai_' + str(year)
era5_path = '../data/era5_' + str(year)

if calibration == 1:
    #Process for model calibration
    # Use of historical ERA5-land meteorological data as climate forcing
    # The goal of the calibration is to minimise the inconsistency between the simulated LAI and the MODIS LAI by parameter tuning with genetic algorithms

    # Obtain MODIS LAI time series data for historical years based on site location and save it to .csv file and .pkl file
    lai_modis = Extract_tif.extract_and_interpolate_lai(longitude, latitude, lai_path)['Value'].to_list()
    pd.DataFrame(lai_modis, columns=['lai_modis']).to_csv('lai_modis.csv')
    with open('lai_measurements_modis.pkl', "wb") as f:
        pickle.dump(lai_modis, f)

    # Extract ERA5-land climate data of the historical year for calibration
    # Air temperature, maximum temperature, minimum temperature, dew point temperature, downward radiation, precipitation, potential evaporation, wind speed, etc.
    ta = Extract_tif.extract_era_temperature_2m(longitude, latitude, os.path.join(era5_path ,'temperature_2m') , year)['Value'].to_list()
    tmin = Extract_tif.extract_era_temperature_2m_min(longitude, latitude, os.path.join(era5_path ,'temperature_2m_min') , year)['Value'].to_list()
    tmax = Extract_tif.extract_era_temperature_2m_max(longitude, latitude, os.path.join(era5_path ,'temperature_2m_max') , year)['Value'].to_list()
    tdew = Extract_tif.extract_era_dewpoint_temperature_2m(longitude, latitude, os.path.join(era5_path ,'dewpoint_temperature_2m'), year)['Value'].to_list()
    rad = Extract_tif.extract_era_surface_solar_radiation_downwards_sum(longitude, latitude, os.path.join(era5_path ,'surface_solar_radiation_downwards_sum'), year)['Value'].to_list()
    prec = Extract_tif.extract_era_total_precipitation_sum(longitude, latitude, os.path.join(era5_path ,'total_precipitation_sum'), year)['Value'].to_list()
    pet = Extract_tif.extract_era_potential_evaporation_sum(longitude, latitude, os.path.join(era5_path ,'potential_evaporation_sum'), year)['Value'].to_list()
    wind_u = Extract_tif.extract_era_u_component_of_wind_10m(longitude, latitude, os.path.join(era5_path ,'u_component_of_wind_10m'), year)['Value'].to_list()
    wind_v = Extract_tif.extract_era_v_component_of_wind_10m(longitude, latitude, os.path.join(era5_path ,'v_component_of_wind_10m'), year)['Value'].to_list()
    humd = [Extract_tif.calculate_relative_humidity(ta_d, tdew_d) for ta_d, tdew_d in zip(ta, tdew)]

    # Conversion of extracted ERA5-land meteorological data into the format required for SWAP model inputs
    # Year	DOY	Solar	T-max	T-min	RelHum	Precip	ET
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

    df_weather = pd.read_excel('../data/weather_era.xlsx')

    # weather data processing and conversion
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
    # save the processed ERA5-land data

    weather=weather.values.tolist()
    temp=[(start_year_cal),(end_year_cal)]

    for i in range(len(weather)):
        # Dealing with possible outliers
        weather[i][7]=max(0.0,weather[i][7])
        weather[i][8]=max(0.0,weather[i][8])
        weather[i][9]=max(0.0,weather[i][9])
        weather[i][10] = max(0.0, weather[i][10])
        weather[i][5]=min(weather[i][5],weather[i][6]*0.95)

    #change weather data
    for i in range(end_year_cal-start_year_cal+1):
        # Use of processed ERA5-land data as meteorological input for the calibration year
        ChangeWeather.change_weather(start_day_seq=1, Year=temp[i], data=weather)
        # Generate meteorological data files that SWAP can read and use


#Extract the site's multilayer soil hydraulic parameters
polaris_data = Extract_tif.extract_polaris(longitude, latitude, polaris_path)
df_polaris = Extract_tif.process_polaris_data(polaris_data)
df_polaris.to_csv('./df_polaris_soil_hydraulic.csv')
list_polaris =df_polaris.values.tolist()
ChangeSwap.change_soilhydraulic(list_polaris) #input the soil hydraulic parameters into the .swp file to run the model

crp_file_path = 'gmaized.crp'
# The .crp input file used for the swap model run. Mainly related to crop growth.
swp_file_path = 'Swap1.swp'
# The .swp input file used for the swap model run. Contains various settings for the swap model.
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

#Extract and update elevation, ground water table, and tile drains data
demvalue = Extract_tif.extract_tif_single_value(longitude, latitude, dem_path)
dtwvalue = Extract_tif.extract_tif_single_value(longitude, latitude, dtw_path)
tiledrainvalue = Extract_tif.extract_tif_single_value(longitude, latitude, tiledrain_path)


def modify_RET_lat(lat_value, dem_value, meteo_etref):
    # Adjustment of the calculation of reference evapotranspiration in the swap model
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
# Reference evapotranspiration calculated by the SWAP model itself

def modify_dtw_tiledrain(dtw_value,tiledrain_value):
    # Update the extracted groundwater level and the tile drain information
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
# Update the extracted groundwater level and the tile drain information

crop_initial_pars =  [0.024, 11.35, 0.5, 0.808]
# default crop parameters, Starting point for parameter calibration

if calibration == 1:
    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
        real_ir_update.modify_irrigation_swp(swp_file_path, 0)

    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 1)
        real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)

    # If calibration is required, call the parameter calibration function to calibrate the parameter with historical year data.
    _, crop_initial_pars_cali, best_value = Calibration.calibration(crop_initial_pars,start_month_cal, start_day_cal, start_year_cal,end_month_cal, end_day_cal, end_year_cal)
    pd.DataFrame(crop_initial_pars_cali, columns=["Parameters"]).to_csv('crop_initial_pars_cali.csv')
    # save the calibrated parameters
    pd.DataFrame([[best_value]], columns=["cali_best_obj_value"]).to_csv('cali_best_obj_value.csv')
    #pd.DataFrame([[best_value]], columns=["cali_best_obj_value"]).to_csv('cali_best_obj_value.csv')
    # save the objective function value (1 - KGE)
    calibrated = 1
    # Setting the ‘calibrated’ to 1 indicates that the model has been calibrated at this time

from datetime import datetime

# Updated calibrated crop parameters in gmaized.crp file
if (calibration == 1) & (pd.read_csv('cali_best_obj_value.csv')['cali_best_obj_value'].to_list()[0] < 0.6):
    #If using the model for the first time, after calibration, update the parameters
    try:
        crop_initial_pars_cali = pd.read_csv('crop_initial_pars_cali.csv')['Parameters'].to_list()
    except ValueError as e:
        print("Error crop para:", e)
elif (calibrated == 1) & (pd.read_csv('cali_best_obj_value.csv')['cali_best_obj_value'].to_list()[0] < 0.6):
    # The second and subsequent re-use of the model: directly using parameters in cali_best_obj_value.csv in thd ./Maize folder.
    # This requires interface development that takes into account the asynchronous independence between different sites of the user.
    # Otherwise, there may be mutual interference between different sites.
    try:
        crop_initial_pars_cali = pd.read_csv('crop_initial_pars_cali.csv')['Parameters'].to_list()
    except ValueError as e:
        print("Error crop para:", e)
else:
    crop_initial_pars_cali = crop_initial_pars

# update crop parameters from the gmaized.crp file (using the calibrated parameters)
for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
    use_crp_cali_pars(crp_file_path, [0.024, 11.35, 0.5, 0.808])

# Convert various dates to day of the year (DOY)
# including: date of start of simulation, date of end of simulation, current date - today, date of start of irrigation scheduling - tomorrow
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
# Date (DOY) of the first data record in the seasonal climate forecast
print('s2s start day')
print(seq_s2s_start)

modify_RET_lat(latitude, demvalue,0 )
modify_dtw_tiledrain(dtwvalue,tiledrainvalue)

# From the first day of the year to the current date, actual weather data from gridmet (a real-time climate dataset) is used
# and after that, s2s's seasonal climate forecast is used.
use_s2s.gridmet_processing(start_year1,1, end_day_seq1, file_number, latitude, longitude, start_day_seq2)
use_s2s.s2s_processing(start_year1, end_day_seq2, file_number, latitude, longitude)


import pandas as pd
from datetime import datetime, timedelta

start_date = datetime(start_year2, start_month2, start_day2)

# Columns in the result_forec.crp file used to read the provisional crop length and dry weight to calculate the objective function for irrigation scheduling
columns = ['Date', 'Daynr', 'Daycrp', 'DVS', 'TSUM', 'LAIpot', 'LAI', 'Height', 'CrpFac', 'RootdPot',
           'Rootd', 'PWLV', 'WLV', 'PWST', 'WST', 'PWRT', 'WRT', 'CPWDM', 'CWDM', 'CPWSO', 'CWSO',
           'PGRASSDM', 'GRASSDM', 'PMOWDM', 'MOWDM', 'PGRAZDM', 'GRAZDM', 'DWLVCROP', 'DWLVSOIL',
           'DWST', 'DWRT', 'DWSO', 'HarLosOrm']

def read_data(file_path, required_columns):
    # Function: read data from file and check column existence
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'Date' not in line:
                values = line.strip().split(',')
                if len(values) == len(columns):
                    data.append(values)
                else:
                    print(f"Skipping malformed line: {line}")
    df = pd.DataFrame(data, columns=columns)
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns in data: {missing_cols}")
    return df

def to_numeric(df, cols):
    #Function: Check and convert to numeric type
    for col in cols:
        df = df[df[col].str.strip().apply(lambda x: x.replace('.', '', 1).isdigit())]
        df[col] = pd.to_numeric(df[col])
    return df

# Module for irrigation optimisation. Distinguishes whether the user has entered previous irrigation information.
if real_irrigation == 1 and calibrated == 1: # If the user enters previously recorded irrigation information (how much water the user already irrigated)
    # Reads lists of irrigation depths and irrigation dates entered by the user
    ir_date2 = ir_date.copy()
    ir_depth2 = ir_depth.copy()

    dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)] # Irrigation scheduled every 4 days starting tomorrow

    first_date = True
    results_all_dates = []

    for current_date in dates_to_evaluate: # Iterate through all the dates for every 4 days starting from tomorrow in a 200-days period
        results = []
        date_t = current_date.strftime('%d-%b-%Y')
        doy = current_date.timetuple().tm_yday

        for ir in [0, 10, 15, 20, 25, 30, 40, 60]: # Iterate 8 irrigation depth options of a single day: 0, 10, 15, 20, 25, 30, 40, 60 mm/ha
            temp_ir_date2 = ir_date2.copy()
            temp_ir_depth2 = ir_depth2.copy()
            temp_ir_date2.append(date_t)
            temp_ir_depth2.append(ir)

            for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
                real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)

            # Simulation of conditions after 7 days under various irrigation rates. Read crop growth after 7 days, etc.
            ForecastStep.run_sub1(start_day_seq1, start_year1, doy + 7, start_year2, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)

            # DVS is development stage value (0.0-2.0), CWDM is total dry weight (kg/ha), CWSO (kg/ha) is dry weight of storage organ (equivalent to yield)
            df = read_data('result_forec.crp', ['Daynr', 'DVS', 'CWDM', 'CWSO'])
            df = df.dropna(subset=['Daynr', 'CWDM', 'CWSO', 'DVS'])
            df = to_numeric(df, ['CWDM', 'CWSO', 'Daynr', 'DVS'])

            cwdm_value = df['CWDM'].iloc[-1]
            cwso_value = df['CWSO'].iloc[-1]
            doy_max = df['Daynr'].iloc[-1]
            dvs_value = df['DVS'].iloc[-1]

            if ir == 0:
                target_value = 0
                cwdm_ir0 = cwdm_value
            else:
                # Irrigation Optimisation Objective: Maximise relative yield and profit increments while minimising the cost of irrigating water cost
                target_value = (cwdm_value - cwdm_ir0) * yield_price_per_kg - ir * water_cost_per_ha_per_mm * weight_index

            results.append((date_t, ir, cwdm_value, cwso_value, target_value))

        results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value'])
        max_target_row = results_df.loc[results_df['target_value'].idxmax()] # Find the irrigation depth option that maximises the objective function

        ir_date2.append(max_target_row['date_t'])
        ir_depth2.append(max_target_row['ir'])

        results_all_dates.append(max_target_row)

        # Save the optimal irrigation depth options for all these scheduled irrigation dates
        if first_date:
            max_target_row.to_frame().T.to_csv('day_scheduled.csv', index=False)
            first_date = False
        else:
            max_target_row.to_frame().T.to_csv('day_scheduled.csv', mode='a', header=False, index=False)

        # If the loop has reached the end of the growing season, end the iteration
        current_dvs = float(df['DVS'].iloc[-1])
        if (current_dvs >= 1.99) & (doy + 4 >= float(doy_max)):
            break

elif real_irrigation == 0 and calibrated == 1:
    # If the user did not enter previously recorded irrigation information (how much water the user already irrigated)
    # Set the irrigation scheme before the current date to SWAP's default weekly irrigation scheme,
    # and use our irrigation optimisation algorithm after the current date
    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 1)
        real_ir_update.modify_schedule_irrigation_date(crp_file_path, 1, 3)
    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:

        real_ir_update.modify_irrigation_swp(swp_file_path, 0)

    # Simulate once first to get the irrigation information estimated by SWAP before the current date
    ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq1, start_year1, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)

    ir_date = []
    ir_depth = []

    # Read the simulated SWAP estimated irrigation information up to the current date.
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
                if doy < start_day_seq2-1:
                    ir_date.append(formatted_date)
                    ir_depth.append(irrigation)

    ir_date2 = ir_date.copy()
    ir_depth2 = ir_depth.copy()

    start_date = datetime(start_year2, start_month2, start_day2)
    dates_to_evaluate = [start_date + timedelta(days=x) for x in range(0, 201, 4)]

    first_date = True
    results_all_dates = []

    for current_date in dates_to_evaluate: # Iterate through all the dates for every 4 days starting from tomorrow in a 200-days period (similar to above codes)
        results = []
        date_t = current_date.strftime('%d-%b-%Y')
        doy = current_date.timetuple().tm_yday

        for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
            real_ir_update.modify_irrigation_crp(crp_file_path, 0)

        for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
            real_ir_update.modify_irrigation_swp(swp_file_path, 1)

        for ir in [0, 10, 15, 20, 25, 30, 40, 60]:
            temp_ir_date2 = ir_date2.copy()
            temp_ir_depth2 = ir_depth2.copy()
            temp_ir_date2.append(date_t)
            temp_ir_depth2.append(ir)

            for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:
                real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)

            ForecastStep.run_sub1(start_day_seq1, start_year1, doy + 7, start_year2, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)

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
                target_value = (cwdm_value - cwdm_ir0) * yield_price_per_kg - ir * water_cost_per_ha_per_mm * weight_index

            results.append((date_t, ir, cwdm_value, cwso_value, target_value))

        results_df = pd.DataFrame(results, columns=['date_t', 'ir', 'cwdm_value', 'cwso_value', 'target_value'])
        max_target_row = results_df.loc[results_df['target_value'].idxmax()]

        ir_date2.append(max_target_row['date_t'])
        ir_depth2.append(max_target_row['ir'])

        results_all_dates.append(max_target_row)

        if first_date:
            max_target_row.to_frame().T.to_csv('day_scheduled.csv', index=False)
            first_date = False
        else:
            max_target_row.to_frame().T.to_csv('day_scheduled.csv', mode='a', header=False, index=False)

        current_dvs = float(df['DVS'].iloc[-1])
        if (current_dvs >= 1.99) & (doy + 4 >= float(doy_max)):
            break

# After the optimal irrigation depths for each of the 4 days are obtained,
# run the model again with these irrigation depths input to obtain information on crop growth, water balance, etc.
if calibrated == 1:
    temp_ir_date2 = ir_date2.copy()
    temp_ir_depth2 = ir_depth2.copy()

    for crp_file_path in ['GmaizeDOriginal.crp', 'gmaized.crp']:
        real_ir_update.modify_irrigation_crp(crp_file_path, 0)

    for swp_file_path in ['SwapOriginal.swp', 'Swap1.swp', 'swap.swp']:

        real_ir_update.modify_irrigation_swp(swp_file_path, 1)

        real_ir_update.update_swp_file(swp_file_path, temp_ir_date2, temp_ir_depth2)

    ForecastStep.run_sub1(start_day_seq1, start_year1, end_day_seq2, start_year2, start_day_seq2, start_year2, 'gmaized.crp', 'swap.swp', divide=0)


# The model simulation should finish here.




















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