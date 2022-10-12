import sys
sys.path.append("..")

import src.config

from src.datasets.loading import statcan, ookla

import numpy as np 
import pandas as pd
import geopandas as gp

from sklearn import preprocessing, pipeline, compose
from sklearn import linear_model, model_selection, svm
from sklearn import metrics

import matplotlib.pyplot as plt

import pickle

def make_next_year_next_quarter(year, quarter):
    
    if quarter == 4:
        next_quarter = 1
        next_year = year + 1
    else:
        next_quarter = quarter + 1
        next_year = year
        
    return next_year, next_quarter

def make_features_table(year, quarter):
    
    ookla_tiles = ookla.canada_tiles()
    ookla_tiles['quadkey'] = ookla_tiles['quadkey'].astype(int)
    da_pops = statcan.dissemination_areas_populations()
    o = gp.read_file(src.config.OVERLAYS_DIR / 'tile_das_overlay') #this can take a few minutes to load.
    tile_da_label = o.dropna(subset=['DAUID','quadkey']).sort_values(by=['quadkey','tile_frac'],ascending=False).drop_duplicates(subset='quadkey', keep='first')
    tile_da_label['quadkey'] = tile_da_label['quadkey'].astype(int)
    tile_da_label['DAUID'] = tile_da_label['DAUID'].astype(int)
    #last_4_quarters = ookla.speed_data(ookla.available_files().loc[('fixed',2021,3):('fixed',2022,2)].path)
    next_year, next_quarter = make_next_year_next_quarter(year, quarter)
    last_4_quarters = ookla.speed_data(ookla.available_files().loc[('fixed',year,quarter):('fixed',next_year,next_quarter)].path)
    down = last_4_quarters.groupby('quadkey').apply(lambda s:np.average(s.avg_d_kbps, weights=s.tests)).rename('avg_d_kbps')
    up = last_4_quarters.groupby('quadkey').apply(lambda s:np.average(s.avg_u_kbps, weights=s.tests)).rename('avg_u_kbps')
    tests = last_4_quarters.groupby('quadkey')['tests'].sum()
    devices = last_4_quarters.groupby('quadkey')['devices'].sum()
    last4_agg = pd.concat([down, up, tests, devices],axis=1)
    features_table = tile_da_label.merge(da_pops, on='DAUID', how='left')
    features_table['DAPOP'] = features_table['DAPOP'].fillna(0).astype(int)
    del features_table['GEO_NAME']
    features_table = pd.DataFrame(features_table)
    del features_table['geometry']
    features_table['POP_DENSITY'] = features_table['DAPOP']/features_table['das_area']*1000**2 #people per square kilometer
    features_table = ookla_tiles.merge(last4_agg, on='quadkey').merge(features_table, on='quadkey')
    pop_info = statcan.boundary('population_centres').to_crs('epsg:4326')
    pop_info = pop_info[['PCUID', 'PCNAME', 'PCTYPE', 'PCPUID', 'PCCLASS', 'geometry']] ##removes some redundant cols from DAs
    features_table = features_table.sjoin(pop_info, how='left')
    del features_table['index_right']
    features_table = features_table.sort_values(by=['PCUID','quadkey']).drop_duplicates(subset=['quadkey']) #keep tiles where overlap was true
    
    #Add population column
    features_table['population_in_tile'] = features_table['tile_area'] * features_table['POP_DENSITY']/1000000
    
    return features_table

if __name__ == "__main__":
    
    print('Calculating the features_table')
    
    #Calculate total number of people
    features_table = make_features_table(2022,3)
    total_people = sum(features_table['POP_DENSITY'] * features_table['tile_area'])/1000000
    
    del features_table
    
    years = [2019, 2020, 2021, 2022]
    quarters = [1, 2, 3, 4]

    ts_greater_than_50 = []
    ts_less_than_50 = []
    # ts_50 = []
    # ts_10 = []
    
    print('Running the loop')
    
    for year in years:
        for quarter in quarters:

            print(year, 'year', quarter, 'quarter')
            
            internet_speed_type = 'avg_d_kbps' #'avg_u_kbps'

            features_table = make_features_table(year, quarter)

            #'POP_DENSITY' > 50
            ts_greater_than_50.append(features_table[features_table[internet_speed_type] > 50*1000]['population_in_tile'].sum()/total_people)
            #'POP_DENSITY' < 50
            ts_less_than_50.append(features_table[features_table[internet_speed_type] < 50*1000]['population_in_tile'].sum()/total_people)
            # #'POP_DENSITY' > 50
            # gdf = features_table[(features_table['POP_DENSITY'] <= 1000) & (features_table['POP_DENSITY'] >= 50)]
            # ts_50.append(gdf['population_in_tile'].sum()/total_people)
            # #'POP_DENSITY' > 10
            # gdf = features_table[(features_table['POP_DENSITY'] <= 10)]
            # ts_10.append(gdf['population_in_tile'].sum()/total_people)

            print('ts_greater_than_50', ts_greater_than_50)
            print('ts_less_than_50', ts_less_than_50)
