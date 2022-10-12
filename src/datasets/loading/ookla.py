
import dotenv 
import pandas as pd
import geopandas as gp
import json
import functools
import requests
import re

import logging

from pathlib import Path

from src.config import DATA_DIRECTORY

#DATA_DIRECTORY = Path(dotenv.dotenv_values()['DATA_DIRECTORY'])
#all_ookla_geometry_tiles = list(CANADA_TILES_DIR.glob('./**/ookla-*.shp'))

def canada_tiles(rows=None):
    canada_tile_geometry_file = DATA_DIRECTORY / 'ookla-canada-tiles' / 'canada-tiles'
    if canada_tile_geometry_file.exists():
        return gp.read_file(canada_tile_geometry_file, rows=rows)
        
    labeled_tiles = DATA_DIRECTORY / 'ookla-canada-tiles' / 'all-tile-labels'
    labeled_tiles = gp.read_file(labeled_tiles, rows=rows)
    
    #only_canada = labeled_tiles.loc[lambda s:s.in_canada==1].copy()
    only_canada = labeled_tiles.loc[lambda s:s.in_canada==1] #Added this line as per Huyue
    #only_canada['quadkey'] = only_canada['quadkey'].astype(int)
    only_canada = only_canada[['quadkey','geometry']]
    only_canada.to_file(canada_tile_geometry_file,driver="ESRI Shapefile")
    return only_canada

def available_files():
    DIR = DATA_DIRECTORY / 'ookla-canada-tiles'
    canada_speed_files = list(DIR.glob('./**/ookla-*.csv'))
    pat = re.compile('.*ookla-canada-(?P<year>[0-9]{4})-Q(?P<quarter>[1-4])-(?P<type>fixed|mobile)-tiles*')

    meta_data = [{'path':file} | pat.match(str(file)).groupdict() for file in canada_speed_files]
    
    data_files = pd.DataFrame(meta_data)
    data_files['year'] = data_files['year'].astype(int)
    data_files['quarter'] = data_files['quarter'].astype(int)
    data_files = data_files.sort_values(by=['type','year','quarter']).reset_index(drop=True).set_index(['type','year','quarter'])

    return data_files

def speed_data(paths=None):
    if paths is None:
        paths = available_files().path

    df = pd.concat([
        pd.read_csv(f) for f in paths        
    ])
    df['quadkey'] = df['quadkey'].astype(int)
    return df


#Changed to ensure proper merge
def canada_speed_tiles():
    
    test_var = 1

    tiles = canada_tiles()
    tiles['quadkey'] = tiles['quadkey'].astype(int)
    return tiles.merge(speed_data(), on='quadkey', validate='1:m')

    #return canada_tiles().merge(speed_data(), on='quadkey', validate='1:m')