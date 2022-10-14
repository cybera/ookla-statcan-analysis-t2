
## load, list statistis canada related data
### including shapefiles
### including census data?

import dotenv 
import pandas as pd
import geopandas as gp
import json
import functools
import requests

import logging
import zipfile

from pathlib import Path

from src.config import DATA_DIRECTORY
# DATA_DIRECTORY = Path(dotenv.dotenv_values()['DATA_DIRECTORY'])


### StatCan Boundary files
STATCAN_SUBDIR = DATA_DIRECTORY / 'boundaries' 

AUTO_DOWNLOAD = True

def statcan_links():
    with open(STATCAN_SUBDIR / "statcan_links.json") as f: 
        links = json.load(f)['links']
    return links

def boundary_names():
    return list(statcan_links().keys())

@functools.cache
def all_boundaries():
    gpfs = {name:boundary(name) for name in boundary_names()}
    return gpfs

@functools.cache
def boundary(name):
    links = statcan_links()
    statscan_shapefiles = {key:value.split("/")[-1] for key, value in links.items()}
    shapefile_path = STATCAN_SUBDIR / statscan_shapefiles[name] 
    if AUTO_DOWNLOAD and not shapefile_path.exists():
        logging.info(f"Downloading file for {name}:{statscan_shapefiles[name]} data.")
        download_boundary(name)
    gpf = gp.read_file(shapefile_path)
    return gpf 


def download_boundary(name):
    url = statcan_links()[name]
    r = requests.get(url)
    name = url.split("/")[-1]
    with open(STATCAN_SUBDIR / name, 'wb') as file:
        file.write(r.content)

def download_boundaries():
    names = statcan_links().keys()
    for name in names:
        download_boundary(name)




### Hexagon data
HEXAGON_SHAPEFILE = DATA_DIRECTORY / 'Map_Data_Shapefile' / 'Data_Hex_Donnees.SHP'
MAP_DATA_URL = 'https://www.ic.gc.ca/eic/site/720.nsf/vwapj/Map_Data_Shapefile.zip/$file/Map_Data_Shapefile.zip'

def download_hexagons():
    #TODO
    pass

@functools.cache
def hexagons():
    return gp.read_file(HEXAGON_SHAPEFILE)

HEXAGON_POPULATIONS_FILE = DATA_DIRECTORY / 'NBD_PHH_Speeds' / 'PHH_Speeds_Current-PHH_Vitesses_Actuelles.csv'
@functools.cache
def hexagon_populations():
    return pd.read_csv(HEXAGON_POPULATIONS_FILE)



### Demographic data
POP_DIR = DATA_DIRECTORY / 'census_data' 
POP_FILE = POP_DIR / '98-400-X2016003_English_CSV_data.csv'
FILE_URL = 'https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/dt-td/CompDataDownload.cfm?LANG=E&PID=109525&OFT=CSV'

def download_pop_data():
    POP_DIR.mkdir(exist_ok=True)
    url = FILE_URL
    r = requests.get(url)
    with open(POP_DIR / "zip_download.zip", 'wb') as file:
        file.write(r.content)
    with open(POP_DIR / "zip_download.zip", 'rb') as file: 
        zipfile.ZipFile(file).extractall(POP_DIR)

@functools.cache
def populations():
    if not POP_FILE.exists() and AUTO_DOWNLOAD:
        download_pop_data()
    return pd.read_csv(POP_FILE)

def _total_pops(id_name, level):
    total_pops = populations().loc[lambda s:s['DIM: Age (in single years) and average age (127)']=='Total - Age']
    total_pops = total_pops[['GEO_CODE (POR)','GEO_LEVEL','GEO_NAME','Dim: Sex (3): Member ID: [1]: Total - Sex']]
    total_pops = total_pops.rename(columns={'GEO_CODE (POR)':"ID", 'Dim: Sex (3): Member ID: [1]: Total - Sex':"DAPOP"})
    
    total_pops = total_pops.loc[lambda s:s.GEO_LEVEL==level]
    del total_pops['GEO_LEVEL']
    total_pops = total_pops.rename(columns={'ID':id_name})
    return total_pops.copy()

def dissemination_areas_populations():
    return _total_pops('DAUID',4)

def census_subdivisions_populations():
    return _total_pops('CSDUID',3)

def census_divisions_populations():
    return _total_pops("CDUID",2)

