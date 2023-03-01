


""""

Checks for all pertinent data


"""

import os
import json
import time
import pickle
import urllib
import requests
import numpy as np
import pandas as pd

import datetime
from dotenv import load_dotenv
from multiprocessing import Process


load_dotenv()
GOOGLE_AUTH = os.environ.get('GOOGLE_AUTH')
SB_USERNAME = os.environ.get('SB_USERNAME')
SB_PASSWORD = os.environ.get('SB_PASSWORD')
DROPBOX_PATH = os.environ.get('DROPBOX_PATH')
WEATHER_AUTH = os.environ.get('WEATHER_AUTH')



def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def EST_to_UTC(time):
    return time + pd.Timedelta(hours=5)

def statsbomb_to_UTC(time):
    return time - pd.Timedelta(hours=1)

def initialize_stf_checklist(stf_sched):
    
    """
    Need separate checklist for team specific data
    """
#     sb_data_cols = ['raw_events','lineups','atomic_sparse','stf']
#     external_data_cols = ['odds','weather','stadiums','transfermarket']

    cklst = stf_sched.copy()
    cklst = cklst[['datetime_UTC','competition_id','season_id','stadium_id','match_status','last_updated','is_upcoming','match_id','team_id']]
    cklst['datetime_UTC'] = pd.to_datetime(cklst['datetime_UTC'])
    cklst['last_updated'] = pd.to_datetime(cklst['last_updated'])
    cklst['last_updated_UTC'] = cklst['last_updated'].apply(lambda x: statsbomb_to_UTC(x))
    cklst = cklst.drop(columns=['last_updated'])
    
    return cklst

def update_stadium_detail():
    
    """
    
    Updates location of stadiums
    Needed before other retrieval/processing functions
    
    """
    
    stadiums = load_dict(os.path.join(DROPBOX_PATH, 'IDs/stadiums'))
    stadium_detail = load_dict(os.path.join(DROPBOX_PATH, 'stadiums/stadium_detail'))

    existing_stadium_ids = set(stadiums.keys())
    existing_stadium_details = set(stadium_detail.keys())
    needed_ids = existing_stadium_ids.difference(existing_stadium_details)

    if len(needed_ids)==0:
        print("No new stadiums need update...")
    else:
        print(f"{len(needed_ids)} stadiums need details updated...")
        for needed in needed_ids:

            sname = stadiums[needed]['name']
            scountry = stadiums[needed]['country_name']
            print(f"Adding {sname} stadium in {scountry}")

            base_url= "https://maps.googleapis.com/maps/api/geocode/json?"
            # # get location
            # if sid == 117873:# can add custom queries
            #     query = sname + " " + "Guam"
            # else:
            query = sname + " " + scountry

            parameters = {"address": query,"key": GOOGLE_AUTH}
            r = requests.get(f"{base_url}{urllib.parse.urlencode(parameters)}")
            data = json.loads(r.content)
            results = data.get('results')
            if len(results) < 1:
                print(f"Error on stadium id {needed}, {sname}, {scountry}")
            else:
                location = results[0]['geometry']['location']
                place_id = results[0]['place_id']
                base_url= "https://maps.googleapis.com/maps/api/elevation/json?"

                lat = location['lat']
                lng = location['lng']

                parameters = {"locations":f"{lat},{lng}","key": GOOGLE_AUTH}
                r = requests.get(f"{base_url}{urllib.parse.urlencode(parameters)}")
                data = json.loads(r.content)
                elv = data['results'][0]['elevation']

                location_query = f"{lat},{lng}"
                test_call = f"http://api.weatherapi.com/v1/timezone.json?key={WEATHER_AUTH}&q={location_query}"
                r = requests.get(test_call)
                tz_data = json.loads(r.content)
                hour_offset= np.round(((datetime.datetime.utcnow() - pd.to_datetime(tz_data['location']['localtime'])).seconds/3600))

                stadium_detail[needed] = {
                    'location':{'lat':lat,'lng':lng},
                    'place_id':place_id,
                    'elevation':elv,
                    'tz':int(hour_offset)
                }

                time.sleep(0.05)
            print("Added!")

        print("Saving new stadiums...")
        save_dict(stadium_detail, os.path.join(DROPBOX_PATH, 'stadiums/stadium_detail'))
    
    return

def create_lineups_cklst(cklst):
    
    """
    There are a couple of conditions where we will try to pull a lineup
    
    1. A new game
    2. A game that has been updated since the lineup file has been modified
    3. An empty lineup file 
    """

    cklst = cklst.drop_duplicates(subset=['match_id']).reset_index(drop=True)
    print("Doing inventory on Statsbomb lineups...")
    cklst['have_lineups'] = False
    cklst['lineups_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/raw/lineups/{x.competition_id}/{x.season_id}/{x.match_id}.json'), axis=1)
    # see if its a new game that doesn't exist yet
    cklst['have_lineups'] = cklst.apply(lambda x: os.path.exists(x.lineups_path), axis=1)
    # see if its a game that has been updated recently
    cklst['lineups_last_modified_UTC'] = cklst['lineups_path'].apply(lambda x: EST_to_UTC(pd.to_datetime(time.ctime(os.path.getmtime(x)))) if os.path.exists(x) else np.nan)
    # the nulls might as well have been updated in 1971. Otherwise, can throw an error
    cklst['lineups_last_modified_UTC'] = cklst['lineups_last_modified_UTC'].fillna(pd.to_datetime('01-01-1971'))
    cklst['have_lineups'] = np.where(cklst['last_updated_UTC']>cklst['lineups_last_modified_UTC'], False, cklst['have_lineups'].copy())
    # see if its an empty lineup file
    cklst['lineups_sizeof'] = cklst['lineups_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    cklst['have_lineups'] = np.where(cklst['lineups_sizeof'].copy()<10, False, cklst['have_lineups'].copy())

    # some matches just aren't available
    cklst['need_to_try'] = cklst['have_lineups'].copy()
    cklst['need_to_try'] = np.where(cklst['match_status']!='available', True, cklst['need_to_try'])
    
    cklst = cklst[['match_id','have_lineups','need_to_try','lineups_path']]
    # have this backward
    cklst['need_to_try'] = cklst['need_to_try'].apply(lambda x: not x)
    cklst = cklst.rename(columns={'need_to_try':'lineups_need_to_try'})
    cklst = cklst.drop_duplicates().reset_index(drop=True)
    print(f"\nNeed to try to download lineups for {cklst.lineups_need_to_try.sum()} games.")
    cklst.to_csv(os.path.join(DROPBOX_PATH, 'checklists/lineups_checklist.csv'), index=False)

    return

def create_events_cklst(cklst):
    """
    * Basically same function as download needed lineups *
    There are a couple of conditions where we will try to pull a event
    
    1. A new game
    2. A game that has been updated since the event file has been modified
    3. An empty event file (only a very small chance spot check, because 99% of these are just blank)
    """
    print("Doing inventory on Statsbomb events...")

    cklst = cklst.drop_duplicates(subset=['match_id']).reset_index(drop=True)
    cklst['have_events'] = False
    cklst['events_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/raw/events/{x.competition_id}/{x.season_id}/{x.match_id}.json'), axis=1)
    # see if its a new game that doesn't exist yet
    cklst['have_events'] = cklst.apply(lambda x: os.path.exists(x.events_path), axis=1)
    # see if its a game that has been updated recently
    cklst['events_last_modified_UTC'] = cklst['events_path'].apply(lambda x: EST_to_UTC(pd.to_datetime(time.ctime(os.path.getmtime(x)))) if os.path.exists(x) else np.nan)
    cklst['have_events'] = np.where(cklst['last_updated_UTC']>cklst['events_last_modified_UTC'], False, cklst['have_events'].copy())
    # see if its an empty event file
    cklst['events_sizeof'] = cklst['events_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    cklst['have_events'] = np.where(cklst['events_sizeof'].copy()<10, False, cklst['have_events'].copy())
    
    # some matches just aren't available
    cklst['need_to_try'] = cklst['have_events'].copy()
    cklst['need_to_try'] = np.where(cklst['match_status']!='available', True, cklst['need_to_try'])
    
    cklst = cklst[['match_id','have_events','need_to_try','events_path']]
    # have this backward
    cklst['need_to_try'] = cklst['need_to_try'].apply(lambda x: not x)
    cklst = cklst.rename(columns={'need_to_try':'events_need_to_try'})
    cklst = cklst.drop_duplicates().reset_index(drop=True)
    cklst.to_csv(os.path.join(DROPBOX_PATH, 'checklists/events_checklist.csv'), index=False)

    print(f"\nNeed to try to download events for {cklst.events_need_to_try.sum()} games.")
    
    return




def make_folder(base_path, comp_id, season_id):
    
    base_path = os.path.join(DROPBOX_PATH, base_path)
    if not os.path.isdir(os.path.join(base_path, f"{comp_id}")):
        print(f"New competition id, {comp_id}")
        os.mkdir(os.path.join(base_path, f"{comp_id}"))
    if not os.path.isdir(os.path.join(base_path, f"{comp_id}/{season_id}")):
        print(f"New season id ( {season_id} ) in competition id, {comp_id}")
        os.mkdir(os.path.join(base_path, f"{comp_id}/{season_id}"))
    
    return

def get_stadium_location(x, stadium_detail):
    if pd.isnull(x):
        return np.nan, np.nan, np.nan
    return stadium_detail[x]['location'].get('lat'),stadium_detail[x]['location'].get('lng'),stadium_detail[x]['tz']

def weather_path_exists(x):
    is_upcoming = x.is_upcoming
    if is_upcoming:
        weather_path = os.path.join(DROPBOX_PATH, f'weather/upcoming/{x.competition_id}/{x.season_id}/{x.match_id}.json')
        exists = os.path.exists(weather_path)
    else:
        weather_path = os.path.join(DROPBOX_PATH, f'weather/historical/{x.competition_id}/{x.season_id}/{x.match_id}.json')
        exists = os.path.exists(weather_path)
    return weather_path, exists

def weather_cklst_process(chklst):
    stadium_detail = load_dict(os.path.join(DROPBOX_PATH, 'stadiums/stadium_detail')) 
    chklst['recently_updated'] = chklst.apply(lambda x: ((datetime.datetime.now()-datetime.datetime.fromtimestamp(os.path.getmtime(x.file_path))).total_seconds()/3600)<2 if x.have_weather else False,axis=1)
    ## drop recently updated
    chklst = chklst.loc[chklst['recently_updated']==False].reset_index(drop=True)

    chklst['lat'], chklst['lng'], chklst['tz'] = zip(*chklst['stadium_id'].apply(lambda x: get_stadium_location(x, stadium_detail)))
    
    # some early games are missing info, so fillna with comp averages
    lat_map = chklst.copy().groupby(['competition_id'])['lat'].mean().to_dict()
    lng_map = chklst.copy().groupby(['competition_id'])['lng'].mean().to_dict()

    chklst['backup_lat'] = chklst['competition_id'].map(lat_map)
    chklst['backup_lng'] = chklst['competition_id'].map(lng_map)

    chklst['lat'] = chklst['lat'].fillna(chklst['backup_lat'].copy())
    chklst['lng'] = chklst['lng'].fillna(chklst['backup_lng'].copy())
    
    chklst['tz'] = chklst['tz'].copy().apply(lambda x: pd.to_timedelta(x) if x <= 12 else pd.to_timedelta(-(24-x)))
    
    chklst['backup_tz'] = chklst.groupby(['competition_id'])['tz'].transform(lambda x: x.mode()[0])
    chklst['tz'] = chklst['tz'].fillna(chklst['backup_tz'].copy())
    chklst = chklst.drop(columns=['backup_lat','backup_lng','backup_tz'])

    chklst['local_datetime'] = chklst['datetime_UTC'].copy() - chklst['tz'].copy()
    chklst['local_hour'] = chklst['local_datetime'].dt.hour
    chklst['local_date'] = chklst['local_datetime'].dt.date

    for index, row in chklst.iterrows():
        make_folder(os.path.join(DROPBOX_PATH, f'weather\\historical'),row['competition_id'], row['season_id'])
        make_folder(os.path.join(DROPBOX_PATH, f'weather\\upcoming'),row['competition_id'], row['season_id'])

    return chklst



def create_weather_cklst(stf_checklist, cutoff=14, update_tol=4):
    """
    update tol is how many hours since last pull to update weather projections
    cutoff is how many days out to get weather
    
    """
    print("Doing inventory on weather data...")

    weather_checklist = stf_checklist.copy()[['datetime_UTC','competition_id','season_id','match_id','stadium_id','is_upcoming']]
    # ## TODO: make best guess for stadium location without id
    weather_checklist = weather_checklist.dropna(subset=['match_id']).reset_index(drop=True)
    weather_checklist = weather_checklist.drop_duplicates(subset=['match_id']).reset_index(drop=True)
    weather_checklist['stadium_id'] = weather_checklist['stadium_id'].astype('Int64')
    weather_checklist['file_path'], weather_checklist['have_weather'] = zip(*weather_checklist.apply(lambda x: weather_path_exists(x), axis=1))

    all_games = len(weather_checklist)
    weather_checklist = weather_checklist.loc[weather_checklist['datetime_UTC']<=datetime.datetime.utcnow()+pd.Timedelta(days=cutoff)].reset_index(drop=True)
    games_less_than_2_weeks_out = len(weather_checklist)
    print(f"We dropped {all_games-games_less_than_2_weeks_out} games from weather collection because they were more than {cutoff} days out")

    print(f"Minimum time since last weather grab (because limited API calls): {update_tol} hours")
    
    weather_checklist = weather_cklst_process(weather_checklist)
    weather_checklist['weather_need_to_try'] = np.where(((weather_checklist['have_weather']==False)&(weather_checklist['recently_updated']==False)), True, False)
    
    hist = weather_checklist.copy().loc[((weather_checklist['is_upcoming']==0)&(weather_checklist['weather_need_to_try']==True))]
    upc = weather_checklist.copy().loc[((weather_checklist['is_upcoming']==1)&(weather_checklist['weather_need_to_try']==True))]
    
    print(f"\nNeed to try to download weather for {len(hist)} past events, and {len(upc)} upcoming events, for {len(hist)+len(upc)} events total...")
    
    weather_checklist = weather_checklist.rename(columns={'file_path':'weather_path'})
    weather_checklist = weather_checklist[['match_id','competition_id','season_id','is_upcoming','weather_need_to_try','weather_path','local_date','local_datetime','local_hour','lat','lng','tz']]
    weather_checklist.to_csv(os.path.join(DROPBOX_PATH, 'checklists/weather_checklist.csv'), index=False)
    
    return 

def data_inventory():

    stf_schedule = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/stf_schedule.csv'))
    stf_checklist = initialize_stf_checklist(stf_schedule)
    update_stadium_detail()
    stadium_detail = load_dict(os.path.join(DROPBOX_PATH, 'stadiums/stadium_detail'))

    jobs = []
    p1 = Process(target=create_events_cklst, args=(stf_checklist.copy(),))
    jobs.append(p1)
    p1.start()
    p2 = Process(target=create_lineups_cklst, args=(stf_checklist.copy(),))
    jobs.append(p2)
    p2.start()
    p3 = Process(target=create_weather_cklst, args=(stf_checklist.copy(),))
    jobs.append(p3)
    p3.start()

    # checks to see if they are finished
    for job in jobs:
        job.join()
        time.sleep(1)

    return



if __name__=='__main__':

    data_inventory()








