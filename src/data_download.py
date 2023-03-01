

import os
import sys
import json
import math
import time
import pickle
import requests

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from statsbombpy import sb
from datetime import datetime
from dotenv import load_dotenv
from multiprocessing import Process
from fuzzywuzzy import fuzz, process
from src.modules.footy import update_footy


load_dotenv()
SB_USERNAME = os.environ.get('SB_USERNAME')
SB_PASSWORD = os.environ.get('SB_PASSWORD')
DROPBOX_PATH = os.environ.get('DROPBOX_PATH')
WEATHER_AUTH = os.environ.get('WEATHER_AUTH')
ODDS_API_KEY = os.environ.get('ODDS_API')
creds={"user":SB_USERNAME,"passwd":SB_PASSWORD}
VERSION = 'v5'
HOSTNAME = 'https://data.statsbombservices.com/'



### Helpers

def make_folder(base_path, comp_id, season_id):
    
    base_path = os.path.join(DROPBOX_PATH, base_path)
    if not os.path.isdir(os.path.join(base_path, f"{comp_id}")):
        print(f"New competition id, {comp_id}")
        os.mkdir(os.path.join(base_path, f"{comp_id}"))
    if not os.path.isdir(os.path.join(base_path, f"{comp_id}/{season_id}")):
        print(f"New season id ( {season_id} ) in competition id, {comp_id}")
        os.mkdir(os.path.join(base_path, f"{comp_id}/{season_id}"))
    
    return

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

stadium_detail = load_dict(os.path.join(DROPBOX_PATH, 'stadiums/stadium_detail'))



def download_Statsbomb_events():
    
    events = pd.read_csv(os.path.join(DROPBOX_PATH, 'checklists/events_checklist.csv'))
    comp_season_ids = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'), usecols=['match_id','competition_id','season_id'])
    events = events.merge(comp_season_ids, how='left', on=['match_id'])
    events = events.drop_duplicates(subset=['match_id']).reset_index(drop=True)
    events = events.loc[events['events_need_to_try']==True].reset_index(drop=True)
    for id_ in ['competition_id','season_id']:
        events[id_] = events[id_].astype(int)
    possible_events_to_download = len(events)
    print(f"There are {possible_events_to_download} possible events to download.")
    if possible_events_to_download > 0:
        successful = []
        for index, row in tqdm(events.iterrows(), total=len(events)):
            path = row['events_path']
            match_id = row['match_id']
            make_folder(os.path.join(DROPBOX_PATH, f'Statsbomb/raw/events/'),row['competition_id'], row['season_id'])
            # the list(.values()) below is unnecessary but is the format we've been using 
            game_events = list(sb.events(creds=creds, match_id=match_id, fmt="dict").values()) 
            # length should be around 3,000
            if len(game_events)>25:
                successful.append(match_id)
                with open(path, 'w') as fp:
                    json.dump(game_events, fp)
            else:
                print(f"Error on match id {match_id}, will not add to successful events downloaded and will retry next run")
        num_events_downloaded = len(successful)
        print(f"Done! {np.round((num_events_downloaded/possible_events_to_download)*100, 1)}% of events were successfully downloaded.")

    return

def download_Statsbomb_lineups():
    
    lineups = pd.read_csv(os.path.join(DROPBOX_PATH, 'checklists/lineups_checklist.csv'))
    comp_season_ids = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'), usecols=['match_id','competition_id','season_id'])
    lineups = lineups.merge(comp_season_ids, how='left', on=['match_id'])
    lineups = lineups.loc[lineups['lineups_need_to_try']==True].reset_index(drop=True)
    for id_ in ['competition_id','season_id']:
        lineups[id_] = lineups[id_].astype(int)
    possible_lineups_to_download = len(lineups)
    if possible_lineups_to_download > 0:
        successful = []
        for index, row in tqdm(lineups.iterrows(), total=len(lineups)):
            path = row['lineups_path']
            match_id = row['match_id']
            make_folder(os.path.join(DROPBOX_PATH, f'Statsbomb/raw/lineups/'),row['competition_id'], row['season_id'])
            game_lineup = sb.lineups(creds=creds, match_id=match_id, fmt="dict")
#             print(len(game_lineup))
            if len(game_lineup)>1:
                successful.append(match_id)
                with open(path, 'w') as fp:
                    json.dump(game_lineup, fp)
            else:
                print(f"Error on match id {match_id}, will not add to successful lineups downloaded and will retry next run")

        num_lineups_downloaded = len(successful)
        print(f"Done! {np.round((num_lineups_downloaded/possible_lineups_to_download)*100, 1)}% of lineups were successfully downloaded.")

    return

### WEATHER FUNCTIONS ###

def fetch_historical_weather(hist):
    
    total_possible = len(hist)
    if total_possible < 1:
        return
    successful = []
    for index, row in tqdm(hist.iterrows(), total=len(hist)):
        
        match_id = row['match_id']
        competition = row['competition_id']
        season = row['season_id']

        date = row['local_date']
        lat = row['lat']
        lng = row['lng']
        if np.isnan(lat):
#             print(match_id)
            continue
        if np.isnan(lng):
#             print(match_id)
            continue

        location = f"{lat},{lng}"
        call = f"http://api.weatherapi.com/v1/history.json?key={WEATHER_AUTH}&q={location}&dt={date}"

        r = requests.get(call)
        try:
            data = json.loads(r.content)
        except:
            print(f"Match id {match_id} did not pull correctly")
            time.sleep(0.1)
            continue
        if 'error' in data.keys():
            print(f"Match id {match_id} did not pull correctly")
            time.sleep(0.1)
            continue
        assert('error' not in data.keys())
        if sys.getsizeof(data) > 25:
            path = os.path.join(DROPBOX_PATH, f'weather/historical/{competition}/{season}/{match_id}.json')
            with open(path, 'w') as f:
                json.dump(data, f)
            successful.append(match_id)
        else:
            print(f"Match id {match_id} did not pull correctly")
            time.sleep(0.1)
            continue
        time.sleep(0.1)
    
    print(f"Done! {np.round((len(successful)/total_possible)*100, 1)}% of historical game weather was successfully downloaded.")
    
    return

def fetch_upcoming_weather(upc):
    
    total_possible = len(upc)
    successful = []
    if total_possible < 1:
        print('\nNo upcoming games to collect weather for ????\n')
        return
    for index, row in tqdm(upc.iterrows(), total=len(upc)):
        match_id = row['match_id']

        competition = row['competition_id']
        season = row['season_id']

        match_date = pd.to_datetime(row['local_datetime'])
        now = datetime.utcnow() - pd.Timedelta(row['tz'])
        diff = (match_date - now).days + 2 # for some, we will need 1, but that logic is easier later

        lat = row['lat']
        lng = row['lng']
        if np.isnan(lat):
    #             print(match_id)
            continue
        if np.isnan(lng):
    #             print(match_id)
            continue

        location = f"{lat},{lng}"
        call = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_AUTH}&q={location}&days={diff}"

        r = requests.get(call)
        try:
            data = json.loads(r.content)
        except:
            data = 0
            time.sleep(0.1)
            print(f"Match id {match_id} did not pull correctly")
            continue

        correct_index = None
        for index, date_data in enumerate(data['forecast']['forecastday']):
            if pd.to_datetime(date_data['date']).date() == match_date.date():
                correct_index = index

        assert('error' not in data.keys())
        if sys.getsizeof(data) > 25:
            upcoming_data = data['forecast']['forecastday'][index]
            path = os.path.join(DROPBOX_PATH, f'weather/upcoming/{competition}/{season}/{match_id}.json')
            with open(path, 'w') as f:
                json.dump(upcoming_data, f)
            successful.append(match_id)
        time.sleep(0.1)
        
    print(f"Done! {np.round((len(successful)/total_possible)*100, 1)}% of upcoming game weather was successfully downloaded.")
        
    return

def get_stadium_location(x):
    return stadium_detail[x]['location'].get('lat'),stadium_detail[x]['location'].get('lng'),stadium_detail[x]['tz']

def to_timedelta(x):
    return np.nan if np.isnan(x) else pd.Timedelta(hours=x) 

def load_schedules():
    
    normal = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'))
    stf = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/stf_schedule.csv'))
    
    normal['datetime_UTC'] = pd.to_datetime(normal['datetime_UTC'].copy())
    stf['datetime_UTC'] = pd.to_datetime(stf['datetime_UTC'].copy())

    normal['match_date'] = normal['datetime_UTC'].copy().dt.date
    stf['match_date'] = stf['datetime_UTC'].copy().dt.date
    
    normal['last_updated'] = pd.to_datetime(normal['last_updated'].copy())
    stf['last_updated'] = pd.to_datetime(stf['last_updated'].copy())
    
    normal = normal.loc[~normal['match_status'].isin(['deleted','collecting','cancelled','postponed'])].reset_index(drop=True)
    stf = stf.loc[~stf['match_status'].isin(['deleted','collecting','cancelled','postponed'])].reset_index(drop=True)
    
    return normal, stf 
def extract_day_weather(path, hour, is_upcoming=False):
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    hours = [int(hour+i) for i in range(3)]
    hours = [h for h in hours if h < 24]

    temps = []
    winds = []
    press = []
    humid = []
    precip = []

    if is_upcoming:
        hourly_weather = data['hour'] # many days, but has already been selected
        if len(hourly_weather) != 24:
            print("Error, not completely updated, removing")
            os.remove(path)
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        try:
            hourly_weather = data['forecast']['forecastday'][0]['hour'] # only one day, have to select it
        except:
            print("THIS IS LOADED DATA", data)
            if data['error']['code'] == 2006:
                print("Error, api key invalid, removing")
                os.remove(path)
                return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            else:
                raise ValueError
    for h in hours:
        hourly = hourly_weather[h]
        temps.append(hourly['temp_f'])
        winds.append(hourly['wind_mph'])
        press.append(hourly['pressure_in'])
        humid.append(hourly['humidity'])
        precip.append(hourly['precip_in'])

    temp, wind, pressure, humidity, rain = np.mean(temps), np.mean(winds), np.mean(press), np.mean(humid), np.mean(precip)
    last_hour = hours[-1]
    prior_rain = [h['precip_in'] for h in hourly_weather]
    prior_rain = prior_rain[:last_hour]
    prior_rain = np.mean(prior_rain)/len(prior_rain)
    
    return [temp, wind, pressure, humidity, rain, prior_rain]

def aggregate_historical(sched):

    hist_weather = pd.read_csv(os.path.join(DROPBOX_PATH, 'weather/historical/aggregated.csv'))
    existing = set(hist_weather.match_id.unique())

    weather_hist_path = os.path.join(DROPBOX_PATH, 'weather/historical')

    EXT = "*.json"
    existing_json_files = set([file
                    for path, subdir, files in os.walk(weather_hist_path)
                    for file in glob(os.path.join(path, EXT))])

    hist_weather = pd.read_csv(os.path.join(DROPBOX_PATH, 'weather/historical/aggregated.csv'))

    hist = pd.Series(list(existing_json_files)).reset_index()
    hist.columns=['index','file_path']
    hist['match_id'] = hist['file_path'].apply(lambda x: int(x[x.rfind('\\')+1:x.rfind('.')]))
    hist = hist.drop(columns=['index'])
    need_aggregated = set(hist.match_id.unique())
    existing = set(hist_weather.match_id.unique())
    to_extract = need_aggregated.difference(existing)

    if len(to_extract) == 0:
        print("No new historical weather needed")
        return

    extract = sched.copy().loc[sched['is_upcoming']==0].copy().reset_index()
    extract = extract.dropna(subset=['match_id','stadium_id']).reset_index(drop=True)
    extract['stadium_id'] = extract['stadium_id'].astype(int)
    extract['lat'], extract['lng'], extract['tz'] = zip(*extract['stadium_id'].apply(lambda x: get_stadium_location(x)))
    extract['tz'] = extract['tz'].copy().apply(lambda x: to_timedelta(x) if x <= 12 else to_timedelta(-(24-x)))

    extract['local_datetime'] = extract['datetime_UTC'].copy() - extract['tz'].copy()
    extract['local_hour'] = extract['local_datetime'].dt.hour
    extract['local_date'] = extract['local_datetime'].dt.date

    extract['file_path'] = extract.apply(lambda x:  os.path.join(DROPBOX_PATH, f'weather/historical/{x.competition_id}/{x.season_id}/{x.match_id}.json'), axis=1)

    extract = extract.copy()[['match_id','file_path','local_hour']]
    extract = extract.loc[extract['match_id'].isin(to_extract)].reset_index(drop=True)
    new_data = []
    for index, game in extract.iterrows():
        game_path = game['file_path']
        game_hour = game['local_hour']
        if not os.path.exists(game_path):
            continue
        row_data = [game['match_id']] + extract_day_weather(game_path, game_hour)
        num_nans = sum(math.isnan(x) for x in row_data)
        if num_nans > 2:
            continue
        new_data.append(row_data)

    if len(new_data) > 0:
        new_data = pd.DataFrame(new_data, columns=list(hist_weather))
        before = len(hist_weather)
        hist_weather = pd.concat([hist_weather, new_data], axis=0).drop_duplicates(subset=['match_id'],keep='last').reset_index(drop=True)
        after = len(hist_weather)
        print(f"Weather for {after-before} historical games added!")
        hist_weather = hist_weather.dropna(subset=['temperature','wind','pressure','humidity','precip','prior_precip']).reset_index(drop=True)
        hist_weather.to_csv(os.path.join(DROPBOX_PATH, 'weather/historical/aggregated.csv'), index=False)
    return

def aggregate_upcoming(sched):

    weather_upc_path = os.path.join(DROPBOX_PATH, 'weather/upcoming')

    EXT = "*.json"
    existing_json_files = set([file
                     for path, subdir, files in os.walk(weather_upc_path)
                     for file in glob(os.path.join(path, EXT))])

    hist_weather = pd.read_csv(os.path.join(DROPBOX_PATH, 'weather/historical/aggregated.csv'))

    upc = pd.Series(list(existing_json_files)).reset_index()
    upc.columns=['index','file_path']
    upc['match_id'] = upc['file_path'].apply(lambda x: int(x[x.rfind('\\')+1:x.rfind('.')]))
    upc = upc.drop(columns=['index'])

    need_deleted = [path for path in list(upc.match_id.values) if path in list(hist_weather.match_id.values)]
    to_delete = upc.copy().loc[upc['match_id'].isin(need_deleted)].reset_index(drop=True)
    print(f"Deleting {len(to_delete)} json files from upcoming as they have now occurred")
    for index, row in to_delete.iterrows():
        fp = row['file_path']
        os.remove(fp)

    upc = upc.copy().loc[~upc['match_id'].isin(need_deleted)].reset_index(drop=True)
    need_aggregated = set(upc.match_id.unique())

    extract = sched.copy().loc[sched['is_upcoming']==1].copy().reset_index()
    extract = extract.dropna(subset=['match_id','stadium_id']).reset_index(drop=True)
    extract['stadium_id'] = extract['stadium_id'].astype(int)
    extract['lat'], extract['lng'], extract['tz'] = zip(*extract['stadium_id'].apply(lambda x: get_stadium_location(x)))
    extract['tz'] = extract['tz'].copy().apply(lambda x: to_timedelta(x) if x <= 12 else to_timedelta(-(24-x)))

    extract['local_datetime'] = extract['datetime_UTC'].copy() - extract['tz'].copy()
    extract['local_hour'] = extract['local_datetime'].dt.hour
    extract['local_date'] = extract['local_datetime'].dt.date

    extract['file_path'] = extract.apply(lambda x:  os.path.join(DROPBOX_PATH, f'weather/upcoming/{x.competition_id}/{x.season_id}/{x.match_id}.json'), axis=1)
    extract = extract.copy()[['match_id','file_path','local_hour']]
    upc = extract.copy().loc[extract['match_id'].isin(need_aggregated)].reset_index(drop=True)

    new_data = []
    for index, game in upc.iterrows():
        game_path = game['file_path']
        game_hour = game['local_hour']
        if np.isnan(game_hour):
            game_hour = 20
        if not os.path.exists(game_path):
            print(game_path)
            continue
        row_data = [game['match_id']] + extract_day_weather(game_path, game_hour, is_upcoming=True)
        new_data.append(row_data)

    if len(new_data) > 0:
        new_data = pd.DataFrame(new_data, columns=['match_id','temperature','wind','pressure','humidity','precip','prior_precip'])
        new_data = new_data.dropna(subset=['temperature','wind','pressure','humidity','precip','prior_precip']).reset_index(drop=True)
        print(f"Weather for {len(new_data)} upcoming games aggregated!")
        new_data.to_csv(os.path.join(DROPBOX_PATH, 'weather/upcoming/aggregated.csv'), index=False)

    return



def download_weather():
    weather_checklist = pd.read_csv(os.path.join(DROPBOX_PATH, 'checklists/weather_checklist.csv'))
    
    weather_checklist = weather_checklist.loc[weather_checklist['weather_need_to_try']==True].copy().reset_index(drop=True)
    hist = weather_checklist.copy()[weather_checklist['is_upcoming']==0].reset_index(drop=True)
    upc = weather_checklist.copy()[weather_checklist['is_upcoming']==1].reset_index(drop=True)
    fetch_historical_weather(hist)
    fetch_upcoming_weather(upc)

    schedule, stf_schedule=load_schedules()
    aggregate_historical(schedule.copy())
    aggregate_upcoming(schedule.copy())

    hist = pd.read_csv(os.path.join(DROPBOX_PATH, 'weather/historical/aggregated.csv'))
    upc = pd.read_csv(os.path.join(DROPBOX_PATH, 'weather/upcoming/aggregated.csv'))

    weather = pd.concat([hist, upc], axis=0).drop_duplicates(subset=['match_id'], keep='first').reset_index(drop=True)
    weather.to_csv(os.path.join(DROPBOX_PATH, 'weather/aggregated.csv'), index=False)
    
    return


### ODDS FUNCTIONS ###

comp2oddsapi = {
    81:'soccer_argentina_primera_division',
    93:'soccer_australia_aleague',
    46:'soccer_belgium_first_div',
    71:'soccer_brazil_campeonato',
    104:'soccer_china_superleague',
    77:'soccer_denmark_superliga',
    2:'soccer_epl',
    3:'soccer_efl_champ',
    4:'soccer_england_league1',
    5:'soccer_england_league2',
    43:'soccer_fifa_world_cup',
    7:'soccer_france_ligue_one',
    8:'soccer_france_ligue_two',
    9:'soccer_germany_bundesliga',
    10:'soccer_germany_bundesliga2',
    12:'soccer_italy_serie_a',
    84:'soccer_italy_serie_b',
    108:'soccer_japan_j_league',
    73:'soccer_mexico_ligamx',
    6:'soccer_netherlands_eredivisie',
    88:'soccer_norway_eliteserien',
    38:'soccer_poland_ekstraklasa',
    13:'soccer_portugal_primeira_liga',
    128:'soccer_russia_premier_league',
    11:'soccer_spain_la_liga',
    42:'soccer_spain_segunda_division',
    43:'soccer_uefa_europa_conference_league',
    51:'soccer_spl',
    75:'soccer_sweden_allsvenskan',
    249:'soccer_sweden_superettan',
    80:'soccer_switzerland_superleague',
    85:'soccer_turkey_super_league',
    16:'soccer_uefa_champs_league',
    35:'soccer_uefa_europa_league',
    353:'soccer_uefa_europa_conference_league',
    102:'soccer_conmebol_copa_libertadores',
    55:'soccer_uefa_european_championship',
    44:'soccer_usa_mls',
    125:'soccer_switzerland_superleague',
    107:'soccer_league_of_ireland',
    106:'soccer_finland_veikkausliiga',
    109:'soccer_korea_kleague1'


    
#     103:'soccer_chile_campeonato', # don't use this anymore
    
    
}

teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
oddsapi2comp = {v:k for k,v in comp2oddsapi.items()}

### print this for list of don't haves
#{k:v for k,v in competitions.items() if (k not in comp2oddsapi)&('WC Qual' not in v['name'])&('Play-offs' not in v['name'])}


### ODDS HELPERS ###

def handle_spreads(row_data,home_team_name,away_team_name):
    assert(len(row_data)==2)
    for i, outcome in enumerate(row_data):
        team_name = outcome['name']
        assert(team_name != 'Draw')
        team_price = outcome['price']
        team_point = outcome['point']
        if team_name == home_team_name:
            home_team_price = team_price
            home_team_point = team_point
        elif team_name == away_team_name:
            away_team_price = team_price 
            away_team_point = team_point
        else:
            raise ValueError("Market has price for team that isn't playing?")
    return [home_team_point, home_team_price, away_team_price]
                                
                                
def handle_totals(row_data):
    assert(len(row_data)==2)
    side_1 = row_data[0]
    side_2 = row_data[1]
    if side_1['name'] == 'Over':
        over = side_1['point']
        over_price = side_1['price']
        under = side_2['point']
        under_price = side_2['price']
    elif side_1['name'] == 'Under':
        under = side_1['point']
        under_price = side_1['price']
        over = side_2['point']
        over_price = side_2['price']
    return [over, over_price, under_price]
                                
                                
def handle_h2h(row_data,home_team_name, away_team_name):
    assert(len(row_data)==3)
    for i, outcome in enumerate(row_data):
        if (i == 0)|(i==1):
            team_name = outcome['name']
            assert(team_name != 'Draw')
            team_price = outcome['price']
            if team_name == home_team_name:
                home_team_price = team_price
            elif team_name == away_team_name:
                away_team_price = team_price      
            else:
                raise ValueError()
        else:
            draw_name = outcome['name']
            assert(draw_name == 'Draw')
            draw_price = outcome['price']
    return [home_team_price, draw_price, away_team_price]



def handle_odds_response(response_json, league_code):
    
    ## probably an empty response
    if response_json[0].get('bookmakers') is None:
        print(f"Failure on league {league_code}")
        return None, False

    valid_book_keys = ['Pinnacle','BetOnline.ag']
    data = pd.json_normalize(response_json.copy(), 
                             record_path=['bookmakers'],
                             meta = ['id','sport_key','sport_title','commence_time','home_team','away_team']
                            )

    ## probably an empty response
    if 'key' not in list(data):
        print(f"Failure on league {league_code}")
        return None, False
    data = data.rename(columns={'title':'book_name'})
    data = data.drop(columns=['key'])
    data = data.loc[data['book_name'].isin(valid_book_keys)].reset_index(drop=True)
    
    h2h_cols = ['home_ml','draw','away_ml']
    h2h_data = []
    spread_cols = ['spread_line','home_price','away_price']
    spread_data = []
    total_cols = ['total_line','over_price','under_price']
    total_data = []

    for index, row in data.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        market_row_data = row['markets']
        # initialize as nans in case they only have some of the markets
        row_total_data, row_spread_data, row_h2h_data = [np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan]

        for market in market_row_data:
            if market['key'] == 'h2h':
                row_h2h_data = handle_h2h(market['outcomes'],home_team,away_team)
            elif market['key'] == 'spreads':
                row_spread_data = handle_spreads(market['outcomes'],home_team,away_team)
            elif market['key'] == 'totals':
                row_total_data = handle_totals(market['outcomes'])

        h2h_data.append(row_h2h_data)
        spread_data.append(row_spread_data)
        total_data.append(row_total_data)

    h2h_data = pd.DataFrame(h2h_data, columns=h2h_cols)  
    spread_data = pd.DataFrame(spread_data, columns=spread_cols)
    total_data = pd.DataFrame(total_data, columns=total_cols)

    extracted_data = pd.concat([h2h_data, spread_data, total_data], axis=1)
    data = data.drop(columns=['markets'])
    data = pd.concat([data, extracted_data], axis=1)
    
    if 'book_name' not in list(data):
        print(f"Failure on league {league_code}")
        return None, False

    return data, True

#### leagues they have we don't 
## 'soccer_finland_veikkausliiga'
## 'soccer_korea_kleague1'
## 'soccer_league_of_ireland'

comp2oddsapi = {
    81:'soccer_argentina_primera_division',
    93:'soccer_australia_aleague',
    46:'soccer_belgium_first_div',
    71:'soccer_brazil_campeonato',
    104:'soccer_china_superleague',
    77:'soccer_denmark_superliga',
    2:'soccer_epl',
    3:'soccer_efl_champ',
    4:'soccer_england_league1',
    5:'soccer_england_league2',
    43:'soccer_fifa_world_cup',
    7:'soccer_france_ligue_one',
    8:'soccer_france_ligue_two',
    9:'soccer_germany_bundesliga',
    10:'soccer_germany_bundesliga2',
    12:'soccer_italy_serie_a',
    84:'soccer_italy_serie_b',
    108:'soccer_japan_j_league',
    73:'soccer_mexico_ligamx',
    6:'soccer_netherlands_eredivisie',
    88:'soccer_norway_eliteserien',
    13:'soccer_portugal_primeira_liga',
    128:'soccer_russia_premier_league',
    11:'soccer_spain_la_liga',
    42:'soccer_spain_segunda_division',
    51:'soccer_spl',
    75:'soccer_sweden_allsvenskan',
    249:'soccer_sweden_superettan',
    80:'soccer_switzerland_superleague',
    85:'soccer_turkey_super_league',
    16:'soccer_uefa_champs_league',
    35:'soccer_uefa_europa_league',
    55:'soccer_uefa_european_championship',
    44:'soccer_usa_mls',
    125:'soccer_switzerland_superleague',
    107:'soccer_league_of_ireland',
    106:'soccer_finland_veikkausliiga',
    109:'soccer_korea_kleague1'
    
}
oddsapi2comp = {v:k for k,v in comp2oddsapi.items()}
def load_upcoming():
    schedule, stf_schedule = load_schedules()

    upcoming = schedule.copy().loc[schedule['is_upcoming']==1]
    upcoming = upcoming.copy()[['match_date_UTC','datetime_UTC','match_id','competition_id','season_id','home_team_id','away_team_id'
                               ]]
    upcoming['home_team_name'] = upcoming['home_team_id'].apply(lambda x: teams.get(x)['name'])
    upcoming['away_team_name'] = upcoming['away_team_id'].apply(lambda x: teams.get(x)['name'])

    upcoming['competition_name'] = upcoming['competition_id'].apply(lambda x: competitions.get(x)['name'])
    upcoming['country_name'] = upcoming['competition_id'].apply(lambda x: competitions.get(x)['country_name'])

    upcoming['oa_comp_id'] = upcoming['competition_id'].map(comp2oddsapi)
    upcoming = upcoming.dropna(subset=['oa_comp_id']).reset_index(drop=True)
    
    return upcoming

def load_schedules():
    
    normal = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'))
    stf = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/stf_schedule.csv'))
    
    normal['datetime_UTC'] = pd.to_datetime(normal['datetime_UTC'].copy())
    stf['datetime_UTC'] = pd.to_datetime(stf['datetime_UTC'].copy())

    normal['match_date_UTC'] = normal['datetime_UTC'].copy().dt.date
    stf['match_date_UTC'] = stf['datetime_UTC'].copy().dt.date
    
    normal['last_updated'] = pd.to_datetime(normal['last_updated'].copy())
    stf['last_updated'] = pd.to_datetime(stf['last_updated'].copy())
    
    normal = normal.loc[~normal['match_status'].isin(['deleted','collecting','cancelled','postponed'])].reset_index(drop=True)
    stf = stf.loc[~stf['match_status'].isin(['deleted','collecting','cancelled','postponed'])].reset_index(drop=True)
    
    comp_map = {
        121:10,
        231:88,
        125:80,
        226:75,
        274:46,
        280:8,
        292:7,
        119:9,
        295:13,
        130:255,
        1269:104,
        218:106
    }

    normal['is_playoff'] = np.where(normal['competition_id'].isin(list(comp_map.keys())), 1, 0)
    normal['competition_id'] = normal['competition_id'].copy().apply(lambda x: comp_map[x] if x in comp_map else x)
    
    stf['is_playoff'] = np.where(stf['competition_id'].isin(list(comp_map.keys())), 1, 0)
    stf['competition_id'] = stf['competition_id'].copy().apply(lambda x: comp_map[x] if x in comp_map else x)
    
    normal['competition_name'] = normal['competition_id'].apply(lambda x: competitions.get(x)['name'])
    stf['competition_name'] = stf['competition_id'].apply(lambda x: competitions.get(x)['name'])
        
    return normal, stf 

def determine_upcoming_statsbomb():

    schedule, stf_schedule = load_schedules()
    upcoming = schedule.copy().loc[schedule['is_upcoming']==1].reset_index(drop=True)
    
    upcoming['home_team_name'] = upcoming['home_team_id'].apply(lambda x: teams.get(x)['name'])
    upcoming['away_team_name'] = upcoming['away_team_id'].apply(lambda x: teams.get(x)['name'])
    upcoming['competition_name'] = upcoming['competition_id'].apply(lambda x: competitions.get(x)['name'])
    upcoming['country_name'] = upcoming['competition_id'].apply(lambda x: competitions.get(x)['country_name'])
    
    upcoming = upcoming.copy()[['datetime_UTC','match_date_UTC','match_id','competition_id','country_name','competition_name','home_team_id','home_team_name','away_team_id','away_team_name']].reset_index(drop=True)
    upcoming['odds_api_sport_key'] = upcoming['competition_id'].map(comp2oddsapi)
    
    sweet_min_cutoff = datetime.utcnow()+pd.Timedelta(days=0)
    sweet_max_cutoff = datetime.utcnow()+pd.Timedelta(days=15)
    sweet_spot = upcoming.copy().loc[(upcoming['datetime_UTC']>=sweet_min_cutoff)&(upcoming['datetime_UTC']<=sweet_max_cutoff)]
    
    print(f"Statsbomb shows {len(sweet_spot)} games between {sweet_min_cutoff.date()} and {sweet_max_cutoff.date()}")
    # no use in keeping ones that odds api doesn't have
    # every few months might be worth checking if they've added leagues
    sweet_spot = sweet_spot.loc[sweet_spot['odds_api_sport_key'].notnull()].reset_index(drop=True)
    print(f"\n{len(sweet_spot)} of those are in leagues that our odds API carries\n")
    
    print(f"They are in the following competitions: \n")
    print(sweet_spot.groupby(['country_name','competition_name'])['match_id'].count())
        
    return sweet_spot

def determine_successful(odds_df, sb_df, threshold, last_attempt=False):

    teams_sb2oddsapi = load_dict(os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_sb2oddsapi'))
    teams_oddsapi2sb = load_dict(os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_oddsapi2sb'))
    
    successful = odds_df.copy().loc[odds_df['match_score']>=threshold].reset_index(drop=True)
    unsuccessful = odds_df.copy().loc[odds_df['match_score']<threshold].reset_index(drop=True)
    
    ## sometimes two match keys will show the same match, keep the one with the higher score
    successful = successful.sort_values(by='match_score', ascending=False).reset_index(drop=True)
    successful = successful.drop_duplicates(subset=['match_key'], keep='first')
        
    if len(successful)>0:
        for index, row in successful.iterrows():
            
            home_team_name = row['home_team']
            away_team_name = row['away_team']
            sb_match = sb_df.loc[sb_df['match_key']==row['match_name']].iloc[0]
            sb_home = sb_match['home_team_name']
            sb_home_id = sb_match['home_team_id']
            sb_away = sb_match['away_team_name']
            sb_away_id = sb_match['away_team_id']
            if home_team_name in teams_oddsapi2sb:
                assert(teams_oddsapi2sb[home_team_name] == sb_home_id)
            else:
                print(f"Odds api team {home_team_name} matched with SB team {sb_home}")
                teams_oddsapi2sb[home_team_name] = sb_home_id
                teams_sb2oddsapi[sb_home_id] = home_team_name
                save_dict(teams_sb2oddsapi, os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_sb2oddsapi'))
                save_dict(teams_oddsapi2sb, os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_oddsapi2sb'))
            if away_team_name in teams_oddsapi2sb:
                assert(teams_oddsapi2sb[away_team_name] == sb_away_id)
            else:
                print(f"Odds api team {away_team_name} matched with SB team {sb_away}")
                teams_oddsapi2sb[away_team_name] = sb_away_id
                teams_sb2oddsapi[sb_away_id] = away_team_name
                save_dict(teams_sb2oddsapi, os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_sb2oddsapi'))
                save_dict(teams_oddsapi2sb, os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_oddsapi2sb'))
                
                
    if last_attempt==True:
        if len(unsuccessful)>0:
            print(unsuccessful)
            return unsuccessful, True
    
    return unsuccessful, False


def test_name_matches(odds):
    
    upcoming = load_upcoming()
    teams_sb2oddsapi = load_dict(os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_sb2oddsapi'))
    teams_oddsapi2sb = load_dict(os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_oddsapi2sb'))
    
    oa_comps = list(odds['sport_key'].unique())

    for comp in oa_comps:
        comp_odds = odds.copy().loc[odds['sport_key']==comp].reset_index(drop=True)
        comp_sb = upcoming.copy().loc[upcoming['oa_comp_id']==comp].reset_index(drop=True)

        comp_odds['sb_home_team_id'] = comp_odds['home_team'].map(teams_oddsapi2sb)
        comp_odds['sb_away_team_id'] = comp_odds['away_team'].map(teams_oddsapi2sb)
        comp_odds = comp_odds.loc[(comp_odds['sb_home_team_id'].isnull())|(comp_odds['sb_away_team_id'].isnull())].reset_index(drop=True)

        ### uncomment this to see teams that are causing errors
    #     print(comp_odds)

        if len(comp_odds)==0:
            if len(comp_sb.competition_id.unique()) > 0:
                print(f"Competition {competitions.get(comp_sb.competition_id.unique()[0])['name']} has already been successfully name matched")
            continue

        comp_sb = upcoming.copy().loc[upcoming['oa_comp_id']==comp].reset_index(drop=True)
        comp_sb['odds_api_home'] = comp_sb['home_team_id'].map(teams_sb2oddsapi)
        comp_sb['odds_api_away'] = comp_sb['away_team_id'].map(teams_sb2oddsapi)
        comp_sb = comp_sb.loc[(comp_sb['odds_api_home'].isnull())|(comp_sb['odds_api_away'].isnull())].reset_index(drop=True)

        if len(comp_sb)==0:
            print(f"Error, no upcoming SB games for {comp}")
            print(f"Comment block above, and try again, might be wrongly matched (OR, statsbomb simply doesn't have the game)")
            continue


        comp_odds['match_key'] = comp_odds['home_team'].copy()+' '+comp_odds['away_team'].copy()
        comp_sb['match_key'] = comp_sb['home_team_name'].copy()+' ' + comp_sb['away_team_name'].copy()
        possible_matches = list(comp_sb['match_key'].unique())

        def string_matcher(x):
            return process.extract(x, possible_matches, scorer=fuzz.token_set_ratio)[0] # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings

        comp_odds['match_name'],comp_odds['match_score']= zip(*comp_odds.apply(lambda x: string_matcher(x.match_key), axis=1))

        ## three rounds with progressive thresholds
        first_thres = 95
        second_thres = 90
        third_thres = 76
        fourth_thres = 50
        thresholds = [first_thres,second_thres,third_thres, fourth_thres]

        to_match = comp_odds.copy()
        for threshold in thresholds:
            teams_sb2oddsapi = load_dict(os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_sb2oddsapi'))
            teams_oddsapi2sb = load_dict(os.path.join(DROPBOX_PATH, 'IDs/odds_api/teams_oddsapi2sb'))
            if threshold == fourth_thres:
                last_chance = True
            else:
                last_chance = False
            to_match, raise_error = determine_successful(to_match.copy(), comp_sb, threshold, last_chance)
            if raise_error:
                raise ValueError('Some were not matched')
    
    
    return


def data_download():

    jobs = []
    
    p1 = Process(target=update_footy.update_footy)
    jobs.append(p1)
    p1.start()
    p2 = Process(target=download_weather)
    jobs.append(p2)
    p2.start()
    p3 = Process(target=download_Statsbomb_events)
    jobs.append(p3)
    p3.start()
    p4 = Process(target=download_Statsbomb_lineups)
    jobs.append(p4)
    p4.start()

    # checks to see if they are finished
    for job in jobs:
        job.join()
        time.sleep(1)

    return

if __name__=='__main__':

    data_download()




# pass