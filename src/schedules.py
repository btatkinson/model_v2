

""""
Initial step of pipeline
Downloads and processes Statsbomb schedule, which everything else is based on 
Updates Statsbomb IDs
Also creates single team version of schedule (stf_schedule)
"""


import gc
import io
import os
import json
import time
import pickle
import requests

import numpy as np
import pandas as pd
import statsbombpy.entities as ents

from tqdm import tqdm
from statsbombpy import sb
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()
SB_USERNAME = os.environ.get('SB_USERNAME')
SB_PASSWORD = os.environ.get('SB_PASSWORD')
DROPBOX_PATH = os.environ.get('DROPBOX_PATH')
creds={"user":SB_USERNAME,"passwd":SB_PASSWORD}
VERSION = 'v5'
HOSTNAME = 'https://data.statsbombservices.com/'
AWSAccessKeyId=os.environ.get('AWS_ACCESS_KEY_ID')
AWSSecretKey=os.environ.get('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME=os.environ.get('BUCKET_NAME')


#################
###  Helpers  ###
#################

def EST_to_UTC(time):
    return time + pd.Timedelta(hours=5)

def statsbomb_to_UTC(time):
    return time - pd.Timedelta(hours=1)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def get_resource(url: str, creds: dict) -> list:
    """
    re-wrote a Statsbomb function to pull future matches
    """
    auth = requests.auth.HTTPBasicAuth(creds["user"], creds["passwd"])
    resp = requests.get(url, auth=auth)

    if resp.status_code != 200:
        print(f"{url} -> {resp.status_code}")
        resp = []
    else:
        resp = resp.json()
    return resp

def matches(competition_id: int, season_id: int, creds: dict, upcoming=False) -> dict:
    url = f"{HOSTNAME}api/{VERSION}/competitions/{competition_id}/seasons/{season_id}/matches"
    matches = get_resource(url, creds)
    if not upcoming:
        matches = ents.matches(matches)
    return matches

#############################
###  Schedule Processing  ###
#############################

def process_comps(comp_df):
    
    
    comp_df['match_updated'] = pd.to_datetime(comp_df['match_updated'])
    comp_df['match_available'] = pd.to_datetime(comp_df['match_available'])
    
    comp_df = comp_df.rename(columns={
        'match_updated':'match_updated_UTC',
        'match_available':'match_available_UTC'
    })
    
    return comp_df


def handle_coaches(coaches):
    """
    Sometimes they also throw in an assistant coach which is a nightmare
    """
    
    managers = []
    for i in range(len(coaches)):
        if type(coaches[i]) == float:
            if np.isnan(coaches[i]):
                managers.append(pd.DataFrame([np.empty(6)], columns=['id', 'name', 'nickname', 'dob', 'country.id', 'country.name']))
        elif len(coaches[i])>1:
            ### handle
            handled = False
            min_id = int(np.min([coach['id'] for coach in coaches[i]])) # lowest id is usually the actual coach, otherwise very tough to handle
            for coach in coaches[i]:
                if coach['id'] == min_id:
                    managers.append(pd.json_normalize(coach))
                    handled = True
            if not handled:
                print(coaches[i])
                raise ValueError("More than one coaches, you have to determine the head coach")
        else:
            managers.append(pd.json_normalize(coaches[i]))
    
    return pd.concat(managers).reset_index(drop=True)

def preprocessing(league_sched):
    # occasionally they give us an empty result
    if len(league_sched) == 0:
        return pd.DataFrame()
    # turn into pandas dataframe
    league_sched = pd.concat([pd.json_normalize(match) for match in league_sched],axis=0).reset_index(drop=True)
    
    # change to more convenient column names
    orig_cols = list(league_sched)
    column_mapping = {
        'competition.competition_id':'competition_id',
        'competition.country_name':'country_name',
        'competition.competition_name':'competition_name',
        'season.season_id':'season_id',
        'season.season_name':'season_name',
        'home_team.home_team_id':'home_team_id',
        'home_team.home_team_name':'home_team_name',
        'home_team.home_team_gender':'home_team_gender',
        'home_team.home_team_youth':'home_team_youth',
        'home_team.home_team_group':'home_team_group',
        'home_team.country.id':'home_team_country_id',
        'home_team.country.name':'home_team_country_name',
        'home_team.managers':'home_team_managers',
        'away_team.away_team_id':'away_team_id',
        'away_team.away_team_name':'away_team_name',
        'away_team.away_team_gender':'away_team_gender',
        'away_team.away_team_youth':'away_team_youth',
        'away_team.away_team_group':'away_team_group',
        'away_team.country.id':'away_team_country_id',
        'away_team.country.name':'away_team_country_name',
        'away_team.managers':'away_team_managers', 
    }
    cols = [column_mapping[oc] if oc in column_mapping else oc for oc in orig_cols]
    cols = [oc.replace('.', '_') for oc in cols]
    league_sched.columns=cols
    
    if 'away_team_managers' in league_sched:
        away_coaches = list(league_sched['away_team_managers'].values)
        away_coaches = handle_coaches(away_coaches)
        if 'home_team_managers' in league_sched:
            home_coaches = list(league_sched['home_team_managers'].values)
            home_coaches = handle_coaches(home_coaches)
            away_coaches.columns=['away_manager_id','away_manager_name','away_manager_nickname','away_manager_dob','away_manager_country_id','away_manager_country_name']
            home_coaches.columns=['home_manager_id','home_manager_name','home_manager_nickname','home_manager_dob','home_manager_country_id','home_manager_country_name']
            coaches = pd.concat([home_coaches, away_coaches], axis=1)
            assert(len(coaches)==len(league_sched))
            league_sched = pd.concat([league_sched, coaches], axis=1)
            league_sched = league_sched.drop(columns=['away_team_managers','home_team_managers'])
        else:
            ### rare case, handle as if no coaches
            # consistent columns
            for col in ['away_manager_id','away_manager_name','away_manager_nickname','away_manager_dob','away_manager_country_id','away_manager_country_name']:
                league_sched[col] = np.nan

    else:
        # consistent columns
        for col in ['away_manager_id','away_manager_name','away_manager_nickname','away_manager_dob','away_manager_country_id','away_manager_country_name']:
            league_sched[col] = np.nan
    
    league_sched['kick_off'] = league_sched['kick_off'].fillna('16:00:00')
    league_sched['datetime_UTC'] = pd.to_datetime(league_sched['match_date'].copy() + ' ' + league_sched['kick_off']) - pd.Timedelta(hours=1)
    for col in ['home_team_managers','away_team_managers']:
        if col in list(league_sched):
            league_sched = league_sched.drop(columns=[col])
    return league_sched.sort_values(by=['datetime_UTC','match_id']).reset_index(drop=True)


####################################
###  Primary Schedule Functions  ###
####################################


def update_schedule(all_comps):
    """
    updates raw data from historical schedules
    """
    print("Pulling new statsbomb matches...")
    match_lists = []
    for index, row in tqdm(all_comps.iterrows(), total=len(all_comps)):
        comp_id = row['competition_id']
        season_id = row['season_id']
        match_list = matches(competition_id=comp_id, season_id=season_id, creds=creds, upcoming=True)
        match_list = preprocessing(match_list)
        if len(match_list)==0:
            continue
        match_lists.append(match_list)
    match_lists = pd.concat(match_lists).reset_index(drop=True)
    return match_lists

def raw_preprocessing(aggregated):
    ## fill in nans with competition's most common kickoff 
    aggregated["kick_off"] = aggregated['kick_off'].fillna(aggregated.groupby('competition_id')['kick_off'].transform(pd.Series.mode))
    ## otherwise a common kick for all comps
    aggregated["kick_off"] = aggregated["kick_off"].fillna("18:00:00")
    aggregated['datetime_UTC'] = pd.to_datetime(aggregated['match_date'].copy() + ' ' + aggregated['kick_off']) - pd.Timedelta(hours=1)
    aggregated = aggregated.sort_values(by=['datetime_UTC','match_id']).reset_index(drop=True)
    
    for col in ['home_team_managers','away_team_managers']:
        if col in list(aggregated):
            aggregated = aggregated.drop(columns=[col])
    
    return aggregated


def main_raw_schedule():

    print("Grabbing existing matches from statsbomb...")
    LAST_RUN_UTC = load_dict(os.path.join(DROPBOX_PATH, 'logging/script_times'))['LAST_SCHEDULE_RUN']
    comps = sb.competitions(creds=creds)
    comps = process_comps(comps)
    needs_update = comps.copy().loc[(comps['match_updated_UTC']>LAST_RUN_UTC)|(comps['match_available_UTC']>LAST_RUN_UTC)].reset_index(drop=True)

    if len(needs_update) > 0:
        # to download all historical, just use "comps" instead of "needs_update"
        updated = update_schedule(needs_update)
        raw_old = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/historical/raw_historical.csv'))
        assert(len(set(list(updated)).symmetric_difference(set(list(raw_old))))==0)
        raw_old['datetime_UTC'] = pd.to_datetime(raw_old['datetime_UTC'])
        raw_schedule = pd.concat([raw_old.copy(), updated.copy()], axis=0).drop_duplicates(subset='match_id', keep='last').sort_values(by=['datetime_UTC','match_id']).reset_index(drop=True)
        raw_schedule = raw_preprocessing(raw_schedule)
        print("Done! uploading raw history...")
        needs_update = True
    else:
        print("No new events found...")
        raw_schedule = None
        needs_update = False
    return raw_schedule, needs_update

#############################
###  Update ID Functions  ###
#############################

def update_stadium_ids(stadiums):
    
    stadium_ids = load_dict(os.path.join(DROPBOX_PATH, 'IDs/stadiums'))
#     stadium_ids = {} #if you want to reset
    stadiums = stadiums.dropna(subset=['stadium_name'])
    stadiums = stadiums.drop_duplicates(subset='stadium_id').reset_index(drop=True)
    ids = set(stadiums.stadium_id.unique())
    existing_stad = set(stadium_ids.keys())
    new_stadium_ids = ids.difference(existing_stad)
    stadiums = stadiums.loc[stadiums['stadium_id'].isin(new_stadium_ids)].reset_index(drop=True)
    stadiums = stadiums.set_index('stadium_id')
    if len(new_stadium_ids) == 0:
        pass
    else:
        for stadium in new_stadium_ids:
            stad_info = stadiums.loc[stadium]
            stadium_ids[int(stadium)] = {}
            stadium_ids[int(stadium)]['name'] = stad_info['stadium_name']
            stadium_ids[int(stadium)]['country_id'] = stad_info['stadium_country_id']
            stadium_ids[int(stadium)]['country_name'] = stad_info['stadium_country_name']
            save_dict(stadium_ids, os.path.join(DROPBOX_PATH, 'IDs/stadiums'))
            print(f"New stadium! {stad_info}")
    
    return

def update_manager_ids(managers):

    manager_ids = load_dict(os.path.join(DROPBOX_PATH, 'IDs/managers'))
#     manager_ids = {} #if you want to reset
    managers = managers.dropna(subset=['manager_name'])
    managers = managers.drop_duplicates(subset='manager_id').reset_index(drop=True)
    ids = set(managers.manager_id.unique())
    existing_coaches = set(manager_ids.keys())
    new_manager_ids = ids.difference(existing_coaches)
    managers = managers.loc[managers['manager_id'].isin(new_manager_ids)].reset_index(drop=True)
    managers = managers.set_index('manager_id')
    if len(new_manager_ids) == 0:
        pass
    else:
        for manager in new_manager_ids:
            coach_info = managers.loc[manager]
            manager_ids[int(manager)] = {}
            manager_ids[int(manager)]['name'] = coach_info['manager_name']
            manager_ids[int(manager)]['nickname'] = coach_info['manager_nickname']
            manager_ids[int(manager)]['dob'] = coach_info['manager_dob']
            manager_ids[int(manager)]['country_id'] = coach_info['manager_country_id']
            manager_ids[int(manager)]['country_name'] = coach_info['manager_country_name']
            save_dict(manager_ids, os.path.join(DROPBOX_PATH, 'IDs/managers'))
            print(f"New manager! {coach_info}")
    
    return

def update_referee_ids(referees):

    referee_ids = load_dict(os.path.join(DROPBOX_PATH, 'IDs/referees'))
#     referee_ids = {} #if you want to reset
    referees = referees.dropna(subset=['referee_name'])
    referees = referees.drop_duplicates(subset='referee_id').reset_index(drop=True)
    ids = set(referees.referee_id.unique())
    existing_refs = set(referee_ids.keys())
    new_referee_ids = ids.difference(existing_refs)
    referees = referees.loc[referees['referee_id'].isin(new_referee_ids)].reset_index(drop=True)
    referees = referees.set_index('referee_id')
    if len(new_referee_ids) == 0:
        pass
    else:
        for referee in new_referee_ids:
            ref_info = referees.loc[referee]
            referee_ids[int(referee)] = {}
            referee_ids[int(referee)]['name'] = ref_info['referee_name']
            referee_ids[int(referee)]['country_id'] = ref_info['referee_country_id']
            referee_ids[int(referee)]['country_name'] = ref_info['referee_country_name']
            save_dict(referee_ids, os.path.join(DROPBOX_PATH, 'IDs/referees'))
            print(f"New referee! {ref_info}")
    
    return

def update_team_ids(teams):

    team_ids = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
#     team_ids = {} #if you want to reset
    teams = teams.dropna(subset=['team_name'])
    teams = teams.drop_duplicates(subset='team_id').reset_index(drop=True)
    ids = set(teams.team_id.unique())
    existing_tms = set(team_ids.keys())
    new_team_ids = ids.difference(existing_tms)
    teams = teams.loc[teams['team_id'].isin(new_team_ids)].reset_index(drop=True)
    teams = teams.set_index('team_id')
    if len(new_team_ids) == 0:
        pass
    else:
        for team in new_team_ids:
            tm_info = teams.loc[team]
            team_ids[int(team)] = {}
            team_ids[int(team)]['name'] = tm_info['team_name']
            team_ids[int(team)]['gender'] = tm_info['team_gender']
            team_ids[int(team)]['youth'] = tm_info['team_youth']
            team_ids[int(team)]['group'] = tm_info['team_group']
            team_ids[int(team)]['country_id'] = tm_info['team_country_id']
            team_ids[int(team)]['country_name'] = tm_info['team_country_name']
            save_dict(team_ids, os.path.join(DROPBOX_PATH, 'IDs/teams'))
            print(f"New team! {tm_info}")
    
    return

def update_season_ids(seasons):

    season_ids = load_dict(os.path.join(DROPBOX_PATH, 'IDs/seasons'))
#     season_ids = {} #if you want to reset
    seasons = seasons.dropna(subset=['season_name'])
    seasons = seasons.drop_duplicates(subset='season_id').reset_index(drop=True)
    ids = set(seasons.season_id.unique())
    existing_szns = set(season_ids.keys())
    new_season_ids = ids.difference(existing_szns)
    seasons = seasons.loc[seasons['season_id'].isin(new_season_ids)].reset_index(drop=True)
    seasons = seasons.set_index('season_id')
    if len(new_season_ids) == 0:
        pass
    else:
        for season in new_season_ids:
            szn_info = seasons.loc[season]
            season_ids[int(season)] = {}
            season_ids[int(season)]['name'] = szn_info['season_name']
            save_dict(season_ids, os.path.join(DROPBOX_PATH, 'IDs/seasons'))
            print(f"New season! {szn_info}")
    
    return

def update_competition_ids(competitions):
    competition_ids = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
#     competition_ids = {} #if you want to reset
    competitions = competitions.dropna(subset=['country_name','competition_name'])
    competitions = competitions.drop_duplicates(subset='competition_id').reset_index(drop=True)
    ids = set(competitions.competition_id.unique())
    existing_comps = set(competition_ids.keys())
    new_competition_ids = ids.difference(existing_comps)
    competitions = competitions.loc[competitions['competition_id'].isin(new_competition_ids)].reset_index(drop=True)
    competitions = competitions.set_index('competition_id')
    if len(new_competition_ids) == 0:
        pass
    else:
        for competition in new_competition_ids:
            comp_info = competitions.loc[competition]
            competition_ids[int(competition)] = {}
            competition_ids[int(competition)]['country_name'] = comp_info['country_name']
            competition_ids[int(competition)]['name'] = comp_info['competition_name']
            save_dict(competition_ids, os.path.join(DROPBOX_PATH, 'IDs/competitions'))
            print(f"New competition! {comp_info}")
    
    return

def update_ids(rsch):
    print("Updating statsbomb ids..")
    competitions = rsch.copy()[['competition_id', 'country_name', 'competition_name']]
    competitions['competition_id'] = competitions['competition_id'].astype('Int64')
    seasons = rsch.copy()[['season_id', 'season_name']]
    seasons['season_id'] = seasons['season_id'].astype('Int64')
    referees = rsch.copy()[['referee_id', 'referee_name', 'referee_country_id', 'referee_country_name']]
    referees['referee_id'] = referees['referee_id'].astype('Int64')
    referees['referee_country_id'] = referees['referee_country_id'].astype('Int64')
    stadiums = rsch.copy()[['stadium_id', 'stadium_name', 'stadium_country_id', 'stadium_country_name']]
    stadiums['stadium_id'] = stadiums['stadium_id'].astype('Int64')
    stadiums['stadium_country_id'] = stadiums['stadium_country_id'].astype('Int64')

    home_managers = rsch.copy()[['home_manager_id', 'home_manager_name', 'home_manager_nickname', 'home_manager_dob', 'home_manager_country_id', 'home_manager_country_name']]
    away_managers = rsch.copy()[['away_manager_id', 'away_manager_name', 'away_manager_nickname', 'away_manager_dob', 'away_manager_country_id', 'away_manager_country_name']]
    home_managers.columns=['manager_id','manager_name','manager_nickname','manager_dob','manager_country_id','manager_country_name']
    away_managers.columns=['manager_id','manager_name','manager_nickname','manager_dob','manager_country_id','manager_country_name']
    managers = pd.concat([home_managers.copy(), away_managers.copy()], axis=0).reset_index(drop=True)
    # had some errors of decimal ids
    managers['manager_id'] = np.floor(pd.to_numeric(managers['manager_id'], errors='coerce')).astype('Int64')
    managers['manager_country_id'] = np.floor(pd.to_numeric(managers['manager_country_id'], errors='coerce')).astype('Int64')
    
    del home_managers
    del away_managers
    gc.collect()

    home_teams = rsch.copy()[['home_team_id', 'home_team_name', 'home_team_gender', 'home_team_youth', 'home_team_group', 'home_team_country_id', 'home_team_country_name']]
    away_teams = rsch.copy()[['away_team_id', 'away_team_name', 'away_team_gender', 'away_team_youth', 'away_team_group', 'away_team_country_id', 'away_team_country_name']]
    home_teams.columns=['team_id','team_name','team_gender','team_youth','team_group','team_country_id','team_country_name']
    away_teams.columns=['team_id','team_name','team_gender','team_youth','team_group','team_country_id','team_country_name']
    teams = pd.concat([home_teams.copy(), away_teams.copy()], axis=0).reset_index(drop=True)
    teams['team_id'] = teams['team_id'].astype('Int64')
    teams['team_country_id'] = teams['team_country_id'].astype('Int64')

    del home_teams
    del away_teams
    gc.collect()
    
    update_competition_ids(competitions)
    update_season_ids(seasons)
    update_team_ids(teams)
    update_referee_ids(referees)
    update_manager_ids(managers)
    update_stadium_ids(stadiums)
    print("Done!")
    
    return

def process_schedule(rsch):

    # fix some statsbomb errors (using the same stadium id in Japan and US)
    rsch['stadium_id'] = np.where((rsch['stadium_name']=='Nissan Stadium')&(rsch['competition_id']==44), 1000259, rsch['stadium_id'].copy())
    rsch['stadium_id'] = np.where((rsch['stadium_name']=='Nissan Stadium')&(rsch['competition_id']==108), 4935, rsch['stadium_id'].copy())
    rsch['stadium_id'] = np.where((rsch['stadium_name']=='Toyota Stadium')&(rsch['competition_id']==44), 4681, rsch['stadium_id'].copy())
    rsch['stadium_id'] = np.where((rsch['stadium_name']=='Toyota Stadium')&(rsch['competition_id']==108), 642, rsch['stadium_id'].copy())
    
    # columns that are stored in ID keys if needed
    rsch = rsch.drop(columns=[col for col in list(rsch) if 'name' in col])
    rsch = rsch.drop(columns=[col for col in list(rsch) if 'gender' in col])
    rsch = rsch.drop(columns=[col for col in list(rsch) if 'youth' in col])
    rsch = rsch.drop(columns=[col for col in list(rsch) if 'country' in col])
    rsch = rsch.drop(columns=[col for col in list(rsch) if 'dob' in col])
    rsch = rsch.drop(columns=[col for col in list(rsch) if 'group' in col])
    
    # columns that are not applicable
    # na_cols = ['match_status_360','last_updated_360', 'metadata_data_version','metadata_shot_fidelity_version','metadata_xy_fidelity_version']
    # rsch = rsch.drop(columns=na_cols)

    # redundant with datetime
    rsch = rsch.drop(columns=['match_date','kick_off']) 
    
    # make sure ids are integers
    for col in list(rsch):
        if '_id' in col:
            # have to floor it because some of the floats were causing issues with precision
            rsch[col] = np.floor(pd.to_numeric(rsch[col], errors='coerce')).astype('Int64')
    
    # drop games more than two weeks out, don't anticipate needed them for any reason
    cutoff = EST_to_UTC(datetime.now()+pd.Timedelta(days=14))
    print(f"CUTOFF, {cutoff}")
    rsch = rsch.loc[rsch['datetime_UTC']<=cutoff].copy().reset_index(drop=True)
    
    # add indicator for upcoming
    rsch['is_upcoming'] = np.where(rsch['datetime_UTC']>EST_to_UTC(datetime.now()),1,0)
    
    # add indicators for international play
    rsch['is_intlc'] = np.where(rsch['competition_id'].isin([16,35,90,102,273,353]), 1, 0) # struggled whether to include England League Cup, 90
    rsch['is_intl'] = np.where(rsch['competition_id'].isin([254,255,256,257,259,55,92,43]), 1, 0)

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
        218:106,
        1259:109,
        1426:108,
        1256:249
    }

    rsch['is_playoff'] = np.where(rsch['competition_id'].isin(list(comp_map.keys())), 1, 0)
    rsch['total'] = rsch['home_score'].copy()+rsch['away_score'].copy()
    rsch['competition_id'] = rsch['competition_id'].copy().apply(lambda x: comp_map[x] if x in comp_map else x)

    
    return rsch


def create_stf(psch):
    print("Converting processed schedule to STF schedule and uploading...")
    
    psch['competition_game_number'] = psch.groupby(['competition_id'])['datetime_UTC'].transform(lambda x: x.rank(method='dense').astype(int))
    homes = psch.copy()
    aways = psch.copy()
    homes.columns = [col.replace('home_','') for col in list(homes)]
    homes.columns = [col.replace('away_','opp_') for col in list(homes)]
    homes['is_home'] = 1
    aways.columns = [col.replace('away_','') for col in list(aways)]
    aways.columns = [col.replace('home_','opp_') for col in list(aways)]
    aways['is_home'] = 0
    stf = pd.concat([homes, aways], axis=0).sort_values(by=['datetime_UTC','match_id','team_id']).reset_index(drop=True)
    stf['team_game_number'] = stf.groupby(['team_id'])['datetime_UTC'].transform(lambda x: x.rank(method='dense'))
    stf['season_number'] = stf.groupby(['competition_id'])['season_id'].transform(lambda x: x.rank(method='dense'))

    # only use games where both teams have played >= 12 and at least second season
    stf['is_keep'] = np.where(((stf['season_number']>=2)&\
                                    (stf['competition_game_number']>=100)&
                                    (stf['team_game_number']>=12)), 1,0)
                                   #(combined['is_intl']==0)&
                                   #(combined['is_intlc']==0))

    # for any match id that doesn't pass the test, make sure the other version of it doesn't either
    invalid_match_ids = list(stf.copy().loc[stf['is_keep']==0].is_keep.unique())
    stf['is_keep'] = np.where(stf['match_id'].isin(invalid_match_ids), 0, stf['is_keep'].copy())
    
    stf['team_game_number'] = stf['team_game_number'].astype(int)
    stf['season_number'] = stf['season_number'].astype(int)
    
    return stf


def schedules():

    raw_schedule, needs_update = main_raw_schedule()

    if needs_update:
        raw_schedule.to_csv(os.path.join(DROPBOX_PATH, 'schedules/historical/raw_historical.csv'),index=False)
        last_run = {
            'LAST_SCHEDULE_RUN':EST_to_UTC(datetime.now())
        }
        save_dict(last_run, os.path.join(DROPBOX_PATH, 'logging/script_times'))

        update_ids(raw_schedule)
        processed_schedule = process_schedule(raw_schedule.copy())
        stf_schedule = create_stf(processed_schedule.copy())
        # steal is_keep from stf
        processed_schedule = processed_schedule.merge(stf_schedule.drop_duplicates(subset='match_id').reset_index(drop=True)[['match_id','is_keep']], how='left', on='match_id')
        for id_ in ['competition_id','season_id','team_id']:
            stf_schedule[id_] = stf_schedule[id_].astype(int)
            if id_ == 'team_id':
                continue # not needed for non-stf
            processed_schedule[id_] = processed_schedule[id_].astype(int)
        stf_schedule.to_csv(os.path.join(DROPBOX_PATH, 'schedules/stf_schedule.csv'),index=False)
        processed_schedule.to_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'),index=False)
        
    return





