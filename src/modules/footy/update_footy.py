


import os
import json
import math
import pickle
import urllib
import datetime

import numpy as np
import pandas as pd
import requests as req

from copy import copy
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter
from fuzzywuzzy import fuzz, process

load_dotenv()
FOOTY_KEY = os.environ.get('FOOTY_KEY')
DROPBOX_PATH = os.environ.get('DROPBOX_PATH')

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def get_needs_update():
    
    """
    
    edit leagues at https://footystats.org/api/u/api-settings?area=3&status=1
    
    Don't really have a great way (could potentially create one) to target only leagues going on -- especially 
    since footy schedule is more comprehensive than SB. So just pulling all league from this year. Req limit is
    3600/hour, so no problem there
    
    """
    
    base_url = f'https://api.football-data-api.com/league-list?key={FOOTY_KEY}&chosen_leagues_only=true'
    r = req.get(f"{base_url}{urllib.parse.urlencode('')}")
    leagues = json.loads(r.content)
    leagues = pd.json_normalize(leagues['data'])
    
    needs_update = []
    for index, league in leagues.iterrows():
        league_name = league['name']
        league_years = league['season']
        for year_dict in league_years:
            season_id = year_dict['id']
            season_name = year_dict['year']
            if str(datetime.datetime.now().year) in str(season_name):
                needs_update.append([league_name, season_id, season_name])

    needs_update = pd.DataFrame(needs_update, columns=['league_name','season_id','season_name'])

    return needs_update


### there's a lot more in the raw data, not much I found useful though ###
cols = [
    'id',
    'homeID',
    'awayID',
    'season',
    'status',
    'roundID',
    'date_unix', 'winningTeam', 'no_home_away',
    'game_week',
    'revised_game_week',
    'homeGoalCount',
    'awayGoalCount',
    'attacks_recorded',
    'team_a_yellow_cards',
    'team_b_yellow_cards',
    'team_a_red_cards',
    'team_b_red_cards',
    'team_a_shotsOnTarget',
    'team_b_shotsOnTarget',
    'team_a_shotsOffTarget',
    'team_b_shotsOffTarget',
    'team_a_shots',
    'team_b_shots',
    'team_a_fouls',
    'team_b_fouls',
    'team_a_possession',
    'team_b_possession',
    'team_a_offsides',
    'team_b_offsides',
    'team_a_dangerous_attacks',
    'team_b_dangerous_attacks',
    'team_a_attacks',
    'team_b_attacks',
    'team_a_xg',
    'team_b_xg',
    'total_xg',
    'team_a_penalties_won',
    'team_b_penalties_won',
    'team_a_penalty_goals',
    'team_b_penalty_goals',
    'team_a_penalty_missed',
    'team_b_penalty_missed',
    'team_a_throwins',
    'team_b_throwins',
    'team_a_freekicks',
    'team_b_freekicks',
    'team_a_goalkicks',
    'team_b_goalkicks',
    'refereeID',
    'coach_a_ID',
    'coach_b_ID',
    'stadium_name',
    'stadium_location',
    'odds_ft_1',
    'odds_ft_x',
    'odds_ft_2',
    'attendance',
    'match_url',
    'competition_id',
    'avg_potential', 'home_url', 'home_image', 'home_name', 'away_url', 'away_image', 'away_name'
]


def pull_footy_new(needs_updated):
    
    print("Pulling footy supplementary stats, season by season...")
    
    for index, row in tqdm(needs_updated.iterrows(), total=len(needs_updated)):
#             print(row['name'], s['id'], int(str(s['year'])[:4]))
        base_url = f'https://api.football-data-api.com/league-matches?key={FOOTY_KEY}&league_id=' + str(row['season_id'])
        r = req.get(f"{base_url}{urllib.parse.urlencode('')}")
        data = json.loads(r.content)
        df = pd.json_normalize(data['data'])
        if len(df) <1:
            print(f"ERROR ON: {row}")
            continue
        season = df['season'].unique()[0][:4]
        path = os.path.join(DROPBOX_PATH, f'footy/historical/{season}/{row["league_name"]}.csv')
        df = df[cols].reset_index(drop=True)
        df['competition_name'] = row['league_name']
        path = os.path.join(DROPBOX_PATH, f'footy/historical/{season}/{row["league_name"]}.csv') 
        if '1/2' in path:
            path = path.replace('1/2','1_2') 
        if '2/3' in path:
            path = path.replace('2/3','2_3') 
        if not os.path.isdir(os.path.join(DROPBOX_PATH, f'footy/historical/{season}')):
            os.mkdir(os.path.join(DROPBOX_PATH, f'footy/historical/{season}'))
        df.to_csv(path,index=False)
    
    return


def assemble_footy():
    
    EXT = "*.csv"
    to_parse = os.path.join(DROPBOX_PATH, 'footy/historical')
    existing_footy_files = set([file
                     for path, subdir, files in os.walk(to_parse)
                     for file in glob(os.path.join(path, EXT))])
    
    print("Aggregating historical footy...")
    master = []
    for footy_csv in tqdm(existing_footy_files):
        league_season = pd.read_csv(footy_csv)
        master.append(league_season)
        
    master = pd.concat(master).reset_index(drop=True)
    
    master['datetime_UTC']  = master['date_unix'].apply(lambda x: datetime.datetime.utcfromtimestamp(x))
    master = master.sort_values(by=['datetime_UTC']).reset_index(drop=True)
    master['match_date_UTC'] = master['datetime_UTC'].copy().dt.date

    master.to_csv(os.path.join(DROPBOX_PATH, 'footy/aggregated.csv'), index=False)

    return




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
    
    return normal, stf 

def load_footy():
    
    footy = pd.read_csv(os.path.join(DROPBOX_PATH, 'footy/aggregated.csv'))
    footy['datetime_UTC'] = pd.to_datetime(footy['datetime_UTC'].copy())
    footy['match_date_UTC'] = pd.to_datetime(footy['match_date_UTC'].copy())
    footy = footy.drop_duplicates(subset=['id']).reset_index(drop=True)
    footy = footy.loc[footy['match_date_UTC']<=datetime.datetime.utcnow()+pd.Timedelta(days=14)].reset_index(drop=True)
    footy = footy.sort_values(by=['match_date_UTC','id']).reset_index(drop=True)
    footy = footy[cols+['datetime_UTC','match_date_UTC','competition_name']]
    # one instance of missing competition name
    footy.loc[footy['competition_id']==1, 'competition_name'] = 'USA MLS'
    footy = footy.loc[footy['match_date_UTC']>=pd.to_datetime('2006-08-01')].reset_index(drop=True) # one outlier date
    print(f"Footy API has {len(footy)} matches")
    return footy

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
    
    return normal, stf 

def prep_name_match_comps(footy_, schedule_):
    
    teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))

    # I made these up. Add new ones in the next cell
    footy_id2comp = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/footy_id2comp'))
    footy_comp2id = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/footy_comp2id'))
    
    recent = footy_.copy().loc[footy_['match_date_UTC']>=pd.to_datetime('01-01-2017')].reset_index(drop=True)
    ### prepping footy for matching function
    recent = recent[['competition_name','match_date_UTC','id']].copy().sort_values(by=['id'])

    recent['competition_id'] = recent['competition_name'].map(footy_comp2id)
    recent = recent.rename(columns={
        # for matcher function
        'match_date_UTC':'match_date',
        'id':'match_id',
        'competition_id':'id',
        'competition_name':'name'
    })
    recent['match_date'] = recent['match_date'].dt.date
    recent = recent[['id','name','match_date','match_id']]

    sched = schedule_.copy()
    sched['home_team_name'] = sched['home_team_id'].apply(lambda x: teams.get(x)['name'])
    sched['away_team_name'] = sched['away_team_id'].apply(lambda x: teams.get(x)['name'])

    to_match = sched.copy()[['match_date_UTC','home_team_id','home_team_name','away_team_name']]

    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
    sched['competition_name'] = sched['competition_id'].apply(lambda x: competitions.get(x)['name'])

    sched = sched[['competition_id','competition_name','match_date_UTC','match_id']].copy()
    sched = sched.rename(columns={
        'competition_id':'id',
        'competition_name':'name',
        'match_date_UTC':'match_date'
    })

    # because of footy format this works better
    sched['name'] = sched['id'].apply(lambda x: competitions.get(x)['country_name'] +' '+ competitions.get(x)['name'])

    return sched, recent


def prep_name_match_teams(footy_, schedule_):

    teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
    comp_id2footy = load_dict(os.path.join(DROPBOX_PATH, f'IDs/footy/comp_id2footy'))

    # I made these up. Add new ones in the next cell
    footy_id2comp = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/footy_id2comp'))
    footy_comp2id = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/footy_comp2id'))

    sched = schedule_.copy()
    sched['home_team_name'] = sched['home_team_id'].apply(lambda x: teams.get(x)['name'])
    sched['away_team_name'] = sched['away_team_id'].apply(lambda x: teams.get(x)['name'])

    sched = sched.loc[sched['competition_id'].isin(list(comp_id2footy.keys()))].reset_index(drop=True)

    to_match = sched.copy()[['match_date_UTC','home_team_id','home_team_name','match_id']]

    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
    sched['competition_name'] = sched['competition_id'].apply(lambda x: competitions.get(x)['name'])
    # prep for matcher function
    sched = sched.rename(columns={
        'home_team_id':'id',
        'home_team_name':'name',
        'match_date_UTC':'match_date'
    })

    sched = sched[['id','name','match_date','match_id']]

    recent = footy_.copy().loc[pd.to_datetime(footy_['match_date_UTC'])>=pd.to_datetime('07-01-2017')].reset_index(drop=True)
    recent = recent.copy().loc[pd.to_datetime(recent['match_date_UTC'])<=pd.to_datetime('03-01-2022')].reset_index(drop=True)
    recent['competition_id'] = recent['competition_name'].map(footy_comp2id)
    recent = recent.dropna(subset=['competition_id'])
    ### this causes some trouble, because the mappings are one way
    # recent = recent.loc[recent['competition_id'].isin(list(comp_id2footy.values()))].reset_index(drop=True)
    print(len(recent))
    print(len(recent.loc[recent['competition_id'].isin(list(comp_id2footy.values()))].reset_index(drop=True)))
    ### prepping footy for matching function
    recent = recent[['homeID','home_name','match_date_UTC','id']].copy()

    recent = recent.rename(columns={
        'homeID':'id',
        'home_name':'name',
        'match_date_UTC':'match_date',
        'id':'match_id'
    })
    
    recent['match_date'] = recent['match_date'].dt.date

    return sched, recent


def name_ask(results, top_n):
    results = results.reset_index(drop=True)
    print(results[['id1_name','id2_name','score','match_cosine_score','name_sim_score']])
    print("Are any of these successful matches? Press 1,2,3 (first row is 1)")
    print("Use 0 or None to skip...")
    response = input()
    response = response.strip().lower()
    if (response == "0")|(response=="")|(response=="none"):
        return None, False
    elif not response.isnumeric():
        return None, False
    else:
        idx = int(response) - 1
        if idx >= top_n:
            print("invalid response, continuing")

        else:
            matched_id = results.iloc[idx]['id_2']
            matched_name = results.iloc[idx]['id2_name']

    print(f"Confirm Y/N, {matched_name} == {results.iloc[idx]['id1_name']}?")
    confirmation = input()
    if confirmation.strip().lower() in ['yes','y']:
        return results.iloc[idx]['id_2'], True
    else:
        return None, False
    
    return



def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def get_counters(df):
    
    match_count = df.copy().groupby(['id','match_date'])['match_id'].count().reset_index()
    # creating counter input
    match_count = match_count.pivot(index='id', columns=['match_date'], values='match_id')
    count_dict = {}
    for index, row in match_count.iterrows():
        dates = row.dropna().to_dict()
        count_dict[index] = dates
    
    return count_dict



def matcher(df1, df2, existing=None, auto=0.75, ask=0.35, proportion=0.7, save_path=None):
    
    print("Team name matching for footy...")
    
    """
    
    Takes in two dataframes with columns [id, name, match_date, match_id] and matches
    If above auto threshold, automatically returns them as match
    If above ask threshold, prompts user if they are a match or not
    Proportion controls how much weight goes to date matcher and how much goes to name matcher
    
    """
    if existing is None:
        translate_dict = {}
    else:
        translate_dict = copy(existing)
        
        # there are a couple I matched outside that aren't in "needed" comps
        # so adding here to not give > 100% matched
        needed = set(df1.id.unique())
        existing = set(existing)
        needed.update(existing)
        
        needed_len = len(needed)
        existing_len = len(existing)
        
        print(needed.difference(existing))
        print(f"{np.round(existing_len/needed_len, 4)*100}% of matches have been found")
        if len(needed.difference(existing))==0:
            return translate_dict, []
        else:
            print("Matching teams, this will take a minute or two...")
    
    df1_counter = get_counters(df1.copy())
    df2_counter = get_counters(df2.copy())

    df1_id2name = df1.drop_duplicates(subset='id').set_index('id').to_dict()['name']
    df2_id2name = df2.drop_duplicates(subset='id').set_index('id').to_dict()['name']

    all_matches = []
    failed_matches_ids = []
    failed_matches_names = []
    # append top n matches 
    i = 0
    top_n = 3
    for id_1, matches_1 in df1_counter.items():
        id1_counter = Counter(matches_1)
        id1_name = df1_id2name[id_1]

        top_matches = []
        for id_2, matches_2 in df2_counter.items():
            id2_counter = Counter(matches_2)
            id2_name = df2_id2name[id_2]
            match_cosine_score = counter_cosine_similarity(id1_counter ,id2_counter)
            name_sim_score = fuzz.ratio(id1_name, id2_name)
            top_matches.append([id_1, id1_name, id_2, id2_name, match_cosine_score, name_sim_score])
        top_matches = pd.DataFrame(top_matches, columns=['id_1', 'id1_name', 'id_2', 'id2_name', 'match_cosine_score', 'name_sim_score'])
        top_matches['score'] = (top_matches['match_cosine_score'].copy()+(top_matches['name_sim_score'].copy()/100))/2
        top_matches = top_matches.sort_values(by=['score'], ascending=False)
        all_matches.append(top_matches.head(top_n))

    all_matches = pd.concat(all_matches)
    all_matches = all_matches.sort_values(by=['score'], ascending=False)

    for index, row in all_matches.iterrows():
        
        if row['id_1'] in failed_matches_ids:
            print(f"{row['id1_name']} already failed")
            continue
        
        if row['id_1'] not in translate_dict:
            if row['score'] > auto:
                translate_dict[row['id_1']] = row['id_2']
            elif row['name_sim_score'] > 0.95:
                translate_dict[row['id_1']] = row['id_2']
            elif row['score'] > ask:
                id_match, success = name_ask(all_matches.loc[all_matches['id_1']==row['id_1']].reset_index(drop=True), top_n)
                if success:
                    translate_dict[row['id_1']] = id_match
                    if save_path is not None:
                        save_dict(translate_dict, save_path)
                else:
                    print(f"{row['id1_name']} failed because match could not be found")
                    failed_matches_ids.append(row['id_1'])
                    failed_matches_names.append(row['id1_name'])
                    continue
            else:
                if (row['id_1'] not in failed_matches_ids):
                    print(f"{row['id1_name']} failed because not above ask or auto threshold")
                    failed_matches_ids.append(row['id_1'])
                    failed_matches_names.append(row['id1_name'])
                continue
        else:
            print(f"{row['id1_name']} already matched")
            continue

    return translate_dict, failed_matches_names


def name_match():

    schedule, stf_schedule = load_schedules()
    footy = load_footy()

    sched, recent = prep_name_match_comps(footy.copy(), schedule.copy())

    auto=0.75
    ask=0.45
    top_n = 3
    proportion=0.7

    df1 = sched.copy()
    df2 = recent.copy()
    save_path = os.path.join(DROPBOX_PATH, f'IDs/footy/comp_id2footy')

    # match comps
    existing = load_dict(save_path)

    df1 = df1.drop_duplicates(subset=['id']).reset_index(drop=True)
    df2 = df2.drop_duplicates(subset=['id']).reset_index(drop=True)
    print("Unmatched ids in df1: \n",
        {(row['id'], row['name']) for index, row in df1.iterrows() if row['id'] not in existing})
    print("Unmatched ids in df2: \n",
        {(row['id'], row['name']) for index, row in df2.iterrows() if row['id'] not in existing.values()})

    matcher(df1, df2, existing=existing, save_path=save_path, auto=auto, ask=ask, proportion=proportion)

    # now do teams
    sched, recent = prep_name_match_teams(footy.copy(), schedule.copy())

    df1 = sched.copy()
    df2 = recent.copy()

    teams_id2footy = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_id2footy'))
    matcher(df1, df2, auto=auto, ask=ask, proportion=proportion, existing=teams_id2footy, save_path=os.path.join(DROPBOX_PATH, 'IDs/footy/teams_id2footy'))

    return


def update_footy():
    needs_update = get_needs_update()
    pull_footy_new(needs_update)
    assemble_footy()

    # name_match()

    return






## pass