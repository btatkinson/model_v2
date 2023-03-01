
"""

figures out which files need to be converted to stf, atomic sparse, etc
then executes those processes

"""


import os
import json
import time
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv
from multiprocessing import Process

from src.modules.statsbomb.json_to_csv import json_to_csv
from src.modules.statsbomb.csv_to_stf import csv_to_stf
from src.modules.statsbomb.stf_to_gvec import STF2gvec

load_dotenv()
DROPBOX_PATH = os.environ.get('DROPBOX_PATH')

### HELPERS ###

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


def make_folder(base_path, comp_id, season_id):
    
    base_path = os.path.join(DROPBOX_PATH, base_path)
    if not os.path.isdir(os.path.join(base_path, f"{comp_id}")):
        print(f"New competition id, {comp_id}")
        os.mkdir(os.path.join(base_path, f"{comp_id}"))
    if not os.path.isdir(os.path.join(base_path, f"{comp_id}/{season_id}")):
        print(f"New season id ( {season_id} ) in competition id, {comp_id}")
        os.mkdir(os.path.join(base_path, f"{comp_id}/{season_id}"))
    
    return


### PROCESSING FUNCTIONS ### 

def initialize_checklist(sched):
    
    # sb_data_cols = ['raw_events','lineups','atomic_sparse','stf']
    # external_data_cols = ['odds','weather','stadiums','transfermarket']

    cklst = sched.copy()
    cklst = cklst[['datetime_UTC','competition_id','season_id','match_status','last_updated','is_upcoming','match_id','home_team_id','away_team_id']]
    cklst['datetime_UTC'] = pd.to_datetime(cklst['datetime_UTC'])
    cklst['last_updated'] = pd.to_datetime(cklst['last_updated'])
    cklst['last_updated_UTC'] = cklst['last_updated'].apply(lambda x: statsbomb_to_UTC(x))
    cklst = cklst.drop(columns=['last_updated'])
    
    return cklst

def initialize_stf_checklist(stf_sched):
    
    """
    Need separate checklist for team specific data
    """
#     sb_data_cols = ['raw_events','lineups','atomic_sparse','stf']
#     external_data_cols = ['odds','weather','stadiums','transfermarket']

    cklst = stf_sched.copy()
    cklst = cklst[['datetime_UTC','competition_id','season_id','match_status','last_updated','is_upcoming','match_id','team_id']]
    cklst['datetime_UTC'] = pd.to_datetime(cklst['datetime_UTC'])
    cklst['last_updated'] = pd.to_datetime(cklst['last_updated'])
    cklst['last_updated_UTC'] = cklst['last_updated'].apply(lambda x: statsbomb_to_UTC(x))
    cklst = cklst.drop(columns=['last_updated'])
    
    return cklst




def atomic_sparse_process(cklst):
    
    print("Converting raw json to csv format...")
    ### check if events file exists
    cklst['events_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/raw/events/{x.competition_id}/{x.season_id}/{x.match_id}.json'), axis=1)
    # see if its a new game that doesn't exist yet
    cklst['events'] = cklst.apply(lambda x: os.path.exists(x.events_path), axis=1)
    # see if its a game that has been updated recently
    cklst['events_last_modified_UTC'] = cklst['events_path'].apply(lambda x: EST_to_UTC(pd.to_datetime(time.ctime(os.path.getmtime(x)))) if os.path.exists(x) else np.nan)
    cklst['events'] = np.where(cklst['last_updated_UTC']>cklst['events_last_modified_UTC'], False, cklst['events'].copy())
    # see if its an empty event file
    cklst['events_sizeof'] = cklst['events_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    cklst['events'] = np.where(cklst['events_sizeof'].copy()<10, False, cklst['events'].copy())
    # some matches just aren't available
    cklst['events'] = np.where(cklst['match_status']!='available', True, cklst['events'])

    cklst['atomic_sparse'] = False # whether or not it exists & is updated & isn't empty
    cklst['as_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/atomic_sparse/{x.competition_id}/{x.season_id}/{x.match_id}.csv'), axis=1)
    # see if its a new game that doesn't exist yet
    cklst['atomic_sparse'] = cklst.apply(lambda x: os.path.exists(x.as_path), axis=1)
    # see if its a game that has been updated recently
    cklst['as_last_modified_UTC'] = cklst['as_path'].apply(lambda x: EST_to_UTC(pd.to_datetime(time.ctime(os.path.getmtime(x)))) if os.path.exists(x) else np.nan)
    cklst['atomic_sparse'] = np.where(cklst['last_updated_UTC']>cklst['as_last_modified_UTC'], False, cklst['atomic_sparse'].copy())
    # see if its an empty atomic_sparse file
    cklst['as_sizeof'] = cklst['as_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    cklst['atomic_sparse'] = np.where(cklst['as_sizeof'].copy()<100, False, cklst['atomic_sparse'].copy())
    # some matches just aren't available
    cklst['atomic_sparse'] = np.where(cklst['match_status']!='available', True, cklst['atomic_sparse'])

    # more efficient would be to update it in the previous function as events are downloaded
    cklst['events_sizeof'] = cklst['events_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    as_to_grab = cklst.copy().loc[(cklst['is_upcoming']==0)&(cklst['atomic_sparse']==False)].reset_index(drop=True)
    print(f"Trying to convert {len(as_to_grab)} raw json to csv")
    no_obv = []
    no_json = []
    to_small = []
    print(as_to_grab)
    for index, row in tqdm(as_to_grab.iterrows(), total=len(as_to_grab)):
        json_path = row['events_path']
        match_id = row['match_id']
        sizeof = row['events_sizeof']
        sparse_path = row['as_path']
        make_folder(os.path.join(DROPBOX_PATH, f'Statsbomb/atomic_sparse/'),row['competition_id'], row['season_id'])
        if os.path.exists(json_path):
            if sizeof > 100:
                with open(json_path) as f:
                    game_json = json.load(f)
                if type(game_json)==dict:
                    game_df = pd.json_normalize(list(game_json.values()))
                else:
                    game_df = pd.json_normalize(game_json)

                assert(len(list(game_df))<250) # used to occasionally have problems with json_normalize
                game_df,success = json_to_csv(game_df)
                if 'obv_total_net' not in list(game_df):
                    # for now, delete and retry
                    # if keeps happening, will handle differently
                    print(json_path, match_id, "did not have OBV, deleted")
                    os.remove(json_path)
                    success = False
                    no_obv.append((row['competition_id'], row['season_id'], match_id))
                    continue
                if success:
                    game_df.to_csv(sparse_path, index=False)
                else:
                    print(f"error on match id {match_id}")

            else:
                ## too small
                to_small.append((row['competition_id'], row['season_id'], match_id))
                os.remove(json_path)
        else:
            no_json.append((row['competition_id'], row['season_id'], match_id))
            

    ## error handling
    if len(no_obv) > 0:
        print(f"{len(no_obv)} matches had no obv, and so they were skipped")
        no_obv = pd.DataFrame(no_obv, columns=['competition_id','season_id','match_id'])
        print("The errors came from these competitions:  \n")
        print(no_obv.groupby(['competition_id','season_id'])['match_id'].count())
    if len(no_json) > 0:
        print(f"{len(no_json)} matches did not have json, and so they were skipped")
        no_json = pd.DataFrame(no_json, columns=['competition_id','season_id','match_id'])
        print("The json errors came from these competitions:  \n")
        print(no_json.groupby(['competition_id','season_id'])['match_id'].count())
    if len(to_small) > 0:
        print(f"\n{len(to_small)} matches were basically empty, and so they were DELETED\n")
        to_small = pd.DataFrame(to_small, columns=['competition_id','season_id','match_id'])
        print("The json errors came from these competitions:  \n")
        print(to_small.groupby(['competition_id','season_id'])['match_id'].count())
    
    return 


def atomic_sparse_to_STF(cklst):
    
    print("Converting atomic_sparse to STF...")

    cklst['STF'] = False
    cklst['stf_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/STF/{x.competition_id}/{x.season_id}/{x.match_id}-{x.team_id}.csv'), axis=1)
    cklst['as_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/atomic_sparse/{x.competition_id}/{x.season_id}/{x.match_id}.csv'), axis=1)
    cklst['as_sizeof'] = cklst['as_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    # see if its a new game that doesn't exist yet
    cklst['STF'] = cklst.apply(lambda x: os.path.exists(x.stf_path), axis=1)
    # see if its a game that has been updated recently
    cklst['stf_last_modified_UTC'] = cklst['stf_path'].apply(lambda x: EST_to_UTC(pd.to_datetime(time.ctime(os.path.getmtime(x)))) if os.path.exists(x) else np.nan)
    cklst['STF'] = np.where(cklst['last_updated_UTC']>cklst['stf_last_modified_UTC'], False, cklst['STF'].copy())
    # # see if its an empty stf file
    cklst['stf_sizeof'] = cklst['stf_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    cklst['STF'] = np.where(cklst['stf_sizeof'].copy()<10, False, cklst['STF'].copy())
    # some matches just aren't available
    cklst['STF'] = np.where(cklst['match_status']!='available', True, cklst['STF'])
    stf_to_grab = cklst.copy().loc[(cklst['is_upcoming']==0)&(cklst['STF']==False)].reset_index(drop=True)
    stf_to_grab = stf_to_grab.drop_duplicates(subset=['match_id']).reset_index(drop=True) # will split into both teams later
    print(f"{len(stf_to_grab)} needs converted sparse to STF")
    for index, row in tqdm(stf_to_grab.iterrows(), total=len(stf_to_grab)):
        input_path = row['as_path']
        match_id = row['match_id']
        comp_id = row['competition_id']
        season_id = row['season_id']
        sizeof = row['as_sizeof']
        make_folder(os.path.join(DROPBOX_PATH, f'Statsbomb/STF/'),row['competition_id'], row['season_id'])
        if os.path.exists(input_path):
            if sizeof > 100:
                sparse = pd.read_csv(input_path)
                if 'obv_total_net' not in list(sparse):
                    # for now, delete and retry
                    # if keeps happening, will handle differently
                    print(comp_id, season_id, match_id, "did not have OBV, deleted")
                    os.remove(input_path)
                    continue
                df1, team_1, df2, team_2 = csv_to_stf(sparse, match_id)
                output_path_1 = os.path.join(DROPBOX_PATH, f'Statsbomb/STF/{comp_id}/{season_id}/{match_id}-{team_1}.csv')
                output_path_2 = os.path.join(DROPBOX_PATH, f'Statsbomb/STF/{comp_id}/{season_id}/{match_id}-{team_2}.csv')
                df1.to_csv(output_path_1, index=False)
                df2.to_csv(output_path_2, index=False)
                cklst.loc[cklst['match_id']==match_id, 'STF'] = True
    print("Done!")
    
    return 

def game_vec_process(stf_checklist):
    
    print("Updating game vecs...")

    gvecs = pd.read_csv(os.path.join(DROPBOX_PATH, 'Statsbomb/game_vecs/game_vecs.csv'))
    gvecs = gvecs.fillna(0)

    # determine existing and needed
    existing = stf_checklist.copy().loc[stf_checklist['STF']==True].reset_index(drop=True)
    existing['exists'] = existing['stf_path'].apply(lambda x: os.path.exists(x))
    existing = existing.loc[existing['exists']==True].reset_index(drop=True)
    have = set(gvecs.match_id.unique())
    can_have = set(stf_checklist.match_id.unique())

    need_added = can_have.symmetric_difference(have)
    to_collect = existing.copy().loc[existing['match_id'].isin(need_added)].reset_index(drop=True)
    print(f"We can add {len(to_collect)} games to game vecs...")

    # calculate new game vecs
    to_append = []
    for game_path in tqdm(list(to_collect.stf_path.values)):
        try:
            game = pd.read_csv(game_path)
        except Exception as e:
            print(e, f"game path {game_path}")
        if 'position_id' not in list(game):
            print("Shit")
            os.remove(game_path)
            continue
        vec, cols = STF2gvec(game)
        match_id = game['match_id'].values[0]
        team_id = game['team_id'].values[0]

        to_append.append([match_id,team_id]+vec)
    if len(to_append) > 0:
        new_cols = ['match_id','team_id']+cols
        to_append = pd.DataFrame(to_append, columns=new_cols)
        assert(list(to_append)==list(gvecs))
        gvecs = pd.concat([gvecs.copy(), to_append.copy()], axis=0).drop_duplicates(subset=['match_id','team_id']).reset_index(drop=True)
        gvecs.to_csv(os.path.join(DROPBOX_PATH, 'Statsbomb/game_vecs/game_vecs.csv'), index=False)

    
    
    return 

def extract_ref_data(game_path):
    
    if os.path.exists(game_path):
        game = pd.read_csv(game_path)
    else:
        return [np.nan, np.nan, np.nan, np.nan]
    foul_count = len(game.loc[game['type_id']==22])
    if 'foul_committed_card_id' not in list(game):
        red_count,yellow_count=0,0
    else:
        red_count = len(game.loc[game['foul_committed_card_id']==5])
        yellow_count = len(game.loc[game['foul_committed_card_id']==7])
    # free_kicks = len(game.loc[game['secondary_type_id']==62]) ## essentially the same as fouls
    penalties = len(game.loc[game['secondary_type_id']==88])
    
    return [foul_count, yellow_count, red_count, penalties]

def update_ref_df():
    
    sched = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'))
    sched['datetime_UTC'] = pd.to_datetime(sched['datetime_UTC'])
    ref_df = pd.read_csv(os.path.join(DROPBOX_PATH, 'Statsbomb/ref_data/ref_data.csv'))
    have = ref_df.copy().dropna(subset=['foul_count','yellow_count','red_count','penalties']).reset_index(drop=True)
    have['sparse_path'] = have.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/atomic_sparse/{x["competition_id"]}/{x["season_id"]}/{x.match_id}.csv'), axis=1)
    have['path_exists'] = have['sparse_path'].apply(lambda x: os.path.exists(x))
    have = have.loc[have['path_exists']==True].reset_index(drop=True)
    
    dont_have = list(set(sched.match_id.values).difference(have.match_id.values))
#     dont_have = list(sched.match_id.values)
    dont_have = sched.copy().loc[sched['match_id'].isin(dont_have)][['datetime_UTC','match_id','competition_id','season_id','referee_id']]
    dont_have['referee_id'] = dont_have['referee_id'].copy().astype('Int64')
    dont_have['sparse_path'] = dont_have.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/atomic_sparse/{x["competition_id"]}/{x["season_id"]}/{x.match_id}.csv'), axis=1)
    dont_have['path_exists'] = dont_have['sparse_path'].apply(lambda x: os.path.exists(x))
    print(len(dont_have))
    to_append = []
    print("Appending ref stats for new games...")
    for index, game_row in tqdm(dont_have.iterrows(), total=len(dont_have)):
        path = game_row['sparse_path']
        match_id = game_row['match_id']
        ref_data = extract_ref_data(path)
        ref_data.insert(0, match_id)
        to_append.append(ref_data)

    to_append = pd.DataFrame(to_append, columns=['match_id','foul_count','yellow_count','red_count','penalties'])
    before=len(dont_have)
    to_append = pd.merge(dont_have, to_append, how='left', on=['match_id'])
    after = len(to_append)
    to_append = to_append.drop(columns=['path_exists'])
    ref_df = ref_df.drop_duplicates(subset=['match_id'], keep='last')
    assert(before==after)
    if len(to_append) > 0:
        ref_df = pd.concat([ref_df, to_append], axis=0).reset_index(drop=True)
    else:
        return ref_df
    
    ref_df.to_csv(os.path.join(DROPBOX_PATH, 'Statsbomb/ref_data/ref_data.csv'), index=False)
    
    return 




def data_process():
    
    num_workers = 4
    schedule = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'))
    cklst = initialize_checklist(schedule)

    # schedule portion (no stf)
    # divide into fourths to speed up
    # i first randomize, or else the fourth worker is doing all the work
    cklst = cklst.sample(frac=1).reset_index(drop=True)
    cutoff = int(len(cklst)*(1/num_workers))
    cutoffs = []
    for i in range(num_workers):
        cutoffs.append((cutoff*i, cutoff*(i+1)))

    data_1 = cklst.copy()[cutoffs[0][0]:cutoffs[0][1]]    
    data_2 = cklst.copy()[cutoffs[1][0]:cutoffs[1][1]]   
    data_3 = cklst.copy()[cutoffs[2][0]:cutoffs[2][1]]  
    data_4 = cklst.copy()[cutoffs[3][0]:]

    jobs = []
    p1 = Process(target=atomic_sparse_process, args=(data_1.copy(),))
    jobs.append(p1)
    p1.start()
    p2 = Process(target=atomic_sparse_process, args=(data_2.copy(),))
    jobs.append(p2)
    p2.start()
    p3 = Process(target=atomic_sparse_process, args=(data_3.copy(),))
    jobs.append(p3)
    p3.start()
    p4 = Process(target=atomic_sparse_process, args=(data_4.copy(),))
    jobs.append(p4)
    p4.start()

    # checks to see if they are finished
    for job in jobs:
        job.join()
        time.sleep(1)

    print("\nAll schedule jobs are done...\n")

    ## stf portion
    ## can also multiprocess atomic_sparse_to_stf, but cannot with game_vecs (reading/writing on same file)
    stf_schedule = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/stf_schedule.csv'))
    stf_checklist = initialize_stf_checklist(stf_schedule)

    gvecs = pd.read_csv(os.path.join(DROPBOX_PATH, 'Statsbomb/game_vecs/game_vecs.csv'),usecols=['match_id','team_id'])
    have = set(gvecs.match_id.unique())
    can_have = set(stf_checklist.match_id.unique())

    need_added = can_have.symmetric_difference(have)

    print(f"{len(stf_checklist.loc[((stf_checklist['match_id'].isin(need_added))&(stf_checklist['match_status']=='available'))])} game vecs can be added")


    # convert csv to match per game per single team
    # schedule portion (no stf)
    # divide into fourths to speed up
    # i first randomize, or else the fourth worker is doing all the work
    cklst = stf_checklist.sample(frac=1).reset_index(drop=True)
    cutoff = int(len(cklst)*(1/num_workers))
    cutoffs = []
    for i in range(num_workers):
        cutoffs.append((cutoff*i, cutoff*(i+1)))

    data_1 = cklst.copy()[cutoffs[0][0]:cutoffs[0][1]]    
    data_2 = cklst.copy()[cutoffs[1][0]:cutoffs[1][1]]   
    data_3 = cklst.copy()[cutoffs[2][0]:cutoffs[2][1]]  
    data_4 = cklst.copy()[cutoffs[3][0]:]

    jobs = []
    p1 = Process(target=atomic_sparse_to_STF, args=(data_1.copy(),))
    jobs.append(p1)
    p1.start()
    p2 = Process(target=atomic_sparse_to_STF, args=(data_2.copy(),))
    jobs.append(p2)
    p2.start()
    p3 = Process(target=atomic_sparse_to_STF, args=(data_3.copy(),))
    jobs.append(p3)
    p3.start()
    p4 = Process(target=atomic_sparse_to_STF, args=(data_4.copy(),))
    jobs.append(p4)
    p4.start()

    # checks to see if they are finished
    for job in jobs:
        job.join()
        time.sleep(1)

    # redo inventory
    cklst['STF'] = False
    cklst['stf_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/STF/{x.competition_id}/{x.season_id}/{x.match_id}-{x.team_id}.csv'), axis=1)
    cklst['as_path'] = cklst.apply(lambda x: os.path.join(DROPBOX_PATH, f'Statsbomb/atomic_sparse/{x.competition_id}/{x.season_id}/{x.match_id}.csv'), axis=1)
    cklst['as_sizeof'] = cklst['as_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    # see if its a new game that doesn't exist yet
    cklst['STF'] = cklst.apply(lambda x: os.path.exists(x.stf_path), axis=1)
    # see if its a game that has been updated recently
    cklst['stf_last_modified_UTC'] = cklst['stf_path'].apply(lambda x: EST_to_UTC(pd.to_datetime(time.ctime(os.path.getmtime(x)))) if os.path.exists(x) else np.nan)
    cklst['STF'] = np.where(cklst['last_updated_UTC']>cklst['stf_last_modified_UTC'], False, cklst['STF'].copy())
    # # see if its an empty stf file
    cklst['stf_sizeof'] = cklst['stf_path'].apply(lambda x: os.path.getsize(x) if os.path.exists(x) else 200)
    cklst['STF'] = np.where(cklst['stf_sizeof'].copy()<10, False, cklst['STF'].copy())
    # some matches just aren't available
    cklst['STF'] = np.where(cklst['match_status']!='available', True, cklst['STF'])

    print("\nAll STF jobs are done...\n")
    game_vec_process(cklst)

    cklst.to_csv(os.path.join(DROPBOX_PATH, 'checklists/stf_checklist.csv'), index=False)
    update_ref_df()
    
    return



if __name__ == '__main__':
    data_process()










