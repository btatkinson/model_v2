

import gc
import os

import torch
import pickle
import catboost

import numpy as np
import pandas as pd

from copy import copy
from dotenv import load_dotenv
from gensim.models import FastText

load_dotenv()
SB_USERNAME = os.environ.get('SB_USERNAME')
SB_PASSWORD = os.environ.get('SB_PASSWORD')
DROPBOX_PATH = os.environ.get('DROPBOX_PATH')
path_to_data = DROPBOX_PATH
path_to_sb = os.path.join(path_to_data, 'Statsbomb')
OUTPUT_DIR = os.path.join(path_to_sb, 'sparse_STF')
path_to_as = os.path.join(path_to_sb, 'atomic_sparse/')
path_to_ff = os.path.join(path_to_sb, 'shot_freeze_frames')

TARGET_ACTIONS = 12

column_list_path = os.path.join(DROPBOX_PATH, 'Statsbomb/atomic_sparse/column_list.txt')
with open(column_list_path, "rb") as fp:
    column_list = sorted(pickle.load(fp))


# Drop shots, they are generally leaky # not anymore
# shot_cols = [sc for sc in list(column_list) if (('shot_' in sc)&('statsbomb_xg' not in sc)&('goalkeeper' not in sc))]

dropped_columns = ['tactics_formation', 'tactics_lineup', 'related_events', 'substitution_replacement_id', ## <- probably shouldn't drop this going forward
                   'substitution_replacement_name','pass_shot_assist','pass_assisted_shot_id','pass_goal_assist',
                  'index','outcome_name','pass_angle','pass_length',
                  'shot_freeze_frame','shot_key_pass_id','shot_deflected', 'shot_kick_off',#<- ?,
                 'shot_redirect', 'shot_saved_off_target', 'shot_saved_to_post']

column_list = [x for x in column_list if x not in dropped_columns]

def drop_meta(df):

    """ These are variables that are meta information and not helpful to xxG """

    df = df.loc[~df['type_name'].isin(['Starting XI','Tactical Shift','Substitution'])].reset_index(drop=True)
    # related events? might be useful in avoiding redundant actions
    df = df.drop(columns=['tactics_formation', 'tactics_lineup', 'related_events',
                         'substitution_replacement_id', 'substitution_replacement_name',
                         'shot_freeze_frame'], errors='ignore')

    return df

def drop_leaky(df):

    """ Some variables tip off there is about to be a goal or a shot """

    leaky = ['pass_shot_assist','pass_assisted_shot_id','pass_goal_assist','shot_end_location','shot_key_pass_id']
    leaky.extend(['shot_deflected', 'shot_kick_off',#<- ?,
                 'shot_redirect', 'shot_saved_off_target', 'shot_saved_to_post'])
    leaky = [l for l in leaky if l in list(df)]
    df = df.drop(columns=leaky)

    # the entire type id 37 is leaky, error, roughly 0.02% of actions
    # replace with 'dispossessed'
    df.loc[df['type_id']==37, 'type_id'] = 3
    return df

def separate_outcomes(df):

    """
    Passes already have an outcome defined through ball receipt. This function
    creates new outcome ids for non ball receipts (500 + outcome id)
    """
    # drop shot outcomes for xxG because it is redundant (and leaky) with goalkeeper outcomes
    outcome_df = df[(df['outcome_name'].notnull())&(~df['type_name'].isin(['Pass','Ball Receipt*','Shot']))].copy(deep=True).reset_index(drop=True)

    # create new ids by adding 500 to outcomes
    outcome_df.loc[:, 'type_id'] = outcome_df['outcome_id'].copy() + 500
    outcome_df.loc[:, 'type_name'] = outcome_df['type_name'].copy() + ' ' + outcome_df['outcome_name']

    # technique names should probably stick with the outcomes
    df.loc[(df['outcome_name'].notnull())&(df['type_name'].isin(['Goal Keeper'])), 'technique_id'] = 0
    df.loc[(df['outcome_name'].notnull())&(df['type_name'].isin(['Goal Keeper'])), 'technique_name'] = np.nan

    # as does body part
    df.loc[(df['outcome_name'].notnull())&(df['type_name'].isin(['Goal Keeper'])), 'bodypart_id'] = 0
    df.loc[(df['outcome_name'].notnull())&(df['type_name'].isin(['Goal Keeper'])), 'bodypart_name'] = np.nan

    # secondary type id and name seem to fit better with outcomes
    df.loc[(df['outcome_name'].notnull()), 'secondary_type_id'] = 0
    df.loc[(df['outcome_name'].notnull()), 'secondary_type_name'] = np.nan

    # goalkeeper position id is a original row value
    outcome_df.loc[:, 'goalkeeper_position_name'] = 0
    outcome_df.loc[:, 'goalkeeper_position_id'] = np.nan

    # for remerge change index
    outcome_df.loc[:, 'index'] = outcome_df['index'].copy() + 0.5

    # outcomes don't need event ids
    outcome_df.loc[:, 'id'] = np.nan

    # remerge & re number index
    df = pd.concat([df, outcome_df],axis=0).sort_values(by='index').drop('index',axis=1)

    # drop outcome columns ### NVM, need it to calculate things like shot on target
#     df = df.drop(columns=['outcome_name','outcome_id'])

    df = df.loc[df['type_id']!=25].reset_index(drop=True)

    return df.reset_index(drop=True)

def split_STF(df):

    """ Each team gets their own dataframe """

    df1 = df.copy(deep=True)
    df2 = df.copy(deep=True)

    team_1, team_2 = df['team_id'].unique()

    team_1_name = df.loc[df['team_id']==team_1][['team_name']].head(1).values[0][0]
    team_2_name = df.loc[df['team_id']==team_2][['team_name']].head(1).values[0][0]

    df1['action_team'] = np.where(df1['team_id']==team_1, 1, 0)
    df2['action_team'] = np.where(df2['team_id']==team_2, 1, 0)

    df1.loc[:, 'team_id'] = team_1
    df1.loc[:, 'team_name'] = team_1_name
    df2.loc[:, 'team_id'] = team_2
    df2.loc[:, 'team_name'] = team_2_name

    MAX_COORDS = (120,80)

    df1.loc[df1['action_team']==0, 'x'] = MAX_COORDS[0] - df1['x'].copy()
    df1.loc[df1['action_team']==0, 'y'] = MAX_COORDS[1] - df1['y'].copy()

    df1.loc[df1['action_team']==0, 'end_x'] = MAX_COORDS[0] - df1['end_x'].copy()
    df1.loc[df1['action_team']==0, 'end_y'] = MAX_COORDS[1] - df1['end_y'].copy()

    df2.loc[df2['action_team']==0, 'x'] = MAX_COORDS[0] - df2['x'].copy()
    df2.loc[df2['action_team']==0, 'y'] = MAX_COORDS[1] - df2['y'].copy()

    df2.loc[df2['action_team']==0, 'end_x'] = MAX_COORDS[0] - df2['end_x'].copy()
    df2.loc[df2['action_team']==0, 'end_y'] = MAX_COORDS[1] - df2['end_y'].copy()

    ## swap OBV
    temp_1 = df1.copy()[['action_team','obv_against_after','obv_against_before','obv_against_net','obv_for_before','obv_for_after','obv_for_net','obv_total_net']]
    temp_2 = df2.copy()[['action_team','obv_against_after','obv_against_before','obv_against_net','obv_for_before','obv_for_after','obv_for_net','obv_total_net']]
    
    temp_1['obv_against_after'] = np.where(temp_1['action_team']==1, df1['obv_against_after'].copy(), df1['obv_for_after'].copy())
    temp_1['obv_against_before'] = np.where(temp_1['action_team']==1, df1['obv_against_before'].copy(), df1['obv_for_before'].copy())
    temp_1['obv_against_net'] = np.where(temp_1['action_team']==1, df1['obv_against_net'].copy(), df1['obv_for_net'].copy())
    
    temp_1['obv_for_after'] = np.where(temp_1['action_team']==1, df1['obv_for_after'].copy(), df1['obv_against_after'].copy())
    temp_1['obv_for_before'] = np.where(temp_1['action_team']==1, df1['obv_for_before'].copy(), df1['obv_against_after'].copy())
    temp_1['obv_for_net'] = np.where(temp_1['action_team']==1, df1['obv_for_net'].copy(), df1['obv_against_net'].copy())
    
    temp_1['obv_total_net'] = np.where(temp_1['action_team']==1, df1['obv_total_net'].copy(), -1*df1['obv_total_net'].copy())
    
    temp_2['obv_against_after'] = np.where(temp_2['action_team']==1, df2['obv_against_after'].copy(), df2['obv_for_after'].copy())
    temp_2['obv_against_before'] = np.where(temp_2['action_team']==1, df2['obv_against_before'].copy(), df2['obv_for_before'].copy())
    temp_2['obv_against_net'] = np.where(temp_2['action_team']==1, df2['obv_against_net'].copy(), df2['obv_for_net'].copy())
    
    temp_2['obv_for_after'] = np.where(temp_2['action_team']==1, df2['obv_for_after'].copy(), df2['obv_against_after'].copy())
    temp_2['obv_for_before'] = np.where(temp_2['action_team']==1, df2['obv_for_before'].copy(), df2['obv_against_after'].copy())
    temp_2['obv_for_net'] = np.where(temp_2['action_team']==1, df2['obv_for_net'].copy(), df2['obv_against_net'].copy())
    
    temp_2['obv_total_net'] = np.where(temp_2['action_team']==1, df2['obv_total_net'].copy(), -1*df2['obv_total_net'].copy())
    
    df1[['obv_against_after','obv_against_before','obv_against_net','obv_for_before','obv_for_after','obv_for_net','obv_total_net']] = temp_1[['obv_against_after','obv_against_before','obv_against_net','obv_for_before','obv_for_after','obv_for_net','obv_total_net']].copy()
    df2[['obv_against_after','obv_against_before','obv_against_net','obv_for_before','obv_for_after','obv_for_net','obv_total_net']] = temp_2[['obv_against_after','obv_against_before','obv_against_net','obv_for_before','obv_for_after','obv_for_net','obv_total_net']].copy()

    return df1, team_1, df2, team_2


def vector_angles(us, vs):

    # dot products
    dots = np.einsum('ij, ij->i', us, vs)

    # magnitudes
    mags = np.sqrt(us[:,0]**2+us[:,1]**2)*np.sqrt(vs[:,0]**2+vs[:,1]**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        angles = np.arccos(dots/mags)

    angles[angles == -np.inf] = 0
    angles[angles == np.inf] = 0

    return np.nan_to_num(angles)

def fe(df):

    """ Add features like dx, dy, angle to goal"""

    goal_coords = (120, 40)
    opp_goal_coords = (0, 40)

    top_post = (120, 36)
    bottom_post = (120, 44)

    opp_top_post = (0, 36)
    opp_bottom_post = (0, 44)

    df['end_x'] = df['end_x'].copy().fillna(df['x'].copy())
    df['end_y'] = df['end_y'].copy().fillna(df['y'].copy())

    df['dx'] = df['end_x'].copy() - df['x'].copy()
    df['dy'] = df['end_y'].copy() - df['y'].copy()

    df.loc[:, 'temp_length'] = np.sqrt(df['dx'].copy()**2 + df['dy'].copy()**2)
    df.loc[:, 'distance'] = df['pass_length'].copy()
    df['distance'] = df['distance'].copy().fillna(df['temp_length'])
    df = df.drop(columns=['temp_length','pass_length'])

    # angle between vector of [dx, dy] and unit vector (always positive)
    us = np.array([df['dx'].values, df['dy'].values]).T
    vs = np.zeros((len(us),2))
    vs[:,0] = 1

    df.loc[:,'angle'] = vector_angles(us,vs)
    df = df.drop('pass_angle',axis=1)
    df['angle'] = df['angle'].copy().fillna(0)

    # get start and end distance to nearest goal
    df.loc[:, 'start_dtg'] = np.sqrt((goal_coords[0]-df['x'].copy())**2 + (goal_coords[1]-df['y'].copy())**2)
    df.loc[:,'end_dtg'] = np.sqrt((goal_coords[0]-df['end_x'].copy())**2 + (goal_coords[1]-df['end_y'].copy())**2)

    df.loc[:,'start_dtog'] = np.sqrt((opp_goal_coords[0]-df['x'].copy())**2 + (opp_goal_coords[1]-df['y'].copy())**2)
    df.loc[:,'end_dtog'] = np.sqrt((opp_goal_coords[0]-df['end_x'].copy())**2 + (opp_goal_coords[1]-df['end_y'].copy())**2)

    # only use nearest
    df.loc[:, 'start_dtg'] = np.min(df[['start_dtg','start_dtog']], axis=1)
    df.loc[:, 'end_dtg'] = np.min(df[['end_dtg','end_dtog']], axis=1)

    df = df.drop(columns=['start_dtog','end_dtog'])

    # delta angle to goal
    # start angle to goal'
    df['satg'] = vector_angles(np.array([goal_coords[0]-df['x'].values, np.repeat(0,len(df))]).T,
                                np.array([goal_coords[0]-df['x'].values, df['y'].values - np.repeat(goal_coords[1],len(df))]).T)
    # end angle to goal
    df['eatg'] = vector_angles(np.array([goal_coords[0]-df['end_x'].values, np.repeat(0,len(df))]).T,
                                np.array([goal_coords[0]-df['end_x'].values, df['end_y'].values - np.repeat(goal_coords[1],len(df))]).T)

    # same for own goal
    # start angle to goal
    df['satog'] = vector_angles(np.array([opp_goal_coords[0]-df['x'].values, np.repeat(0,len(df))]).T,
                                np.array([opp_goal_coords[0]-df['x'].values, df['y'].values - np.repeat(opp_goal_coords[1],len(df))]).T)
    # end angle to goal
    df['eatog'] = vector_angles(np.array([opp_goal_coords[0]-df['end_x'].values, np.repeat(0,len(df))]).T,
                                np.array([opp_goal_coords[0]-df['end_x'].values, df['end_y'].values - np.repeat(opp_goal_coords[1],len(df))]).T)

    # change up to negative, down to positive
    df.loc[:,'satg'] = np.where(df['y']>=goal_coords[1], -1*df['satg'].copy(), df['satg'].copy())
    df.loc[:,'eatg'] = np.where(df['y']>=goal_coords[1], -1*df['eatg'].copy(), df['eatg'].copy())

    df.loc[:,'satog'] = np.where(df['y']>=goal_coords[1], -1*df['satog'].copy(), df['satog'].copy())
    df.loc[:,'eatog'] = np.where(df['y']>=goal_coords[1], -1*df['eatog'].copy(), df['eatog'].copy())

    df.loc[:,'da'] = np.abs(df['eatg'].copy() - df['satg'].copy())
    df.loc[:,'daog'] = np.abs(df['eatog'].copy() - df['satog'].copy())

    # start angle to top post
    df['satp'] = vector_angles(np.array([top_post[0]-df['x'].values, np.repeat(0,len(df))]).T,
                                np.array([top_post[0]-df['x'].values, df['y'].values - np.repeat(top_post[1],len(df))]).T)
    df['sabp'] = vector_angles(np.array([bottom_post[0]-df['x'].values, np.repeat(0,len(df))]).T,
                                np.array([bottom_post[0]-df['x'].values, df['y'].values - np.repeat(bottom_post[1],len(df))]).T)

    # end angle to top post
    df['eatp'] = vector_angles(np.array([top_post[0]-df['end_x'].values, np.repeat(0,len(df))]).T,
                                np.array([top_post[0]-df['end_x'].values, df['end_y'].values - np.repeat(top_post[1],len(df))]).T)
    df['eabp'] = vector_angles(np.array([bottom_post[0]-df['end_x'].values, np.repeat(0,len(df))]).T,
                                np.array([bottom_post[0]-df['end_x'].values, df['end_y'].values - np.repeat(bottom_post[1],len(df))]).T)

    # start angle to own top post
    df['satop'] = vector_angles(np.array([df['x'].values - opp_top_post[0], np.repeat(0,len(df))]).T,
                                np.array([df['x'].values - opp_top_post[0], df['y'].values - np.repeat(opp_top_post[1],len(df))]).T)
    df['sabop'] = vector_angles(np.array([df['x'].values - opp_bottom_post[0], np.repeat(0,len(df))]).T,
                                np.array([df['x'].values - opp_bottom_post[0], df['y'].values - np.repeat(opp_bottom_post[1],len(df))]).T)

    # end angle to own top post
    df['eatop'] = vector_angles(np.array([df['end_x'].values - opp_top_post[0], np.repeat(0,len(df))]).T,
                                np.array([df['end_x'].values - opp_top_post[0], df['end_y'].values - np.repeat(opp_top_post[1],len(df))]).T)
    df['eabop'] = vector_angles(np.array([df['end_x'].values - opp_bottom_post[0], np.repeat(0,len(df))]).T,
                                np.array([df['end_x'].values - opp_bottom_post[0], df['end_y'].values - np.repeat(opp_bottom_post[1],len(df))]).T)


    # make passes up negative
    df.loc[:,'satp'] = np.where(df['y']>=top_post[1], -1*df['satp'].copy(), df['satp'].copy())
    df.loc[:,'eatp'] = np.where(df['y']>=top_post[1], -1*df['eatp'].copy(), df['eatp'].copy())

    df.loc[:,'satop'] = np.where(df['y']>=opp_top_post[1], -1*df['satop'].copy(), df['satop'].copy())
    df.loc[:,'eatop'] = np.where(df['y']>=opp_top_post[1], -1*df['eatop'].copy(), df['eatop'].copy())

    df.loc[:,'sabp'] = np.where(df['y']>=bottom_post[1], -1*df['sabp'].copy(), df['sabp'].copy())
    df.loc[:,'eabp'] = np.where(df['y']>=bottom_post[1], -1*df['eabp'].copy(), df['eabp'].copy())

    df.loc[:,'sabop'] = np.where(df['y']>=opp_bottom_post[1], -1*df['sabop'].copy(), df['sabop'].copy())
    df.loc[:,'eabop'] = np.where(df['y']>=opp_bottom_post[1], -1*df['eabop'].copy(), df['eabop'].copy())

    # start and end goalmouth angle
    df['sgma'] = np.abs(df['satp'].copy()-df['sabp'].copy())
    df['egma'] = np.abs(df['eatp'].copy()-df['eabp'].copy())

    # start and end own goalmouth angle
    df['sogma'] = np.abs(df['satop'].copy()-df['sabop'].copy())
    df['eogma'] = np.abs(df['eatop'].copy()-df['eabop'].copy())

    # start doing only closest goal
    df.loc[:,'da'] = np.where(df['x']>=goal_coords[0]/2, df['da'].copy(), df['daog'].copy())
    df.loc[:,'start_gma'] = np.where(df['x']>=goal_coords[0]/2, df['sgma'].copy(), df['sogma'].copy())
    df.loc[:,'end_gma'] = np.where(df['x']>=goal_coords[0]/2, df['egma'].copy(), df['eogma'].copy())

    df = df.drop(columns=['satp','sabp','satop','sabop','eatp','eabp','eatop','eabop','daog','sogma','eogma',
                         'satg', 'eatg', 'satog', 'eatog','sgma','egma'])

    df['da'] = df['da'].copy().fillna(0)

    df['duration'] = df['duration'].copy().fillna(0.05)
    df['duration'] = np.where(df['duration'].copy()==0, 0.05, df['duration'].copy())

    #speeds
    df['x_speed'] = df['dx'].copy()/df['duration'].copy()
    df['y_speed'] = df['dy'].copy()/df['duration'].copy()
    df['a_speed'] = df['da'].copy()/df['duration'].copy()
    df['speed'] = df['distance'].copy()/df['duration'].copy()

    df['x_speed'] = df['x_speed'].fillna(0)
    df['y_speed'] = df['y_speed'].fillna(0)
    df['a_speed'] = df['a_speed'].fillna(0)
    df['speed'] = df['speed'].fillna(0)

    if 'ball_z' in list(df):
        df['ball_z'] = df['ball_z'].fillna(-1) # because 0 is meaningful
    else:
        df.loc[:, 'ball_z'] = -1

    return df

def add_man_adv(df):

    df.loc[:, 'team_lose_man'] = 0
    df.loc[:, 'opp_lose_man'] = 0

    # masks for different types of ways to lose people
    injury_loss = ((df.player_off_permanent==True)&(df.action_team==1))
    bad_behave_red = ((df.bad_behaviour_card_id==5)&(df.action_team==1))
    bad_behave_sy = ((df.bad_behaviour_card_id==6)&(df.action_team==1))
    foul_red = ((df.foul_committed_card_id==5)&(df.action_team==1))
    foul_sy = ((df.foul_committed_card_id==6)&(df.action_team==1))

    team_man_off = [injury_loss, bad_behave_red, bad_behave_sy, foul_red, foul_sy]

    injury_loss_opp = ((df.player_off_permanent==True)&(df.action_team==0))
    bad_behave_red_opp = ((df.bad_behaviour_card_id==5)&(df.action_team==0))
    bad_behave_sy_opp = ((df.bad_behaviour_card_id==6)&(df.action_team==0))
    foul_red_opp = ((df.foul_committed_card_id==5)&(df.action_team==0))
    foul_sy_opp = ((df.foul_committed_card_id==6)&(df.action_team==0))


    df.loc[injury_loss|bad_behave_red|bad_behave_sy|foul_red|foul_sy, 'team_lose_man'] = 1

    df.loc[injury_loss_opp|bad_behave_red_opp|bad_behave_sy_opp|foul_red_opp|foul_sy_opp, 'opp_lose_man'] = 1

    df['team_strength'] = 11 - df['team_lose_man'].copy().cumsum()
    df['opp_strength'] = 11 - df['opp_lose_man'].copy().cumsum()

    df = df.drop(columns=['team_lose_man', 'opp_lose_man'])

    return df



def add_target(df):

    df.loc[:, 'VAEP_target'] = np.nan
    df.loc[:, 'team_goal'] = np.nan
    df.loc[:, 'opp_goal'] = np.nan

    # masks for different types of goals
    ts_reg = ((df.secondary_type_id==26)&(df.type_id!=20)&(df.action_team==0))
    ts_pen = ((df.secondary_type_id==28)&(df.type_id!=20)&(df.action_team==0))
    ts_og = ((df.type_id==20)&(df.action_team==0))

    os_reg = ((df.secondary_type_id==26)&(df.type_id!=20)&(df.action_team==1))
    os_pen = ((df.secondary_type_id==28)&(df.type_id!=20)&(df.action_team==1))
    os_og = ((df.type_id==20)&(df.action_team==1))

    df.loc[ts_reg|ts_pen|ts_og, 'VAEP_target'] = 2
    df.loc[os_reg|os_pen|os_og, 'VAEP_target'] = 0

    df.loc[ts_reg|ts_pen|ts_og, 'team_goal'] = 1
    df.loc[os_reg|os_pen|os_og, 'opp_goal'] = 1
    df['team_goal'].fillna(0, inplace=True)
    df['opp_goal'].fillna(0, inplace=True)

    df['VAEP_target'] = df['VAEP_target'].copy().backfill(limit=TARGET_ACTIONS)
    df['VAEP_target'] = df['VAEP_target'].fillna(1).astype(int)

    df['team_score'] = df['team_goal'].cumsum().fillna(0).astype(int)
    df['opp_score'] = df['opp_goal'].cumsum().fillna(0).astype(int)

    return df


def add_missing_cols(df):

    global column_list

    needed_cols = list(set(column_list) - set(list(df)))
    df.loc[:, needed_cols] = np.nan

    added_cols = ['action_team','distance','angle','x_speed','y_speed','a_speed','speed','da','start_gma','end_gma','VAEP_target',
                'outcome_name', 'dx', 'end_dtg', 'dy', 'start_dtg', 'bodypart_id', 'bodypart_name',
                'team_score','opp_score','team_goal','opp_goal']

    new_cols = list(set(list(df))-set(column_list)-set(added_cols))

    if len(new_cols) > 0:
        print(f"NEW COLS: {new_cols}")

    # assert(len(new_cols)==0) # make sure SB isn't slipping in new stuff

    df = df[column_list+added_cols]

    return df

def action_keys(df):

    # bools for each action
    bool_dict = {
        'Ball Recovery':['ball_recovery_offensive','ball_recovery_recovery_failure'],
        'Block':['block_deflection', 'block_offensive', 'block_save_block'],
        'Clearance':['clearance_aerial_won','clearance_other'],
        'Dribble':['dribble_nutmeg','dribble_overrun','dribble_no_touch'],
        'Foul':['foul_committed_advantage', 'foul_committed_offensive', 'foul_committed_penalty',
                'foul_won_advantage','foul_won_defensive', 'foul_won_penalty'],
        'Goalkeeper':['goalkeeper_position_id','goalkeeper_lost_in_play', 'goalkeeper_lost_out', 'goalkeeper_penalty_saved_to_post','goalkeeper_punched_out',
                    'goalkeeper_saved_to_post', 'goalkeeper_shot_saved_off_target', 'goalkeeper_shot_saved_to_post',
                    'goalkeeper_success_in_play', 'goalkeeper_success_out'],
        'Miscontrol':['miscontrol_aerial_won'],
        'Pass':['pass_height_id','pass_aerial_won','pass_backheel','pass_cross', 'pass_cut_back', 'pass_deflected',
                'pass_miscommunication', 'pass_no_touch','pass_switch'],
        'Shot':['shot_aerial_won', 'shot_one_on_one','shot_first_time', 'shot_follows_dribble','shot_open_goal']

# happened after shot but shouldn't be factored into shot xxG
#['shot_deflected', 'shot_kick_off'<- ?, 'shot_redirect', 'shot_saved_off_target', 'shot_saved_to_post']
# I think the at least the last two are captured with goalkeeper actions
    }



    specific_ids = ['pass_height_id','goalkeeper_position_id']

    encoding_cols = []

    for action_type, action_bools in bool_dict.items():
        encoding_col = None
        encoding_col_name = action_type.lower().replace(' ','_') + '_encoding'
        for ab in action_bools:
            if ab not in specific_ids:
                df[ab] = df[ab].copy().fillna(False)
                df[ab] = df[ab].astype(int)
            else:
                df[ab] = df[ab].fillna("0")

        encoding_vals = df[action_bools].astype(str).values
        joint_encoding = ['-'.join(row) for row in encoding_vals]
        df.loc[:, encoding_col_name] = joint_encoding
        df = df.drop(columns=action_bools)
        encoding_cols.append(encoding_col_name)


    df['specific_encode'] = np.nan
    for ec in encoding_cols:
        # if there is nothing, replace with nan
        df.loc[~df[ec].str.contains("1"), ec] = np.nan
        df['specific_encode'] = df['specific_encode'].copy().fillna(df[ec].copy())
        df = df.drop(ec, axis=1)
    df['specific_encode'] = df['specific_encode'].fillna(0)


    useful_bools = ['counterpress','out','under_pressure','injury_stoppage_in_chain']
    for ub in useful_bools:
        df[ub] = df[ub].copy().fillna(False)
        df[ub] = df[ub].astype(int)

    encoding_vals = df[useful_bools].astype(str)

    # complexity reduction #
    encoding_vals = encoding_vals.drop(columns=['injury_stoppage_in_chain']).values

    joint_encoding = ['-'.join(row) for row in encoding_vals]
    df.loc[:, 'useful_encode'] = joint_encoding
    df = df.drop(columns=useful_bools)


    core_ids = ['type_id','body_part_id','secondary_type_id','technique_id']

    encoding_vals = df[core_ids].astype(str)

    # complexity reduction # # combine right and left of every body part
    # not needed with better action encodings

#     encoding_vals.loc[encoding_vals['body_part_id']=='38', 'body_part_id'] = '40'
#     encoding_vals.loc[encoding_vals['body_part_id']=='41', 'body_part_id'] = '42'

    encoding_vals = encoding_vals.values

    joint_encoding = ['-'.join(row) for row in encoding_vals]
    df.loc[:, 'action_encode'] = joint_encoding
    # keep type id & secondary id because it's useful for calculating boxscore features
#     df = df.drop(columns=core_ids)
    df = df.drop(columns=['body_part_id','technique_id'])

    final_encodings = ['action_encode','useful_encode','specific_encode']
    encoding_vals = df[final_encodings].astype(str).values
    joint_encoding = ['-'.join(row) for row in encoding_vals]
    df.loc[:, 'action_encoding'] = joint_encoding
    df = df.drop(columns=final_encodings)

    # make the encodings team_specific
    df['action_encoding'] = np.where(df['action_team']==1, df['action_encoding'].copy(), '*-'+df['action_encoding'].copy())

    desc_cols = ['type_name','body_part_name','secondary_type_name','technique_name']
    desc_vals = df[desc_cols].astype(str).replace('nan','').values
    joint_desc = [' '.join(row) for row in desc_vals]
    df.loc[:, 'action_desc'] = joint_desc
    df['action_desc'] = df['action_desc'].str.strip()
    df = df.drop(columns=desc_cols)

    return df

def optimal_dtypes(df):

    """Condense storage as much as possible"""

    droppable = ['bad_behaviour_card_name',# 'bad_behaviour_card_id','foul_committed_card_id', # need these for stats
                 'foul_committed_card_name', 'goalkeeper_end_location', # already in 'end_x','end_y'
                 'goalkeeper_position_name','half_end_early_video_end', 'half_end_match_suspended',
                 'half_start_late_video_start','off_camera','pass_height_name', 'pass_recipient_id', # should be in player id
                 'pass_recipient_name','player_off_permanent',#'position_id','position_name',
                 'possession_team_name'
                ]

    df = df.drop(columns=droppable)

    float_32s = ['x','y','ball_z','end_x','end_y','distance','angle','x_speed','y_speed','a_speed','speed','da','start_gma','end_gma']
    int_16s = ['minute','period','play_pattern_id','possession','second','VAEP_target']

    df[float_32s] = df[float_32s].astype('float32')
    df[int_16s] = df[int_16s].astype('int16')

    # complexity reduction # # only specify certain play patterns
    # no n
#     legit = [5,6]
#     df.loc[~df['play_pattern_id'].isin(legit), 'play_pattern_id'] = 1

    return df

### predict xxG funx ###
xxG_NUM_ACTIONS = 4

xxG_feat_cols = ['x','y','end_x','end_y','x_speed','y_speed','distance','speed','duration',
                 'dx', 'end_dtg', 'dy', 'start_dtg',
                 'angle','da','a_speed','start_gma','end_gma','play_pattern_id','ball_z']

# meta saved separately because does not need to be 3D
xxG_meta_cols = ['match_id','team_id','id','action_encoding','VAEP_target']

# might be useful at some point
xxG_situation_cols = ['minute','second','timestamp','period','team_strength','opponent_strength']

leaky_encodes = [
    '555-0-28-45-0-0-0-0',
    '*-555-0-26-46-0-0-0-0',
    '*-555-0-26-45-0-0-0-0',
    '*-555-0-28-45-0-0-0-0',
    '23-0-0-0-0-0-0-0',
    '555-0-26-45-0-0-0-0',
    '*-23-0-0-0-0-0-0-0',
    '555-0-26-46-0-0-0-0'
]

def load_embed_model(path=os.path.join(DROPBOX_PATH, 'models/action_embeds/skip_gram_5000.model')):
    return FastText.load(path)

def map_vecs(x):
    return embed_model.wv[x] # works on out-of-vocabulary words

def action_encode(df):
    return np.matrix(df['action_encoding'].apply(lambda x: map_vecs(x)).tolist())

def feature_cols(df, feat_cols):
    return np.matrix(df[feat_cols].copy())

def situation_cols(df, sit_cols):
    return np.matrix(df[sit_cols].copy())

def meta_info(df):
    # must separate out strings
    return np.matrix(df[xxG_meta_cols].drop(columns=['id','action_encoding']).reset_index()),df[['id','action_encoding']].values.reshape(-1,2).astype(str) # reset_index so that action index is kept

def indices_to_keep(df):
    # save on memory by dropping indexes that are unlikely to be interesting
    keep_prob = np.clip((np.abs(df['x']-60)+10)/60,0,1)
    keep_prob = np.where(df['action_encoding'].isin(leaky_encodes), 0, keep_prob)
    return np.random.random(len(keep_prob))<keep_prob

def to_timeseries(np2D):

    window = xxG_NUM_ACTIONS

    # this function does all the heavy lifting, but leaves out the first few samples with nans
    ts = torch.from_numpy(np.pad(np2D, ((window-1, 0), (0,0)))).unfold(0, window, 1).numpy().astype(float)

    # makes more sense with columns as second dimension
    return ts.swapaxes(2, 1)

embed_model = load_embed_model()

### don't use this anymore, opp adj are later ###
# def add_opp_rating(df, match_id, team_id):

#     opp_path = os.path.join(OLD_DROPBOX, 'transfermarkt/v3_opp_adj.csv')
#     opp_ratings = pd.read_csv(opp_path)
#     opp_ratings['match_id'] = opp_ratings['match_id'].astype(int)
#     opp_ratings['team_id'] = opp_ratings['team_id'].astype(int)
    
#     opp_rating = opp_ratings.loc[((opp_ratings['match_id']==int(match_id))&(opp_ratings['team_id']==int(team_id)))].copy(deep=True)
    
#     assert(len(opp_rating)<=1)
#     if len(opp_rating) == 1:
#         opp_rtg = opp_rating['opp_rating'].values[0]
#     else:
#         print(f"{match_id} ratings need updated")
#         raise ValueError()
#         # opp_rtg = -0.9 # generic bad team
        
#     df['opp_adj'] = opp_rtg
    
#     return df

def prep_for_predict(df, situation=True, opponent=True, training=False):
    
    actions = action_encode(df)
    meta, action_id = meta_info(df) # action_id must be saved separately because it's a string
    # indicies_to_keep = indices_to_keep(df) # not needed in production

    feat_cols = ['x','y','end_x','end_y','x_speed','y_speed','distance','speed','duration',
                 'dx', 'end_dtg', 'dy', 'start_dtg',
                 'angle','da','a_speed','start_gma','end_gma','play_pattern_id','ball_z']
    
    features = feature_cols(df, feat_cols)
    sit_cols = ['minute','period','team_strength','opp_strength','team_score','opp_score']
    if not situation:
        sit_cols = []

    if situation:
        with open(os.path.join(DROPBOX_PATH, 'utility/sit_xxG_cols'), "rb") as fp:   # Unpickling
            verify_cols = pickle.load(fp)
        assert(verify_cols==feat_cols+sit_cols)
    else:
        with open(os.path.join(DROPBOX_PATH, 'utility/xxG_cols'), "rb") as fp:   # Unpickling
            verify_cols = pickle.load(fp)
        assert(verify_cols==feat_cols)

    combined = np.concatenate([actions, features], axis=1).astype(float)

    lstm_input = to_timeseries(combined)
    gb_input = lstm_input.reshape(lstm_input.shape[0],-1)

    if situation:
        situational = situation_cols(df, sit_cols)
        gb_input = np.concatenate([gb_input, situational], axis=1)

    ## not needed in production
    # if training: # drop indicies of low importance
    #     meta = meta[indicies_to_keep]
    #     action_id = action_id[indicies_to_keep]
    #     lstm_input = lstm_input[indicies_to_keep]
    #     gb_input = gb_input[indicies_to_keep]
    
    return meta, action_id, gb_input, lstm_input


def load_catboost(MODEL_NO, situational=True):
    model = catboost.CatBoostClassifier()
    if situational:
        sit = 'sit'
    else:
        sit = ''
        
    opp = ''

    model_path = os.path.join(DROPBOX_PATH, f'models/xxG/2022-02-01-catboost_model-{sit}{opp}-{MODEL_NO}.dump')
    model.load_model(model_path)
    return model



standard_model_1 = load_catboost(0, False)
standard_model_2 = load_catboost(1, False)
standard_model_3 = load_catboost(2, False)

sit_model_1 = load_catboost(0, True)
sit_model_2 = load_catboost(1, True)
sit_model_3 = load_catboost(2, True)



_all_feat_cols = ['x','y','end_x','end_y','x_speed','y_speed','distance','speed','duration',
                 'dx', 'end_dtg', 'dy', 'start_dtg',
                 'angle','da','a_speed','start_gma','end_gma','play_pattern_id','ball_z']

# meta saved separately because does not need to be 3D
_meta_cols = ['match_id','team_id','id','action_encoding','VAEP_target']

_feat_cols = [f'a_{i}' for i in range(6)]+_all_feat_cols

_cols = [fc + f'_{j}' for j in range(4) for fc in _feat_cols]


def predict_xxG(t1_df, t2_df):
    
    meta1, action_id1, X_gb1, X_lstm1 = prep_for_predict(t1_df, situation=False, training=False)
    meta2, action_id2, X_gb2, X_lstm2 = prep_for_predict(t2_df, situation=False, training=False)
    
    # non situation
    cols = copy(_cols)
    
    ## edit 11/9/2021 dropping columns
    X_gb1 = pd.DataFrame(X_gb1, columns=cols)
    X_gb2 = pd.DataFrame(X_gb2, columns=cols)
    
    X_gb1['avg_speed'] = X_gb1[['speed_0','speed_1','speed_2','speed_3']].mean(axis=1)
    X_gb1['total_dist'] = X_gb1[['distance_0','distance_1','distance_2','distance_3']].sum(axis=1)
    X_gb2['avg_speed'] = X_gb2[['speed_0','speed_1','speed_2','speed_3']].mean(axis=1)
    X_gb2['total_dist'] = X_gb2[['distance_0','distance_1','distance_2','distance_3']].sum(axis=1)
    
    cols.append('avg_speed')
    cols.append('total_dist')
    
    to_remove=['team_score','opp_score','play_pattern_id_0','play_pattern_id_1','play_pattern_id_2','ball_z_0','ball_z_1','ball_z_2','team_strength','opp_strength']
    to_remove.extend(['speed_0','speed_1','speed_2','speed_3','distance_0','distance_1','distance_2','distance_3','period'])

    X_gb1 = X_gb1.drop(columns=to_remove, errors='ignore')
    X_gb2 = X_gb2.drop(columns=to_remove, errors='ignore')

    cols = [c for c in cols if c not in to_remove]
    
    preds_1_1 = standard_model_1.predict_proba(np.asarray(X_gb1))
    preds_1_2 = standard_model_2.predict_proba(np.asarray(X_gb1))
    preds_1_3 = standard_model_3.predict_proba(np.asarray(X_gb1))

    preds_1 = (preds_1_1 + preds_1_2 + preds_1_3)/3
    
    preds_2_1 = standard_model_1.predict_proba(np.asarray(X_gb2))
    preds_2_2 = standard_model_2.predict_proba(np.asarray(X_gb2))
    preds_2_3 = standard_model_3.predict_proba(np.asarray(X_gb2))

    preds_2 = (preds_2_1 + preds_2_2 + preds_2_3)/3

    t1_df.at[:, ['cc', 'cn','cg']] = (preds_1 + np.flip(preds_2, 1))/2 # use average
    t2_df.at[:, ['cc', 'cn','cg']] = (preds_2 + np.flip(preds_1, 1))/2 # use average
    
    #### situational ####
    
    meta1, action_id1, X_gb1, X_lstm1 = prep_for_predict(t1_df, situation=True, training=False)
    meta2, action_id2, X_gb2, X_lstm2 = prep_for_predict(t2_df, situation=True, training=False)
    
    _sit_cols = ['minute','period','team_strength','opp_strength','team_score','opp_score']
    cols = copy(_cols) + _sit_cols
    
    ## edit 11/9/2021 dropping columns
    X_gb1 = pd.DataFrame(X_gb1, columns=cols)
    X_gb2 = pd.DataFrame(X_gb2, columns=cols)
    X_gb1['avg_speed'] = X_gb1[['speed_0','speed_1','speed_2','speed_3']].mean(axis=1)
    X_gb1['total_dist'] = X_gb1[['distance_0','distance_1','distance_2','distance_3']].sum(axis=1)
    X_gb2['avg_speed'] = X_gb2[['speed_0','speed_1','speed_2','speed_3']].mean(axis=1)
    X_gb2['total_dist'] = X_gb2[['distance_0','distance_1','distance_2','distance_3']].sum(axis=1)
    X_gb1['man_adv'] = X_gb1['team_strength'].copy()-X_gb1['opp_strength'].copy()
    X_gb2['man_adv'] = X_gb2['team_strength'].copy()-X_gb2['opp_strength'].copy()
    X_gb1['score_diff'] = X_gb1['team_score'].copy() - X_gb1['opp_score'].copy()
    X_gb2['score_diff'] = X_gb2['team_score'].copy() - X_gb2['opp_score'].copy()
    cols.append('man_adv')
    cols.append('score_diff')
    cols.append('avg_speed')
    cols.append('total_dist')
    
    to_remove=['team_score','opp_score','play_pattern_id_0','play_pattern_id_1','play_pattern_id_2','ball_z_0','ball_z_1','ball_z_2','team_strength','opp_strength']
    to_remove.extend(['speed_0','speed_1','speed_2','speed_3','distance_0','distance_1','distance_2','distance_3','period'])

    X_gb1 = X_gb1.drop(columns=to_remove, errors='ignore')
    X_gb2 = X_gb2.drop(columns=to_remove, errors='ignore')

    cols = [c for c in cols if c not in to_remove]
    
    preds_1_1 = sit_model_1.predict_proba(np.asarray(X_gb1))
    preds_1_2 = sit_model_2.predict_proba(np.asarray(X_gb1))
    preds_1_3 = sit_model_3.predict_proba(np.asarray(X_gb1))

    preds_1 = (preds_1_1 + preds_1_2 + preds_1_3)/3
    
    preds_2_1 = sit_model_1.predict_proba(np.asarray(X_gb2))
    preds_2_2 = sit_model_2.predict_proba(np.asarray(X_gb2))
    preds_2_3 = sit_model_3.predict_proba(np.asarray(X_gb2))

    preds_2 = (preds_2_1 + preds_2_2 + preds_2_3)/3

    t1_df.at[:, ['sit_cc', 'sit_cn','sit_cg']] = (preds_1 + np.flip(preds_2, 1))/2 # use average
    t2_df.at[:, ['sit_cc', 'sit_cn','sit_cg']] = (preds_2 + np.flip(preds_1, 1))/2 # use average
    
    t1_df['xxG_diff'] = t1_df['cg'].copy()-t1_df['cc'].copy()
    t1_df['sit_xxG_diff'] = t1_df['sit_cg'].copy()-t1_df['sit_cc'].copy()
    # game['ns_cg_cc_diff'] = game['no_sit_cg'].copy()-game['no_sit_cc'].copy()
    
    # game['opponent_boost'] = game['cg_cc_diff'].copy()-game['no_cg_cc_diff']
    t1_df['sit_boost'] = t1_df['sit_xxG_diff'].copy()-t1_df['xxG_diff'].copy()
    
    t2_df['xxG_diff'] = t2_df['cg'].copy()-t2_df['cc'].copy()
    t2_df['sit_xxG_diff'] = t2_df['sit_cg'].copy()-t2_df['sit_cc'].copy()
    t2_df['sit_boost'] = t2_df['sit_xxG_diff'].copy()-t2_df['xxG_diff'].copy()
    
    # game = game.drop(columns=['cg_cc_diff','no_cg_cc_diff','ns_cg_cc_diff'])
    
    # reminder to nan prediction where action encoding is in leaky encodes
    leaky_indices = np.where(pd.Series(action_id1[:, -1]).isin(leaky_encodes), True, False)

    t1_df.loc[leaky_indices, ['cc','cn','cg','sit_cc','sit_cn','sit_cg','xxG_diff','sit_xxG_diff','sit_boost']] = np.nan
    t2_df.loc[leaky_indices, ['cc','cn','cg','sit_cc','sit_cn','sit_cg','xxG_diff','sit_xxG_diff','sit_boost']] = np.nan
    
    return t1_df, t2_df

def add_game_clock(game):
    
    game['time'] = game['minute'].copy() + (game['second'].copy()/60)
    to_add = game.groupby(['period'])['time'].max().to_dict()
    to_add[0] = 45
    game['previous_period'] = game['period'].copy() - 1
    game['to_add'] = game['previous_period'].map(to_add)
    game['time'] = game['time'].copy() + game['to_add'].copy() - 45
    
    return game.drop(columns=['previous_period','to_add'])


win_prob_columns = ['cg','cc','time','score_diff','score_total','man_adv']
def load_wp_model():
    clf = catboost.CatBoostClassifier()
    return clf.load_model(os.path.join(DROPBOX_PATH, 'models/win_prob/naive_win_prob'))

wp_model = load_wp_model()

def add_win_prob(game):
    
    # sample_game['in_possession'] = np.where(sample_game['team_id'].copy() == sample_game['possession_team_id'].copy(),1,0)
    game_X = game.copy()
    game_X['man_adv'] = game_X['team_strength'].copy() - game_X['opp_strength'].copy()
    game_X['score_diff'] = game_X['team_score'].copy() - game_X['opp_score'].copy()
    game_X['score_total'] = game_X['team_score'].copy() + game_X['opp_score'].copy()
    game_X = add_game_clock(game_X)
    
    game_X = game_X[win_prob_columns].copy()
    game[['prob_loss','prob_draw','prob_win']] = wp_model.predict_proba(game_X)
    
    return game



def csv_to_stf(df, match_id):

    # drop unneeded
    df = drop_meta(df)
    df = drop_leaky(df)

    # separate outcomes from actions
    df = separate_outcomes(df)

    # fill x and y
    df['x'] = df['x'].fillna(50)
    df['y'] = df['y'].fillna(50)

    # for a few reasons, give each team their own dataframe
    df1, team_1, df2, team_2 = split_STF(df)

    del df
    gc.collect()

    # feature engineer
    df1 = fe(df1)
    df2 = fe(df2)

    # add VAEP target
    df1 = add_target(df1)
    df2 = add_target(df2)
    
    # standardize column list
    df1 = add_missing_cols(df1)
    df2 = add_missing_cols(df2)
    
    df1 = add_man_adv(df1)
    df2 = add_man_adv(df2)

    # calculate descriptive action keys
    df1 = action_keys(df1)
    df2 = action_keys(df2)

    # efficient dtypes (also drop columns not needed for xxG)
    df1 = optimal_dtypes(df1)
    df2 = optimal_dtypes(df2)

    col_list = sorted(list(df1))
    df1 = df1[col_list]
    df2 = df2[col_list]
    
    # print(match_id)
#     df1 = add_opp_rating(df1, match_id, team_1)
#     df2 = add_opp_rating(df2, match_id, team_2)

    df1,df2 = predict_xxG(df1, df2)

    df1=add_win_prob(df1)
    df2=add_win_prob(df2)

    return df1, team_1, df2, team_2

