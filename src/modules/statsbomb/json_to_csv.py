
import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

DROPBOX_PATH = os.environ.get('DROPBOX_PATH')

##########################
###  Helper Functions  ###
##########################

def split_loc_arrays(df):
    """
    Statsbomb chooses to have locations in an array of x,y instead of separating them into x,y.
    Pandas doesn't love that format. I separate them here.
    """

    # split array types
    df.loc[:, 'end_location'] = np.nan
    df['end_location'] = df['end_location'].fillna(df['pass_end_location'])
    df['end_location'] = df['end_location'].fillna(df['carry_end_location'])

    array_cols = ['location','end_location']

    for col in array_cols:
        x_name = col.replace('location','x')
        y_name = col.replace('location','y')
        z_name = 'ball_z'
        df[col] = df[col].fillna({i: [None,None] for i in df[col].index})
        to_merge = pd.DataFrame(df[col].tolist())
        if to_merge.shape[1] == 3:
            df[[x_name, y_name, z_name]] = pd.DataFrame(df[col].tolist())
        else:
            df[[x_name, y_name]] = pd.DataFrame(df[col].tolist())

    df = df.drop(columns=['location','end_location','pass_end_location','carry_end_location'])

    return df


def combine_bp(df):

    """No need for multiple body part columns, all ids are the same."""

    df.loc[:,'body_part_id'] = np.nan
    df.loc[:,'body_part_name'] = np.nan

    to_drop = ['pass_body_part_id','pass_body_part_name','shot_body_part_id','shot_body_part_name']

    df['body_part_id'] = df['body_part_id'].copy().fillna(df['pass_body_part_id'])
    df['body_part_name'] = df['body_part_name'].copy().fillna(df['pass_body_part_name'])

    df['body_part_id'] = df['body_part_id'].copy().fillna(df['shot_body_part_id'])
    df['body_part_name'] = df['body_part_name'].copy().fillna(df['shot_body_part_name'])

    if 'clearance_body_part_id' in list(df):
        df['body_part_id'] = df['body_part_id'].copy().fillna(df['clearance_body_part_id'])
        df['body_part_name'] = df['body_part_name'].copy().fillna(df['clearance_body_part_name'])
        to_drop.append('clearance_body_part_id')
        to_drop.append('clearance_body_part_name')
    if 'goalkeeper_body_part_id' in list(df):
        df['body_part_id'] = df['body_part_id'].copy().fillna(df['goalkeeper_body_part_id'])
        df['body_part_name'] = df['body_part_name'].copy().fillna(df['goalkeeper_body_part_name'])
        to_drop.append('goalkeeper_body_part_id')
        to_drop.append('goalkeeper_body_part_name')

    df = df.drop(columns=to_drop)
    df['body_part_id'] = df['body_part_id'].copy().fillna(0)
    df['body_part_id'] = df['body_part_id'].copy().astype('Int16')

    # for some reason there are redundant columns on clearances
    clearance_cols = ['clearance_left_foot','clearance_right_foot','clearance_head']
    # check if one of those cols is in list
    if len(list(set(clearance_cols) - set(list(df)))) < 3:

        # assert they don't contain unique data
        bp_col_data_count = df.loc[df['type_name']=='Clearance']['body_part_name'].notnull().sum(axis=0)

        indv_data_count = 0

        clearance_cols = [cc for cc in clearance_cols if cc in list(df)]
        for indv_bp_col in clearance_cols:
            indv_data_count += df.loc[df['type_name']=='Clearance'][indv_bp_col].notnull().sum(axis=0)
        assert(bp_col_data_count>=indv_data_count)

        # drop unnecessary columns
        df = df.drop(columns=clearance_cols)

    return df



def combine_outcomes(df):

    """No need for multiple outcome columns, combine to save space"""

    outcome_names = [col for col in list(df) if 'outcome_name' in col]
    outcome_ids = [col for col in list(df) if 'outcome_id' in col]

    df.loc[:,'outcome_id'] = np.nan
    df.loc[:,'outcome_name'] = np.nan

    for on in outcome_names:
        df['outcome_name'] = df['outcome_name'].copy().fillna(df[on])
    for oid in outcome_ids:
        df['outcome_id'] = df['outcome_id'].copy().fillna(df[oid])

    to_drop = outcome_names + outcome_ids
    df = df.drop(columns=to_drop)
    df['outcome_id'] = df['outcome_id'].copy().fillna(0)
    df['outcome_id'] = df['outcome_id'].copy().astype('Int16')

    return df


def combine_stids(df):

    """No need for multiple secondary type id columns, combine to save space"""

    stid_names = [col for col in list(df) if '_type_name' in col]
    stid_ids = [col for col in list(df) if '_type_id' in col]

    df.loc[:,'secondary_type_id'] = np.nan
    df.loc[:,'secondary_type_name'] = np.nan

    for sn in stid_names:
        df['secondary_type_name'] = df['secondary_type_name'].copy().fillna(df[sn])
    for sid in stid_ids:
        df['secondary_type_id'] = df['secondary_type_id'].copy().fillna(df[sid])

    to_drop = stid_names + stid_ids
    df = df.drop(columns=to_drop)
    df['secondary_type_id'] = df['secondary_type_id'].copy().fillna(0)
    df['secondary_type_id'] = df['secondary_type_id'].copy().astype('Int16')

    return df


def combine_techniques(df):

    """No need for multiple technique columns, combine to save space"""

    tech_names = [col for col in list(df) if '_technique_name' in col]
    tech_ids = [col for col in list(df) if '_technique_id' in col]

    df.loc[:,'technique_id'] = np.nan
    df.loc[:,'technique_name'] = np.nan

    for tn in tech_names:
        df['technique_name'] = df['technique_name'].copy().fillna(df[tn])
    for tid in tech_ids:
        df['technique_id'] = df['technique_id'].copy().fillna(df[tid])

    to_drop = tech_names + tech_ids
    df = df.drop(columns=to_drop)
    df['technique_id'] = df['technique_id'].copy().fillna(0)
    df['technique_id'] = df['technique_id'].copy().astype('Int16')

    # drop redundant columns
    pass_techs = list(df.loc[df['type_name']=='Pass']['technique_name'].unique())
    pass_techs = ['pass_'+pt.lower().replace(' ','_') for pt in pass_techs if pt is not np.nan]
    df = df.drop(columns=pass_techs)

    return df


def id2int(df):

    """
    Convert id cols to integers to save space
    """

    id_cols = [col for col in list(df) if 'id' in col]

    # non ints (uuids)
    non_ints = ['id', 'pass_assisted_shot_id', 'shot_key_pass_id']

    # save space
    int16s = ['type_id','play_pattern_id','position_id','pass_height_id','foul_committed_card_id','bad_behaviour_card_id',
              'goalkeeper_position_id','period','minute','second','possession']

    # now merged, might want to unmerge at some point
    # other_int16s = ['pass_outcome_id','ball_receipt_outcome_id','shot_outcome_id','goalkeeper_outcome_id',
    #'dribble_outcome_id','interception_outcome_id','substitution_outcome_id','50_50_outcome_id','duel_outcome_id']

    # normal (above 30,000) (up to 2 billion)
    int32s = ['match_id','possession_team_id','team_id','player_id','pass_recipient_id','substitution_replacement_id']

    to_remove = list(set(int16s) - set(id_cols))
    for tr in to_remove:
        int16s.remove(tr)
    if 'substitution_replacement_id' not in df:
        df['substitution_replacement_id'] = 0
    df[int16s+int32s] = df[int16s+int32s].fillna(0)

    df[int16s] = df[int16s].copy().astype('Int16')
    df[int32s] = df[int32s].copy().astype(int)

    return df

def json_to_csv(df):
    """
    All processing functions in one place
    """
    # more conventional column name
    col_list = list(df)
    df.columns = [c.replace('.','_') for c in col_list]
    
    if 'pass_end_location' not in list(df):
        print("Missing pass end locations, wait on SB to update")
        success = False 
        return df, success
    else:
        success = True
    df = split_loc_arrays(df)
    df = combine_bp(df)
    df = combine_outcomes(df)
    df = combine_stids(df)
    df = combine_techniques(df)
    df = id2int(df)

    return df, success

