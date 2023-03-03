"""

easiest to handle as own file, although could be dissected into data_process and data_model_prep with some work

"""

import os
import json
import math
import time
import pickle
import random
import urllib
import catboost
import simdkalman

import numpy as np
import pandas as pd
import requests as req

from copy import copy
from glob import glob
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,mean_squared_error
from fuzzywuzzy import fuzz, process
from sklearn.metrics import make_scorer
from datetime import datetime, timedelta
from bayes_opt import BayesianOptimization
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

from multiprocessing import Process

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


DROPBOX_PATH = 'C:\\Users\Blake\G Street Dropbox\Blake Atkinson\shared_soccer_data\data'

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
def load_schedules():
    
    normal = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/processed_schedule.csv'))
    stf = pd.read_csv(os.path.join(DROPBOX_PATH, 'schedules/stf_schedule.csv'))

    normal['home_team_name'] = normal['home_team_id'].apply(lambda x: teams.get(x)['name'])
    normal['away_team_name'] = normal['away_team_id'].apply(lambda x: teams.get(x)['name'])

    stf['team_name'] = stf['team_id'].apply(lambda x: teams.get(x)['name'])
    stf['opp_team_name'] = stf['opp_team_id'].apply(lambda x: teams.get(x)['name'])
    
    normal['datetime_UTC'] = pd.to_datetime(normal['datetime_UTC'].copy())
    stf['datetime_UTC'] = pd.to_datetime(stf['datetime_UTC'].copy())

    normal['match_date_UTC'] = normal['datetime_UTC'].copy().dt.date
    stf['match_date_UTC'] = stf['datetime_UTC'].copy().dt.date
    
    normal['last_updated'] = pd.to_datetime(normal['last_updated'].copy())
    stf['last_updated'] = pd.to_datetime(stf['last_updated'].copy())
    
    normal = normal.loc[~normal['match_status'].isin(['deleted','collecting','cancelled','postponed'])].reset_index(drop=True)
    stf = stf.loc[~stf['match_status'].isin(['deleted','collecting','cancelled','postponed'])].reset_index(drop=True)
    
    return normal, stf 




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
    'competition_name',
    'avg_potential', 'home_url', 'home_image', 'home_name', 'away_url', 'away_image', 'away_name'
]





def load_footy():
    
    footy = pd.read_csv(os.path.join(DROPBOX_PATH, 'footy/aggregated.csv'))
    footy['datetime_UTC'] = pd.to_datetime(footy['datetime_UTC'].copy())
    footy['match_date_UTC'] = pd.to_datetime(footy['match_date_UTC'].copy())
    footy = footy.drop_duplicates(subset=['id']).reset_index(drop=True)
    footy = footy.loc[footy['match_date_UTC']<=datetime.utcnow()+pd.Timedelta(days=14)].reset_index(drop=True)
    footy = footy.sort_values(by=['match_date_UTC','id']).reset_index(drop=True)
    footy = footy[cols+['datetime_UTC','match_date_UTC']]
    footy = footy.loc[footy['match_date_UTC']>=pd.to_datetime('2006-08-01')].reset_index(drop=True) # one outlier date
    print(f"Footy API has {len(footy)} matches")
    return footy


def to_STF(tdata):
    
    tdata = tdata.rename(columns={
        'homeGoalCount':'team_a_goals',
        'awayGoalCount':'team_b_goals',
        'homeID':'team_a_id',
        'awayID':'team_b_id'
    })
    
    unchanged_cols = ['match_date_UTC', 'game_week']
    homes = tdata.copy()
    aways = tdata.copy()
    homes.columns=[col.replace('team_a_','team_') for col in list(homes)]
    aways.columns=[col.replace('team_b_','team_') for col in list(aways)]
    homes.columns=[col.replace('team_b_','opp_') for col in list(homes)]
    aways.columns=[col.replace('team_a_','opp_') for col in list(aways)]
    
    homes['is_home'] = 1
    aways['is_home'] = 0
    tdata = pd.concat([homes, aways], axis=0).sort_values(by=['match_date_UTC','team_id']).reset_index(drop=True)
    
    return tdata

def prep_footy_data(fdata, teams_footy2id):
    
    stf = to_STF(fdata)
    stf = stf.loc[stf['status']=='complete'].reset_index(drop=True)
    
    # replace empty fields with nans
    for col in ['team_shots','opp_shots','team_shotsOnTarget','opp_shotsOnTarget','team_possession','opp_possession']:
        stf[col] = stf[col].copy().replace(-1, np.nan)

    for col in ['team_attacks','opp_attacks','team_dangerous_attacks','opp_dangerous_attacks','team_xg','opp_xg',
            'odds_ft_1','odds_ft_2','odds_ft_x']:
        stf[col] = stf[col].copy().replace(0, np.nan)

    stats = ['goals','shots','shotsOnTarget','possession','attacks','dangerous_attacks','xg']
    
    for stat in stats:
        stf[f'{stat}_diff'] = stf[f'team_{stat}'].copy() - stf[f'opp_{stat}'].copy()
        
    pace_stats = ['goals','shots','shotsOnTarget','attacks','dangerous_attacks','xg']
#     pace_thresholds = [2.5, 22, 10, 206, 102,3.05]
    
    for i,ps in enumerate(pace_stats):
#         pace_threshold = pace_thresholds[i]
        stf[f'{ps}_total'] = stf[f'team_{ps}'].copy()+stf[f'opp_{ps}'].copy()
        
    stf =stf.rename(columns={'team_id':'footy_team_id','opp_id':'footy_opp_id'})
    stf['team_id'] =stf['footy_team_id'].map(teams_footy2id)
    stf['opp_team_id'] =stf['footy_opp_id'].map(teams_footy2id)
    
    return stf


def merge_game_ids(past, fresults, schedule):
    
    sb_matches = past.copy()[['match_date_UTC','datetime_UTC','match_id','team_id','opp_team_id']]
    
    sb_matches['match_date_UTC'] = pd.to_datetime(sb_matches['match_date_UTC'])

    sb_matches = sb_matches.drop_duplicates(subset=['match_id','team_id'])
    num_possible = len(sb_matches)
    
    combined = fresults.merge(sb_matches, how='left', on=['match_date_UTC','team_id','opp_team_id'])
    
    ## try fillna with one day higher (helps for 2,000 north american games)
    sb_matches2 = sb_matches.copy()
    sb_matches2['match_date_UTC'] = sb_matches2['match_date_UTC'] + pd.Timedelta(hours=24)
    combined2 = fresults.merge(sb_matches2, how='left', on=['match_date_UTC', 'team_id', 'opp_team_id'])
    combined['match_id'] = combined['match_id'].copy().fillna(combined2['match_id'].copy())
    combined['team_id'] = combined['team_id'].copy().fillna(combined2['team_id'].copy())
    combined['opp_team_id'] = combined['opp_team_id'].copy().fillna(combined2['opp_team_id'].copy())

    ## try fillna with one day lower (japanese games maybe?)
    sb_matches2 = sb_matches.copy()
    sb_matches2['match_date_UTC'] = sb_matches2['match_date_UTC'] - pd.Timedelta(hours=24)
    combined2 = fresults.merge(sb_matches2, how='left', on=['match_date_UTC', 'team_id', 'opp_team_id'])
    combined['match_id'] = combined['match_id'].copy().fillna(combined2['match_id'].copy())
    combined['team_id'] = combined['team_id'].copy().fillna(combined2['team_id'].copy())
    combined['opp_team_id'] = combined['opp_team_id'].copy().fillna(combined2['opp_team_id'].copy())
        
    combined = combined.sort_values(by=['match_date_UTC','match_id','is_home'], ascending=[True, True, False])
    total_added = len(combined.loc[combined['match_id'].notnull()])
    print(f"{np.round(total_added/num_possible, 3)*100}% of SB games matched")
    missing = schedule.loc[~schedule['match_id'].isin(list(combined.match_id.unique()))].reset_index(drop=True)
    
    return combined, missing

def add_sb_metrics(data, stf_schedule):
    
    game_vecs = pd.read_csv(os.path.join(DROPBOX_PATH, 'Statsbomb/game_vecs/game_vecs.csv'))

#     data = data.merge(game_vecs[['match_id','team_id',#'score_diff',# for testing merge
#                                  'xxG_diff','obv_diff','pace','obv_pace']], how='left', on=['match_id','team_id'])
    data = data.rename(columns={
        'competition_id':'footy_comp_id',
        'competition_name':'footy_comp_name'
    })
    before = len(data)
    data = data.merge(game_vecs, how='outer', on=['match_id','team_id'])
    after = len(data)
    print(f"{after-before} games with statsbomb stats but no footy")
    
    meta = stf_schedule.copy()[['match_id','team_id','competition_id','season_id','match_date_UTC','datetime_UTC','opp_team_id','is_upcoming']]
    meta = meta.rename(columns={'match_date_UTC':'backup_match_date'})
    meta = meta.rename(columns={'datetime_UTC':'backup_datetime'})
    meta = meta.rename(columns={'opp_team_id':'backup_opp_team_id'})
    data = data.merge(meta, how='left', on=['match_id','team_id'])
#     data = data.rename(columns={
# #         'score_diff':'test', #todo: figure out why a handful of games have different score stats
#         'xxG_diff':'xxg_side',
#         'pace':'xxg_total',
#         'obv_diff':'obv_side',
#         'obv_pace':'obv_total'
#     })
    data['match_date_UTC'] = data['match_date_UTC'].fillna(data['backup_match_date'].copy())
    data['datetime_UTC'] = data['datetime_UTC'].fillna(data['backup_datetime'].copy())
    data['opp_team_id'] = data['opp_team_id'].fillna(data['backup_opp_team_id'].copy())
    data = data.drop(columns=['backup_match_date', 'backup_datetime'])
    
    
    return data

def processing_steps(data, schedule):
    
    ## try to get agreement on attendance data
    data.loc[data['attendance']==-1, 'attendance'] = np.nan
    data = data.rename(columns={'attendance':'footy_attendance'})
    data = data.merge(schedule.copy().rename(columns={'attendance':'sb_attendance'})[[
        'match_id','sb_attendance'
    ]], how='left', on=['match_id'])
    ## make the attendance columns agree
    data['attendance'] = data[['footy_attendance','sb_attendance']].mean(axis=1)
    # 22 instances of this, and they are just small attendance numbers
    data.loc[(data['footy_attendance']==0)&data['sb_attendance'].notnull(), 'attendance'] = data.sb_attendance
    data['attendance'] = data['attendance'].fillna(data['sb_attendance'].copy())
    data['attendance'] = data['attendance'].fillna(data['footy_attendance'].copy())
    data['max_attendance'] = data.groupby(['stadium_name'])['attendance'].transform('max')
    data['pct_attendance'] = data['attendance'].copy()/data['max_attendance'].copy()
    data = data.drop(columns=['sb_attendance','footy_attendance','max_attendance'])
    
    ## for forward looking version, we will use market odds
    # remove vig and drop one
    data['inv_odds_ft_1'] = 1/data['odds_ft_1'].copy()
    data['inv_odds_ft_x'] = 1/data['odds_ft_x'].copy()
    data['inv_odds_ft_2'] = 1/data['odds_ft_2'].copy()
    data['inv_odds_total'] = data[['inv_odds_ft_1','inv_odds_ft_x','inv_odds_ft_2']].copy().sum(axis=1)
    data['odds_ft_1'] = 1/(data['inv_odds_ft_1'].copy()/data['inv_odds_total'].copy())
    data['odds_ft_x'] = 1/(data['inv_odds_ft_x'].copy()/data['inv_odds_total'].copy())
    data['odds_ft_2'] = 1/(data['inv_odds_ft_2'].copy()/data['inv_odds_total'].copy())
    data['market_home_pct'] = data['inv_odds_ft_1'].copy()/data['inv_odds_total'].copy()
    data['market_draw_pct'] = data['inv_odds_ft_x'].copy()/data['inv_odds_total'].copy()
    data['market_away_pct'] = data['inv_odds_ft_2'].copy()/data['inv_odds_total'].copy()

    data = data.drop(columns=['odds_ft_2']) # only need 2 of 3 way ML, last is implied
    
    
    ## sb not in team/opp format, instead in diff/total format for key stats. recovering team/opp here.
    target_pred_cols = ['score_diff', 'xG_diff', 'shot_diff', 'sot_diff','xxG_diff','obv_diff'] # man_adv_v2
    tm_col_name_map = {
        'score_diff':'sb_team_goals',
        'xG_diff':'sb_team_xG',
        'shot_diff':'sb_team_shots',
        'sot_diff':'sb_team_SOTs',
        'xxG_diff':'sb_team_xxG',
        'obv_diff':'sb_team_obv'
    }

    tm_total_col_name = {
        'score_diff':'score_total',
        'xG_diff':'xG_total',
        'shot_diff':'shot_total',
        'sot_diff':'sot_total',
        'xxG_diff':'pace',
        'obv_diff':'obv_pace'
    }

    predict_cols = []
    for tpc in tqdm(target_pred_cols):
        tm_col_name = tm_col_name_map[tpc]
        opp_col_name = tm_col_name.replace('team','opp')
        predict_cols.extend([tm_col_name, opp_col_name])
        total_col_name = tm_total_col_name[tpc]

        data['tpc_temp'] = (data[total_col_name].copy()+1)*(data[tpc].copy())
        data[tm_col_name] = (data['tpc_temp'].copy()+data[total_col_name].copy())/2
        data[opp_col_name] = data[total_col_name].copy()-data[tm_col_name].copy()
        data = data.drop(columns=['tpc_temp'])
        
    ## handful of instances where we're missing data from one source or the other
    col_fillnas = {
        'datetime_UTC':'footy_datetime_UTC',
        'sb_team_goals':'team_goals',
        'sb_opp_goals':'opp_goals',
        'sb_team_xG':'team_xg',
        'sb_opp_xG':'opp_xg',
        'sb_team_shots':'team_shots',
        'sb_opp_shots':'opp_shots',
        'sb_team_SOTs':'team_shotsOnTarget',
        'sb_opp_SOTs':'opp_shotsOnTarget'
    }

    for k,v in col_fillnas.items():
        data[k] = data[k].copy().fillna(data[v].copy())
        data[v] = data[v].copy().fillna(data[k].copy())


    ## standardize season ids
    data = data.drop(columns=['footy_datetime_UTC'])
    season_map = data.groupby(['season'])['season_id'].apply(lambda x: pd.Series.mode(x)).reset_index().drop(columns='level_1').set_index('season_id').to_dict()['season']
    data['backup_season'] = data['season_id'].map(season_map)
    data['season'] = data['season'].fillna(data['backup_season'].copy())
    data = data.drop(columns=['backup_season','season_id'])
    return data




def gg1():
    ## gg 1: non market
    print("Creating non market game grades...")
    ### data prep ###
    teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
    
    schedule, stf_schedule = load_schedules()
    teams_id2footy = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_id2footy'))
    teams_footy2id = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_footy2id'))
    footy = load_footy()
    footy = prep_footy_data(footy.copy(), teams_footy2id)

    df, missing = merge_game_ids(stf_schedule.copy(), footy.copy().rename(columns={'datetime_UTC':'footy_datetime_UTC'}), schedule.copy())

    df = add_sb_metrics(df, stf_schedule)
    df = df.rename(columns={'id':'footy_match_id'})

    df = processing_steps(df, schedule.copy())

    sb_data = df.copy().loc[df['match_id'].notnull()].reset_index(drop=True)
    sb_data = sb_data.merge(schedule[['match_id','season_id']], how='left')
    non_sb_data = df.copy().loc[df['match_id'].isnull()].reset_index(drop=True)


    stats = [
        'score_diff', 'xG_diff', 'shot_diff', 'sq_diff', 'sot_diff', 'score_total', 'xG_total', 'shot_total', 'sq_total', 'sot_total', 'sit_boost', 'pace', 'obv_pace', 'xxG_diff', 'pct_xxG_diff', 'obv_diff', 'pct_obv_diff', 'cgoal_skew', 'cgoal_kurt', 'cgoal_sum', 'cgoal_std', 'cconcede_skew', 'cconcede_kurt', 'cconcede_sum', 'cconcede_std', 'xxG_conversion', 'xG_conversion', 'opp_xG_conversion', 'opp_xxG_conversion', 'win_prob', 'draw_prob', 'opp_dt_eff', 'opp_mt_eff', 'opp_at_eff', 'opp_d_qual', 'opp_mid_qual', 'opp_atck_qual', 'team_dt_eff', 'team_mt_eff', 'team_at_eff', 'team_d_qual', 'team_mid_qual', 'team_atck_qual', 'opp_gk_pass_loc', 'opp_def_pass_loc', 'opp_mid_pass_loc', 'opp_atck_pass_loc', 'opp_gk_pass_angle', 'opp_def_pass_angle', 'opp_mid_pass_angle', 'opp_atck_pass_angle', 'opp_gk_pass_count', 'opp_def_pass_count', 'opp_mid_pass_count', 'opp_atck_pass_count', 'opp_gk_pass_dist', 'opp_def_pass_dist', 'opp_mid_pass_dist', 'opp_atck_pass_dist', 'opp_gk_pass_obv_vol', 'opp_def_pass_obv_vol', 'opp_mid_pass_obv_vol', 'opp_atck_pass_obv_vol', 'opp_gk_pass_obv_eff', 'opp_def_pass_obv_eff', 'opp_mid_pass_obv_eff', 'opp_atck_pass_obv_eff', 'team_gk_pass_loc', 'team_def_pass_loc', 'team_mid_pass_loc', 'team_atck_pass_loc', 'team_gk_pass_angle', 'team_def_pass_angle', 'team_mid_pass_angle', 'team_atck_pass_angle', 'team_gk_pass_count', 'team_def_pass_count', 'team_mid_pass_count', 'team_atck_pass_count', 'team_gk_pass_dist', 'team_def_pass_dist', 'team_mid_pass_dist', 'team_atck_pass_dist', 'team_gk_pass_obv_vol', 'team_def_pass_obv_vol', 'team_mid_pass_obv_vol', 'team_atck_pass_obv_vol', 'team_gk_pass_obv_eff', 'team_def_pass_obv_eff', 'team_mid_pass_obv_eff', 'team_atck_pass_obv_eff', 'opp_gk_carry_angle', 'opp_def_carry_angle', 'opp_mid_carry_angle', 'opp_atck_carry_angle', 'opp_gk_carry_count', 'opp_def_carry_count', 'opp_mid_carry_count', 'opp_atck_carry_count', 'opp_gk_carry_dist', 'opp_def_carry_dist', 'opp_mid_carry_dist', 'opp_atck_carry_dist', 'opp_gk_carry_vol', 'opp_def_carry_vol', 'opp_mid_carry_vol', 'opp_atck_carry_vol', 'opp_gk_carry_eff', 'opp_def_carry_eff', 'opp_mid_carry_eff', 'opp_atck_carry_eff', 'team_gk_carry_angle', 'team_def_carry_angle', 'team_mid_carry_angle', 'team_atck_carry_angle', 'team_gk_carry_dist', 'team_def_carry_dist', 'team_mid_carry_dist', 'team_atck_carry_dist', 'team_gk_carry_count', 'team_def_carry_count', 'team_mid_carry_count', 'team_atck_carry_count', 'team_gk_carry_vol', 'team_def_carry_vol', 'team_mid_carry_vol', 'team_atck_carry_vol', 'team_gk_carry_eff', 'team_def_carry_eff', 'team_mid_carry_eff', 'team_atck_carry_eff', 'opp_gk_dfn_pos', 'opp_def_dfn_pos', 'opp_mid_dfn_pos', 'opp_atck_dfn_pos', 'opp_gk_dfn_obv_vol', 'opp_def_dfn_obv_vol', 'opp_mid_dfn_obv_vol', 'opp_atck_dfn_obv_vol', 'opp_gk_dfn_obv_eff', 'opp_def_dfn_obv_eff', 'opp_mid_dfn_obv_eff', 'opp_atck_dfn_obv_eff', 'opp_gk_dfn_xxG_vol', 'opp_def_dfn_xxG_vol', 'opp_mid_dfn_xxG_vol', 'opp_atck_dfn_xxG_vol', 'team_gk_dfn_pos', 'team_def_dfn_pos', 'team_mid_dfn_pos', 'team_atck_dfn_pos', 'team_gk_dfn_obv_vol', 'team_def_dfn_obv_vol', 'team_mid_dfn_obv_vol', 'team_atck_dfn_obv_vol', 'team_gk_dfn_obv_eff', 'team_def_dfn_obv_eff', 'team_mid_dfn_obv_eff', 'team_atck_dfn_obv_eff', 'team_gk_dfn_xxG_vol', 'team_def_dfn_xxG_vol', 'team_mid_dfn_xxG_vol', 'team_atck_dfn_xxG_vol', 'team_weighted_fouls', 'opp_weighted_fouls', 'opp_corners', 'team_corners', 'team_crosses', 'opp_crosses', 'team_cross_pct', 'opp_cross_pct', 'team_pens', 'opp_pens', 'carry_lengths', 'gk_dist', 'fields_gained', 'fields_gained_comp', 'med_def_action', 'threat_pp', 'dthreat_pp', 'off_embed_0', 'off_embed_1', 'off_embed_2', 'off_embed_3', 'off_embed_4', 'off_embed_5', 'def_embed_0', 'def_embed_1', 'def_embed_2', 'def_embed_3', 'def_embed_4', 'def_embed_5', 'opp_poss', 'opp_ppp', 'opp_spp', 'team_poss', 'team_ppp', 'team_spp', 'team_poss_start', 'opp_poss_start', 'team_poss_len', 'opp_poss_len', 'team_poss_width', 'opp_poss_width', 'team_poss_time_sum', 'opp_poss_time_sum', 'team_poss_time_median', 'opp_poss_time_median', 'oxG_f3', 'txG_f3', 'pct_lead', 'pct_tied', 'pct_trail', 'man_adv', 'team_dx_sec', 'team_xxG_sec', 'opp_dx_sec', 'opp_xxG_sec', 'opp_d3_passes', 'opp_d3_comp%', 'opp_d3_fcomp%', 'opp_m3_passes', 'opp_m3_comp%', 'opp_m3_fcomp%', 'opp_a3_passes', 'opp_a3_comp%', 'opp_a3_fcomp%', 'team_d3_passes', 'team_d3_comp%', 'team_d3_fcomp%', 'team_m3_passes', 'team_m3_comp%', 'team_m3_fcomp%', 'team_a3_passes', 'team_a3_comp%', 'team_a3_fcomp%', 'opp_SOT%', 'team_save%', 'team_SOT%', 'opp_save%', 'team_XGOT/SOT', 'opp_XGOT/SOT', 'opp_end_d3', 'opp_end_m3', 'opp_end_a3', 'team_end_d3', 'team_end_m3', 'team_end_a3', 'team_switches', 'opp_switches', 'opp_d3_press', 'opp_m3_press', 'opp_a3_press', 'team_d3_press', 'team_m3_press', 'team_a3_press', 'opp_wide_poss', 'team_wide_poss',
    ]
    stats+=[
    'team_goals', 'opp_goals', 'attacks_recorded', 'team_yellow_cards', 'opp_yellow_cards', 'team_red_cards', 'opp_red_cards', 'team_shotsOnTarget', 'opp_shotsOnTarget', 'team_shotsOffTarget', 'opp_shotsOffTarget', 'team_shots', 'opp_shots', 'team_fouls', 'opp_fouls', 'team_possession', 'opp_possession', 'team_offsides', 'opp_offsides', 'team_dangerous_attacks', 'opp_dangerous_attacks', 'team_attacks', 'opp_attacks', 'team_xg', 'opp_xg', 'total_xg', 'team_penalties_won', 'opp_penalties_won', 'team_penalty_goals', 'opp_penalty_goals', 'team_penalty_missed', 'opp_penalty_missed', 'team_throwins', 'opp_throwins', 'team_freekicks', 'opp_freekicks', 'team_goalkicks', 'opp_goalkicks'
    ]
    stats+=[
        'goals_total', 'shots_total', 'shotsOnTarget_total', 'attacks_total', 'dangerous_attacks_total', 'xg_total'
    ]

    stats+=[
        'avg_potential','goals_diff', 'shots_diff', 'shotsOnTarget_diff', 'possession_diff', 'attacks_diff', 'dangerous_attacks_diff', 'xg_diff','no_home_away','is_home'
    ]

    # leaky for what happened in past
    # stats+=[
    #     'odds_ft_1', 'odds_ft_x', 'odds_ft_2'
    # ]

    # found to not be useful # these were useful for pace
    # 171           team_pens    0.000000
    # 172            opp_pens    0.000000
    # 283  team_penalties_won    0.000000
    for stat in ['opp_penalty_goals','team_penalty_goals','team_penalties_won','opp_penalties_won','opp_penalty_missed','attacks_recorded','goals_total']:
        stats.remove(stat)

    total_non_sb = len(non_sb_data)
    # drop cols where 93% nulls
    threshold = int(0.93*total_non_sb)
    footy_stats = stats.copy()
    to_drop = list(non_sb_data[stats].isnull().sum()[non_sb_data[stats].isnull().sum()>threshold].index)
    for td in to_drop:
        footy_stats.remove(td)
    print(non_sb_data.shape)
    non_sb_data = non_sb_data.drop(columns=to_drop)
    print(non_sb_data.shape)
    non_sb_data = non_sb_data.drop(columns=['team_id','opp_team_id','match_id','competition_id','is_upcoming'])
    print(list(non_sb_data))
    def prep_non_sb_data(data):
    
        data['target_temp'] = data['team_goals'].copy()-data['opp_goals'].copy()
        data['target_2_temp'] = data['team_goals'].copy()+data['opp_goals'].copy()
        data['target_num_games'] = data.groupby(['footy_team_id','season'])['target_temp'].transform('count')
        data['target_szn_sum'] = data.groupby(['footy_team_id','season'])['target_temp'].transform('sum')
        data['target_2_szn_sum'] = data.groupby(['footy_team_id','season'])['target_2_temp'].transform('sum')
        data['target'] = (data['target_szn_sum'].copy()-data['target_temp'])/(data['target_num_games'].copy()-1)
        data['target_2'] = (data['target_2_szn_sum'].copy()-data['target_2_temp'])/(data['target_num_games'].copy()-1)
        
        data = data.drop(columns=['target_temp','target_2_temp','target_num_games','target_szn_sum','target_2_szn_sum'])
        data = data.dropna(subset=['target','target_2'])
        
        return data


    non_sb_data = prep_non_sb_data(non_sb_data)

    def prep_sb_data(data):
        
        data['target_temp'] = data['team_goals'].copy()-data['opp_goals'].copy()
        data['target_2_temp'] = data['team_goals'].copy()+data['opp_goals'].copy()
        data['target_num_games'] = data.groupby(['team_id','season_id'])['target_temp'].transform('count')
        data['target_szn_sum'] = data.groupby(['team_id','season_id'])['target_temp'].transform('sum')
        data['target_2_szn_sum'] = data.groupby(['team_id','season_id'])['target_2_temp'].transform('sum')
        data['target'] = (data['target_szn_sum'].copy()-data['target_temp'])/(data['target_num_games'].copy()-1)
        data['target_2'] = (data['target_2_szn_sum'].copy()-data['target_2_temp'])/(data['target_num_games'].copy()-1)
        data = data.drop(columns=['target_temp','target_2_temp','target_num_games','target_szn_sum','target_2_szn_sum'])
        data = data.dropna(subset=['target'])
        
        return data

    sb_data = prep_sb_data(sb_data)

    sb_data = sb_data.drop(columns=['season_id'])
    non_sb_data = non_sb_data.drop(columns=['season'])

    def add_game_grades_non_sb(data):
        
        # # ### for getting feature importance
        X = data[footy_stats+['target','target_2']].copy()
        # X = X.sample(frac=1)
        y1 = X['target'].copy()
        y2 = X['target_2'].copy()
        X = X.drop(columns=['target','target_2'])

        eval_cv = KFold(3, shuffle=True, random_state=17)
        prod_cv = KFold(10, shuffle=True, random_state=17)
        reg = catboost.CatBoostRegressor(verbose=False)
        # 08/23/22 Score diff cross val 0.17366522630199602
        print("Score diff cross val", np.mean(cross_val_score(reg, X, y1, cv=eval_cv)))
        # # option 1
        reg = catboost.CatBoostRegressor(verbose=False)
        reg.fit(X, y1)
        feat_importances = pd.DataFrame({
            'stats':footy_stats,
            'importance':reg.feature_importances_
        }).sort_values(by='importance',ascending=False)
        print(feat_importances)

        data['game_score'] = cross_val_predict(reg, X, y1, cv=prod_cv)
        reg = catboost.CatBoostRegressor(verbose=False)
        
        # 08/23/22 Pace cross val 0.23575666835862108
        print("Pace cross val", np.mean(cross_val_score(reg, X, y2, cv=eval_cv)))
        data['pace_score'] = cross_val_predict(reg, X, y2, cv=prod_cv)
        
        reg = catboost.CatBoostRegressor(verbose=False)
        reg.fit(X, y2)
        feat_importances = pd.DataFrame({
            'stats':footy_stats,
            'importance':reg.feature_importances_
        }).sort_values(by='importance',ascending=False)
        print(feat_importances)
        
        return data


    non_sb_data = add_game_grades_non_sb(non_sb_data)

    def add_game_grades_sb(data, show_score=False):
        
        # # ### for getting feature importance
        X = data[stats+['target','target_2']].copy()
        # X = X.sample(frac=1)
        y1 = X['target'].copy()
        y2 = X['target_2'].copy()
        X = X.drop(columns=['target','target_2'])

        eval_cv = KFold(3, shuffle=True, random_state=17)
    #     np.mean(cross_val_score(reg, X, y1, cv=eval_cv))
        prod_cv = KFold(6, shuffle=True, random_state=17)
        
    #     sd_params = {'iterations': 19524,
    #      'od_wait': 1270,
    #      'learning_rate': 0.011194201649297929,
    #      'reg_lambda': 44.63399671384952,
    #      'subsample': 0.7496120387317952,
    #      'random_strength': 33.75970628327746,
    #      'depth': 6,
    #      'min_data_in_leaf': 18,
    #      'leaf_estimation_iterations': 5}
        reg = catboost.CatBoostRegressor(verbose=False)#, **sd_params)
        # # option 2
        # Score diff cross val 0.2784195893187421 # did better without opt?
        if show_score:
            print("Score diff cross val", np.mean(cross_val_score(reg, X, y1, cv=eval_cv)))
        # # option 1
        reg = catboost.CatBoostRegressor(verbose=False)#,**sd_params)
        reg.fit(X, y1)
        feat_importances = pd.DataFrame({
            'stats':stats,
            'importance':reg.feature_importances_
        }).sort_values(by='importance',ascending=False)
        print(feat_importances)

        data['game_score'] = cross_val_predict(reg, X, y1, cv=prod_cv)
        reg = catboost.CatBoostRegressor(verbose=False)
    #     Pace cross val 0.33348896960501095
        if show_score:
            print("Pace cross val", np.mean(cross_val_score(reg, X, y2, cv=eval_cv)))
        data['pace_score'] = cross_val_predict(reg, X, y2, cv=prod_cv)
        
        reg = catboost.CatBoostRegressor(verbose=False)
        reg.fit(X, y2)
        feat_importances = pd.DataFrame({
            'stats':stats,
            'importance':reg.feature_importances_
        }).sort_values(by='importance',ascending=False)
        print(feat_importances)
        
        return data


    sb_data = add_game_grades_sb(sb_data)

    non_sb_scores = non_sb_data.copy()[['footy_match_id','footy_team_id','footy_opp_id','match_date_UTC',
                                'game_score','pace_score']]
    sb_game_scores = sb_data.copy()[['footy_match_id','footy_team_id','footy_opp_id','match_date_UTC','match_id','team_id','opp_team_id',
                                'game_score','pace_score']]

    game_scores = pd.concat([sb_game_scores, non_sb_scores], axis=0).sort_values(by='match_date_UTC').reset_index(drop=True)

    def prep_for_network(df):
        
        opp_data = df.copy().rename(columns={'id':'opp_id','opp_id':'id'})
        opp_data = opp_data.rename(columns={'game_score':'opp_game_score','pace_score':'opp_pace_score'})

        df = df.merge(opp_data[['match_date','footy_match_id','id','opp_id']+['opp_game_score','opp_pace_score']], how='left', on=['match_date','footy_match_id','id','opp_id'])
        # # just do two networks for now (later more)
        df['rtg_avg'] = (df['game_score'].copy() + (-1*df['opp_game_score'].copy()))/2
        df['pace_avg'] = (df['pace_score'].copy() + (df['opp_pace_score'].copy()))/2


        min_date = df.match_date.min()
        # # helper col
        df['rating_period'] = df['match_date'].copy().rank(method='dense').astype(int)
        df['date_since_inception'] = (df['match_date'].copy()-min_date).dt.days
        df['days_since_last'] = df['date_since_inception'].diff()
        df['game_no'] = df.groupby(['id'])['footy_match_id'].transform('cumcount')
        df['opp_game_no'] = df.groupby(['opp_id'])['footy_match_id'].transform('cumcount')
        df = df.drop_duplicates(subset=['footy_match_id']).reset_index(drop=True)
        
        ## only using statsbomb teams
        df['backup_team_id'] = df['id'].map(teams_footy2id)
        df['backup_opp_id'] = df['opp_id'].map(teams_footy2id)
        df['team_id'] = df['team_id'].fillna(df['backup_team_id'].copy())
        df['opp_team_id'] = df['opp_team_id'].fillna(df['backup_opp_id'].copy())
        df = df.drop(columns=['backup_team_id','backup_opp_id'])

        
        return df

    game_scores = game_scores.rename(columns={'footy_team_id':'id', 'footy_opp_id':'opp_id','match_date_UTC':'match_date'})
    game_scores = prep_for_network(game_scores)

    # create network
    df = game_scores.copy().dropna(subset=['team_id','opp_team_id','rtg_avg','pace_avg'])
    df['days_since_last'] = df['days_since_last'].fillna(1)# one case
    df = df.drop(columns=['id','opp_id'])
    teams = list(set(list(df.team_id.unique())+list(df.opp_team_id.unique())))
    num_teams = len(teams)
    teams.sort()
    team_map = {}
    for idx, team in enumerate(teams):
        team_map[team] = idx
        
    prank_mat = np.zeros((num_teams,num_teams))
    neg_prank_mat = np.zeros((num_teams,num_teams))
    pace_mat = np.ones((num_teams, num_teams))*2.5
    pace_std = np.ones((num_teams, num_teams))
    rating_periods = list(df.rating_period.unique())
    df.match_date = pd.to_datetime(df.match_date.copy())
    df = df.sort_values(by='match_date')


    def calc_ratings(protag_matrix):
        
        N = biggest_index = protag_matrix.shape[0]    
        d = 5e-3
        A = (d * protag_matrix + (1 - d) / N)    
        v = np.repeat(1/biggest_index, biggest_index)
        for i in range(150):
            v = A@v
            norm = np.linalg.norm(v)
            v = v/norm
        
        return v

    history = []
    for rp in tqdm(rating_periods):
        rating_update = np.zeros((num_teams, num_teams))
        neg_rating_update = np.zeros((num_teams, num_teams))
        pace_update = np.zeros((num_teams, num_teams))
        pace_std_update = np.zeros((num_teams, num_teams))
        rp_data = df.copy().loc[df.rating_period==rp].reset_index(drop=True)
        days_since = rp_data.days_since_last.copy().max()
        if days_since < 1:
            days_since = 1
        
        # time decay the matrix
        prank_mat *= np.exp(-(1/150)*days_since)
        neg_prank_mat *= np.exp(-(1/150)*days_since)
        pace_mat *= np.exp(-(1/125)*days_since)
        pace_std *= np.exp(-(1/125)*days_since)
        
        ratings_vec = calc_ratings(prank_mat)
        neg_ratings_vec = calc_ratings(neg_prank_mat)
        
        pace_vec = calc_ratings(pace_mat.copy()/pace_std.copy())
    #     ratings_entr = entropy(prank_mat)
    #     pace_entr = entropy(pace_mat.copy()/pace_std.copy())
        for index, row in rp_data.iterrows():
            match_id = row['match_id']
            protag_id = row['team_id']
            antag_id = row['opp_team_id']
            protag_index = team_map[protag_id]
            antag_index = team_map[antag_id]
            ## grab ratings going into games
            protag_rating = ratings_vec[protag_index] 
            antag_rating = ratings_vec[antag_index]
            protag_neg_rating = neg_ratings_vec[protag_index]
            antag_neg_rating = neg_ratings_vec[antag_index]
    #         protag_entr = ratings_entr[protag_index]
    #         antag_entr = ratings_entr[antag_index]
            protag_pace = pace_vec[protag_index]
            antag_pace = pace_vec[antag_index]
    #         protag_pentr = pace_entr[protag_index]
    #         antag_pentr = pace_entr[antag_index]
            history.append([match_id, protag_id, antag_id, protag_rating, antag_rating, protag_neg_rating,antag_neg_rating,#protag_entr, antag_entr,
                            protag_pace, antag_pace]) #, protag_pentr, antag_pentr])
            
            # update
            game_rating = row['rtg_avg']
            pace_rating = row['pace_avg']
            if np.isnan(game_rating):
                continue
            if np.isnan(pace_rating):
                continue
                
            if game_rating > 0:
                rating_update[protag_index][antag_index]+=np.abs(game_rating)
                neg_rating_update[antag_index][protag_index]+=np.abs(game_rating)
            else:
                rating_update[antag_index][protag_index]+=np.abs(game_rating)
                neg_rating_update[protag_index][antag_index]+=np.abs(game_rating)
            pace_update[protag_index][antag_index] += pace_rating
            pace_update[antag_index][protag_index] += pace_rating
            pace_std_update[protag_index][antag_index] += 1
            pace_std_update[antag_index][protag_index] += 1
            
        prank_mat+=rating_update
        neg_prank_mat+=neg_rating_update
        pace_mat += pace_update
        pace_std+= pace_std_update

    history = pd.DataFrame(history, columns=['match_id','id','opp_id','rating','opp_rating','neg_rating','neg_opp_rating',#'entropy','opp_entropy',
                                            'pace','opp_pace'])#,'pace_entropy','opp_pace_entropy'])
    history = history.merge(stf_schedule[['match_id','team_id','score','opp_score']].rename(columns={'team_id':'id'}),
                        how='left', on=['match_id','id'])

    history['score_diff'] = history['score'].copy()-history['opp_score'].copy()
    history['ratings_diff'] = history['rating'].copy()-history['opp_rating'].copy()
    history['ratings_v2'] = history['rating'].copy()-history['neg_rating'].copy()
    history['opp_ratings_v2'] = history['opp_rating'].copy()-history['neg_opp_rating'].copy()
    history['rtg_diff_v2'] = history['ratings_v2'].copy()-history['opp_ratings_v2'].copy()
    history['score_total'] = history['score'].copy()+history['opp_score'].copy()
    history['pace_comb'] = history['pace'].copy()+history['opp_pace'].copy()

    # 
    to_save = history.copy().dropna(subset=['match_id']).reset_index(drop=True)
    to_save = to_save.drop(columns=['score','opp_score','score_diff','score_total'])
    to_save = to_save.rename(columns={'id':'team_id','opp_id':'opp_team_id'})
    to_save_opp = to_save.copy().reset_index(drop=True)
    to_save_opp.columns=['match_id','opp_team_id','team_id','opp_rating','rating','neg_opp_rating','neg_rating',
                        #'opp_entropy','entropy',
                        'opp_pace','pace',
                        #'opp_pace_entropy','pace_entropy',
                        'ratings_diff','opp_ratings_v2','ratings_v2','rtg_diff_v2','pace_comb']
    to_save_opp = to_save_opp[list(to_save)]
    to_save_opp['ratings_diff'] = to_save_opp['ratings_diff'].copy()*-1
    to_save_opp['rtg_diff_v2'] = to_save_opp['rtg_diff_v2'].copy()*-1
    to_save = pd.concat([to_save, to_save_opp], axis=0).reset_index(drop=True)
    to_save.sort_values(by=['match_id'])
    to_save = to_save.drop(columns=['opp_rating','rating','neg_opp_rating','neg_rating','ratings_diff'])

    def get_current_ratings():
        
        current_rtgs = []
        for protag_id,protag_index, in team_map.items():
            protag_index = team_map[protag_id]
            ## grab ratings going into games
            protag_rating = ratings_vec[protag_index] 
            protag_neg_rating = neg_ratings_vec[protag_index]
            #protag_entr = ratings_entr[protag_index]
            protag_pace = pace_vec[protag_index]
            #protag_pentr = pace_entr[protag_index]
            
            current_rtgs.append([protag_id, protag_rating, protag_neg_rating, protag_pace]) #, protag_entr, protag_pace, protag_pentr])
        
        return pd.DataFrame(current_rtgs, columns=['team_id','rating','neg_rating','pace'])#,'entropy','pace','pace_entropy'])
    #     protag_id = row['team_id']

    current_ratings = get_current_ratings()
    opp_current_ratings = current_ratings.copy()
    opp_current_ratings.columns=['opp_team_id','opp_rating','neg_opp_rating','opp_pace']#'opp_entropy','opp_pace','opp_pace_entropy']
    upc = stf_schedule.copy().loc[stf_schedule['is_upcoming']==1].reset_index(drop=True)[['match_id','team_id','opp_team_id']]


    
    upc = upc.merge(current_ratings.copy(), how='left', on=['team_id'])
    upc = upc.merge(opp_current_ratings.copy(), how='left', on='opp_team_id')
    # upc['ratings_diff'] = upc['rating'].copy()-upc['opp_rating'].copy()
    upc['ratings_v2'] = upc['rating'].copy()-upc['neg_rating'].copy()
    upc['opp_ratings_v2'] = upc['opp_rating'].copy()-upc['neg_opp_rating'].copy()
    upc['rtg_diff_v2'] = upc['ratings_v2'].copy()-upc['opp_ratings_v2'].copy()
    upc = upc.drop(columns=['opp_rating','rating','neg_opp_rating','neg_rating'])
    upc['pace_comb'] = upc['pace'].copy()+upc['opp_pace'].copy()
    to_save = pd.concat([to_save, upc], axis=0).reset_index(drop=True)

    to_save.to_csv(os.path.join(DROPBOX_PATH, 'team_ratings/grade_network_no_mkt.csv'), index=False)

    return


def gg2():

    ## gg 2: market gg
    print("Creating market game grades...")
    ### data prep ###
    teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
    
    schedule, stf_schedule = load_schedules()
    teams_id2footy = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_id2footy'))
    teams_footy2id = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_footy2id'))
    footy = load_footy()
    footy = prep_footy_data(footy.copy(), teams_footy2id)

    df, missing = merge_game_ids(stf_schedule.copy(), footy.copy().rename(columns={'datetime_UTC':'footy_datetime_UTC'}), schedule.copy())

    df = add_sb_metrics(df, stf_schedule)
    df = df.rename(columns={'id':'footy_match_id'})

    df = processing_steps(df, schedule.copy())

    # remove vig and drop one
    # remove vig
    df['inv_odds_ft_1'] = 1/df['odds_ft_1'].copy()
    df['inv_odds_ft_x'] = 1/df['odds_ft_x'].copy()
    # df['inv_odds_ft_2'] = 1/df['odds_ft_2'].copy()
    df['inv_odds_total'] = df[['inv_odds_ft_1','inv_odds_ft_x','inv_odds_ft_2']].copy().sum(axis=1)
    df['odds_ft_1'] = 1/(df['inv_odds_ft_1'].copy()/df['inv_odds_total'].copy())
    df['odds_ft_x'] = 1/(df['inv_odds_ft_x'].copy()/df['inv_odds_total'].copy())
    # df['odds_ft_2'] = 1/(df['inv_odds_ft_2'].copy()/df['inv_odds_total'].copy())

    # df = df.drop(columns=['odds_ft_2']) # only need 2 of 3 way ML, last is implied

    sb_data = df.copy().loc[df['match_id'].notnull()].reset_index(drop=True)
    sb_data = sb_data.merge(schedule[['match_id','season_id']], how='left')
    non_sb_data = df.copy().loc[df['match_id'].isnull()].reset_index(drop=True)


    """

    very weird bug when i try to train with dropping games remaining <= 5
    so this is a workaround

    """
    def prep_non_sb_data(data):
        
        data['target_1_temp'] = data['team_goals'].copy()-data['opp_goals'].copy()
        data['target_2_temp'] = data['team_goals'].copy()+data['opp_goals'].copy()
        data['target_num_games'] = data.groupby(['footy_team_id','season'])['target_1_temp'].transform('count')
        data['target_1_szn_sum'] = data.groupby(['footy_team_id','season'])['target_1_temp'].transform('sum')
        data['target_2_szn_sum'] = data.groupby(['footy_team_id','season'])['target_2_temp'].transform('sum')
    #     print(data[['target_1_temp','target_2_temp','target_num_games','target_1_szn_sum','target_2_szn_sum']].isnull().sum())
        # now using these as backup for end of season
        data['target_1'] = (data['target_1_szn_sum'].copy()-data['target_1_temp'])/(data['target_num_games'].copy()-1)
        data['target_2'] = (data['target_2_szn_sum'].copy()-data['target_2_temp'])/(data['target_num_games'].copy()-1)
        data = data.drop(columns=['target_1_temp','target_2_temp','target_num_games','target_1_szn_sum','target_2_szn_sum'])
    #     data = data.dropna(subset=['target'])
        
        
        data['datetime_UTC'] = pd.to_datetime(data['datetime_UTC'] )
        data = data.sort_values(by=['datetime_UTC'])
        data['games_in_season'] = data.groupby(['footy_team_id','season'])['footy_match_id'].transform('count')
        data['szn_already_played'] = data.groupby(['footy_team_id','season'])['footy_match_id'].transform('cumcount')
        data['games_remaining'] = data['games_in_season'].copy()-data['szn_already_played'].copy()
        # print(list(non_sb_data))
        # sns.displot(data.sample(frac=0.05)['games_remaining'])
        data['score_diff_temp'] = data['team_goals'].copy()-data['opp_goals'].copy()
        data['sd_season'] = data.groupby(['footy_team_id','season'])['score_diff_temp'].transform('sum')
        data['sd_already'] = data.groupby(['footy_team_id','season'])['score_diff_temp'].transform('cumsum')
        data['sd_ROS'] = data['sd_season'].copy()-data['sd_already'].copy()

        data['score_total_temp'] = data['team_goals'].copy()+data['opp_goals'].copy()
        data['st_season'] = data.groupby(['footy_team_id','season'])['score_total_temp'].transform('sum')
        data['st_already'] = data.groupby(['footy_team_id','season'])['score_total_temp'].transform('cumsum')
        data['st_ROS'] = data['st_season'].copy()-data['st_already'].copy()

        data['side_target'] = data['sd_ROS'].copy()/data['games_remaining'].copy()
        data['total_target'] = data['st_ROS'].copy()/data['games_remaining'].copy()

        data = data.drop(columns=['games_in_season','szn_already_played','sd_season','st_season','sd_already','st_already',
                                'st_ROS','sd_ROS','score_diff_temp','score_total_temp'])
        
        # if less than 2 games remaining, use backup target
        data['side_target'] = np.where(data['games_remaining'] <= 2, data['target_1'].copy(), data['side_target'].copy())
        data['total_target'] = np.where(data['games_remaining'] <= 2, data['target_2'].copy(), data['total_target'].copy())
        
    #     print(data[['side_target','total_target','target_1','target_2']].corr())
        data = data.drop(columns=['target_1','target_2'])
        data = data.dropna(subset=['side_target','total_target'])
        
        return data.reset_index(drop=True)

    def prep_sb_data(data):
        
        ## first get backup target
        data['target_1_temp'] = data['team_goals'].copy()-data['opp_goals'].copy()
        data['target_2_temp'] = data['team_goals'].copy()+data['opp_goals'].copy()
        data['target_num_games'] = data.groupby(['team_id','season_id'])['target_1_temp'].transform('count')
        data['target_1_szn_sum'] = data.groupby(['team_id','season_id'])['target_1_temp'].transform('sum')
        data['target_2_szn_sum'] = data.groupby(['team_id','season_id'])['target_2_temp'].transform('sum')
    #     print(data[['target_1_temp','target_2_temp','target_num_games','target_1_szn_sum','target_2_szn_sum']].isnull().sum())
        # now using these as backup for end of season
        data['target_1'] = (data['target_1_szn_sum'].copy()-data['target_1_temp'])/(data['target_num_games'].copy()-1)
        data['target_2'] = (data['target_2_szn_sum'].copy()-data['target_2_temp'])/(data['target_num_games'].copy()-1)
        data = data.drop(columns=['target_1_temp','target_2_temp','target_num_games','target_1_szn_sum','target_2_szn_sum'])
        
        data['datetime_UTC'] = pd.to_datetime(data['datetime_UTC'] )
        data = data.sort_values(by=['datetime_UTC'])
        data['games_in_season'] = data.groupby(['team_id','season_id'])['match_id'].transform('count')
        data['szn_already_played'] = data.groupby(['team_id','season_id'])['match_id'].transform('cumcount')
        data['games_remaining'] = data['games_in_season'].copy()-data['szn_already_played'].copy()
        # print(list(non_sb_data))
        # sns.displot(data.sample(frac=0.05)['games_remaining'])
        data['score_diff_temp'] = data['team_goals'].copy()-data['opp_goals'].copy()
        data['sd_season'] = data.groupby(['team_id','season_id'])['score_diff_temp'].transform('sum')
        data['sd_already'] = data.groupby(['team_id','season_id'])['score_diff_temp'].transform('cumsum')
        data['sd_ROS'] = data['sd_season'].copy()-data['sd_already'].copy()

        data['score_total_temp'] = data['team_goals'].copy()+data['opp_goals'].copy()
        data['st_season'] = data.groupby(['footy_team_id','season'])['score_total_temp'].transform('sum')
        data['st_already'] = data.groupby(['footy_team_id','season'])['score_total_temp'].transform('cumsum')
        data['st_ROS'] = data['st_season'].copy()-data['st_already'].copy()

        data['side_target'] = data['sd_ROS'].copy()/data['games_remaining'].copy()
        data['total_target'] = data['st_ROS'].copy()/data['games_remaining'].copy()
        
    #     print(data.loc[((data['competition_id']==2)&(data['season_id']==90))][['games_in_season','szn_already_played','games_remaining']])
    #     print(data.loc[((data['competition_id']==2)&(data['season_id']==90))][['sd_season','sd_already','sd_ROS']])
    #     print(data.loc[((data['competition_id']==2)&(data['season_id']==90))][['st_season','st_already','st_ROS']])
        
        data = data.drop(columns=['games_in_season','szn_already_played','sd_season','st_season','sd_already','st_already',
                                'st_ROS','sd_ROS','score_diff_temp','score_total_temp'])

        data = data.dropna(subset=['side_target','total_target'])
        
        # if less than 2 games remaining, use backup target
        data['side_target'] = np.where(data['games_remaining'] <= 2, data['target_1'].copy(), data['side_target'].copy())
        data['total_target'] = np.where(data['games_remaining'] <= 2, data['target_2'].copy(), data['total_target'].copy())
        
    #     print(data[['side_target','total_target','target_1','target_2']].corr())
        data = data.drop(columns=['target_1','target_2'])
        data = data.dropna(subset=['side_target','total_target'])
        
        return data

        
    non_sb_data = prep_non_sb_data(non_sb_data)
    sb_data = prep_sb_data(sb_data)


    # sb_data['man_adv'].sum()
    # sns.displ[ot(sb_data['man_adv'])
    stats = [
        'score_diff', 'xG_diff', 'shot_diff', 'sq_diff', 'sot_diff', 'score_total', 'xG_total', 'shot_total', 'sq_total', 'sot_total', 'sit_boost', 'pace', 'obv_pace', 'xxG_diff', 'pct_xxG_diff', 'obv_diff', 'pct_obv_diff', 'cgoal_skew', 'cgoal_kurt', 'cgoal_sum', 'cgoal_std', 'cconcede_skew', 'cconcede_kurt', 'cconcede_sum', 'cconcede_std', 'xxG_conversion', 'xG_conversion', 'opp_xG_conversion', 'opp_xxG_conversion', 'win_prob', 'draw_prob', 'opp_dt_eff', 'opp_mt_eff', 'opp_at_eff', 'opp_d_qual', 'opp_mid_qual', 'opp_atck_qual', 'team_dt_eff', 'team_mt_eff', 'team_at_eff', 'team_d_qual', 'team_mid_qual', 'team_atck_qual', 'opp_gk_pass_loc', 'opp_def_pass_loc', 'opp_mid_pass_loc', 'opp_atck_pass_loc', 'opp_gk_pass_angle', 'opp_def_pass_angle', 'opp_mid_pass_angle', 'opp_atck_pass_angle', 'opp_gk_pass_count', 'opp_def_pass_count', 'opp_mid_pass_count', 'opp_atck_pass_count', 'opp_gk_pass_dist', 'opp_def_pass_dist', 'opp_mid_pass_dist', 'opp_atck_pass_dist', 'opp_gk_pass_obv_vol', 'opp_def_pass_obv_vol', 'opp_mid_pass_obv_vol', 'opp_atck_pass_obv_vol', 'opp_gk_pass_obv_eff', 'opp_def_pass_obv_eff', 'opp_mid_pass_obv_eff', 'opp_atck_pass_obv_eff', 'team_gk_pass_loc', 'team_def_pass_loc', 'team_mid_pass_loc', 'team_atck_pass_loc', 'team_gk_pass_angle', 'team_def_pass_angle', 'team_mid_pass_angle', 'team_atck_pass_angle', 'team_gk_pass_count', 'team_def_pass_count', 'team_mid_pass_count', 'team_atck_pass_count', 'team_gk_pass_dist', 'team_def_pass_dist', 'team_mid_pass_dist', 'team_atck_pass_dist', 'team_gk_pass_obv_vol', 'team_def_pass_obv_vol', 'team_mid_pass_obv_vol', 'team_atck_pass_obv_vol', 'team_gk_pass_obv_eff', 'team_def_pass_obv_eff', 'team_mid_pass_obv_eff', 'team_atck_pass_obv_eff', 'opp_gk_carry_angle', 'opp_def_carry_angle', 'opp_mid_carry_angle', 'opp_atck_carry_angle', 'opp_gk_carry_count', 'opp_def_carry_count', 'opp_mid_carry_count', 'opp_atck_carry_count', 'opp_gk_carry_dist', 'opp_def_carry_dist', 'opp_mid_carry_dist', 'opp_atck_carry_dist', 'opp_gk_carry_vol', 'opp_def_carry_vol', 'opp_mid_carry_vol', 'opp_atck_carry_vol', 'opp_gk_carry_eff', 'opp_def_carry_eff', 'opp_mid_carry_eff', 'opp_atck_carry_eff', 'team_gk_carry_angle', 'team_def_carry_angle', 'team_mid_carry_angle', 'team_atck_carry_angle', 'team_gk_carry_dist', 'team_def_carry_dist', 'team_mid_carry_dist', 'team_atck_carry_dist', 'team_gk_carry_count', 'team_def_carry_count', 'team_mid_carry_count', 'team_atck_carry_count', 'team_gk_carry_vol', 'team_def_carry_vol', 'team_mid_carry_vol', 'team_atck_carry_vol', 'team_gk_carry_eff', 'team_def_carry_eff', 'team_mid_carry_eff', 'team_atck_carry_eff', 'opp_gk_dfn_pos', 'opp_def_dfn_pos', 'opp_mid_dfn_pos', 'opp_atck_dfn_pos', 'opp_gk_dfn_obv_vol', 'opp_def_dfn_obv_vol', 'opp_mid_dfn_obv_vol', 'opp_atck_dfn_obv_vol', 'opp_gk_dfn_obv_eff', 'opp_def_dfn_obv_eff', 'opp_mid_dfn_obv_eff', 'opp_atck_dfn_obv_eff', 'opp_gk_dfn_xxG_vol', 'opp_def_dfn_xxG_vol', 'opp_mid_dfn_xxG_vol', 'opp_atck_dfn_xxG_vol', 'team_gk_dfn_pos', 'team_def_dfn_pos', 'team_mid_dfn_pos', 'team_atck_dfn_pos', 'team_gk_dfn_obv_vol', 'team_def_dfn_obv_vol', 'team_mid_dfn_obv_vol', 'team_atck_dfn_obv_vol', 'team_gk_dfn_obv_eff', 'team_def_dfn_obv_eff', 'team_mid_dfn_obv_eff', 'team_atck_dfn_obv_eff', 'team_gk_dfn_xxG_vol', 'team_def_dfn_xxG_vol', 'team_mid_dfn_xxG_vol', 'team_atck_dfn_xxG_vol', 'team_weighted_fouls', 'opp_weighted_fouls', 'opp_corners', 'team_corners', 'team_crosses', 'opp_crosses', 'team_cross_pct', 'opp_cross_pct', 'team_pens', 'opp_pens', 'carry_lengths', 'gk_dist', 'fields_gained', 'fields_gained_comp', 'med_def_action', 'threat_pp', 'dthreat_pp', 'off_embed_0', 'off_embed_1', 'off_embed_2', 'off_embed_3', 'off_embed_4', 'off_embed_5', 'def_embed_0', 'def_embed_1', 'def_embed_2', 'def_embed_3', 'def_embed_4', 'def_embed_5', 'opp_poss', 'opp_ppp', 'opp_spp', 'team_poss', 'team_ppp', 'team_spp', 'team_poss_start', 'opp_poss_start', 'team_poss_len', 'opp_poss_len', 'team_poss_width', 'opp_poss_width', 'team_poss_time_sum', 'opp_poss_time_sum', 'team_poss_time_median', 'opp_poss_time_median', 'oxG_f3', 'txG_f3', 'pct_lead', 'pct_tied', 'pct_trail', 'man_adv', 'team_dx_sec', 'team_xxG_sec', 'opp_dx_sec', 'opp_xxG_sec', 'opp_d3_passes', 'opp_d3_comp%', 'opp_d3_fcomp%', 'opp_m3_passes', 'opp_m3_comp%', 'opp_m3_fcomp%', 'opp_a3_passes', 'opp_a3_comp%', 'opp_a3_fcomp%', 'team_d3_passes', 'team_d3_comp%', 'team_d3_fcomp%', 'team_m3_passes', 'team_m3_comp%', 'team_m3_fcomp%', 'team_a3_passes', 'team_a3_comp%', 'team_a3_fcomp%', 'opp_SOT%', 'team_save%', 'team_SOT%', 'opp_save%', 'team_XGOT/SOT', 'opp_XGOT/SOT', 'opp_end_d3', 'opp_end_m3', 'opp_end_a3', 'team_end_d3', 'team_end_m3', 'team_end_a3', 'team_switches', 'opp_switches', 'opp_d3_press', 'opp_m3_press', 'opp_a3_press', 'team_d3_press', 'team_m3_press', 'team_a3_press', 'opp_wide_poss', 'team_wide_poss',
    ]
    stats+=[
    'team_goals', 'opp_goals', 'attacks_recorded', 'team_yellow_cards', 'opp_yellow_cards', 'team_red_cards', 'opp_red_cards', 'team_shotsOnTarget', 'opp_shotsOnTarget', 'team_shotsOffTarget', 'opp_shotsOffTarget', 'team_shots', 'opp_shots', 'team_fouls', 'opp_fouls', 'team_possession', 'opp_possession', 'team_offsides', 'opp_offsides', 'team_dangerous_attacks', 'opp_dangerous_attacks', 'team_attacks', 'opp_attacks', 'team_xg', 'opp_xg', 'total_xg', 'team_penalties_won', 'opp_penalties_won', 'team_penalty_goals', 'opp_penalty_goals', 'team_penalty_missed', 'opp_penalty_missed', 'team_throwins', 'opp_throwins', 'team_freekicks', 'opp_freekicks', 'team_goalkicks', 'opp_goalkicks'
    ]
    stats+=[
        'goals_total', 'shots_total', 'shotsOnTarget_total', 'attacks_total', 'dangerous_attacks_total', 'xg_total'
    ]

    stats+=[
        'avg_potential','goals_diff', 'shots_diff', 'shotsOnTarget_diff', 'possession_diff', 'attacks_diff', 'dangerous_attacks_diff', 'xg_diff','no_home_away','is_home'
    ]

    # leaky for what happened in past
    ## but fixed this notebook
    stats+=[
        'odds_ft_1', 'odds_ft_x','games_remaining'
    ]

    # found to not be useful # these were useful for pace
    # 171           team_pens    0.000000
    # 172            opp_pens    0.000000
    # 283  team_penalties_won    0.000000
    for stat in ['opp_penalty_goals','team_penalty_goals','team_penalties_won','opp_penalties_won','opp_penalty_missed','attacks_recorded','goals_total']:
        stats.remove(stat)
        
    total_non_sb = len(non_sb_data)
    # drop cols where 93% nulls
    threshold = int(0.93*total_non_sb)
    footy_stats = stats.copy()
    to_drop = list(non_sb_data[stats].isnull().sum()[non_sb_data[stats].isnull().sum()>threshold].index)
    for td in to_drop:
        footy_stats.remove(td)
    non_sb_data = non_sb_data.drop(columns=to_drop)
    non_sb_data = non_sb_data.drop(columns=['team_id','opp_team_id','match_id','competition_id','is_upcoming'])

    non_sb_data = non_sb_data.drop_duplicates(subset=['footy_match_id','footy_team_id'])

    def add_game_grades_non_sb(data, show_score=False, show_fi=False):
    
        # # ### for getting feature importance
        X = data[footy_stats+['side_target','total_target']].copy()
        # X = X.sample(frac=1)
        y1 = X['side_target'].copy()
        y2 = X['total_target'].copy()
        X = X.drop(columns=['side_target','total_target'])
        y1_tr = y1[X['games_remaining']>=5]
        y2_tr = y2[X['games_remaining']>=5]
        X_tr = X.copy().loc[X['games_remaining']>=5].reset_index(drop=True)
        
        X_tr=X_tr.drop(columns=['games_remaining'])
        
        eval_cv = KFold(3, shuffle=True, random_state=17)
        prod_cv = KFold(8, shuffle=True, random_state=17)
        
        if show_score:
            reg = catboost.CatBoostRegressor(verbose=False)
            # 08/23/22 Score diff cross val 0.17366522630199602
            print("Score diff cross val", np.mean(cross_val_score(reg, X_tr, y1_tr, cv=eval_cv)))
        # # option 1
        stat_display = footy_stats.copy()
        stat_display.remove('games_remaining')
        
        params = {
            
        }

        if show_fi:
            reg = catboost.CatBoostRegressor(verbose=False)
            reg.fit(X_tr, y1_tr)
            feat_importances = pd.DataFrame({
                'stats':stat_display,
                'importance':reg.feature_importances_
            }).sort_values(by='importance',ascending=False)
            print(feat_importances)
            
        X = X.drop(columns=['games_remaining'])
        reg = catboost.CatBoostRegressor(verbose=False, **params)
    #     reg = xgb.XGBRegressor()
        data['game_score'] = cross_val_predict(reg, X, y1, cv=prod_cv)
        
        reg = catboost.CatBoostRegressor(verbose=False, **params)
        data['pace_score'] = cross_val_predict(reg, X, y2, cv=prod_cv)
    
        
        return data


    # Score diff cross val 0.1750731599982609 without odds
    # Score diff cross val 0.174051975593478513 with odds
    # 08/23/22 Pace cross val 0.23575666835862108 without odds
    # Pace cross val 0.12337012842172541 with odds
    non_sb_data = add_game_grades_non_sb(non_sb_data)

    def add_game_grades_sb(data, show_score=False, show_fi=False):
    
        # # ### for getting feature importance
        X = data[stats+['side_target','total_target']].copy()
        # X = X.sample(frac=1)
        y1 = X['side_target'].copy()
        y2 = X['total_target'].copy()
        X = X.drop(columns=['side_target','total_target'])
        y1_tr = y1[X['games_remaining']>=5]
        y2_tr = y2[X['games_remaining']>=5]
        X_tr = X.copy().loc[X['games_remaining']>=5].reset_index(drop=True)
        
        eval_cv = KFold(3, shuffle=True, random_state=17)
        prod_cv = KFold(6, shuffle=True, random_state=17)
        X_tr = X_tr.drop(columns=['games_remaining'])
        if show_score:
            reg = catboost.CatBoostRegressor(verbose=False)
            # 08/23/22 Score diff cross val 0.17366522630199602
            print("Score diff cross val", np.mean(cross_val_score(reg, X_tr, y1_tr, cv=eval_cv)))
        # # option 1
        stat_display = stats.copy()
        stat_display.remove('games_remaining')
        if show_fi:
            reg = catboost.CatBoostRegressor(verbose=False)
            reg.fit(X_tr, y1_tr)
            feat_importances = pd.DataFrame({
                'stats':stat_display,
                'importance':reg.feature_importances_
            }).sort_values(by='importance',ascending=False)
            print(feat_importances)
            
        X = X.drop(columns=['games_remaining'])
        reg = catboost.CatBoostRegressor(verbose=False)
        data['game_score'] = cross_val_predict(reg, X, y1, cv=prod_cv)
        
        reg = catboost.CatBoostRegressor(verbose=False)
        data['pace_score'] = cross_val_predict(reg, X, y2, cv=prod_cv)
        
        return data

    sb_data = add_game_grades_sb(sb_data)

    
    non_sb_scores = non_sb_data.copy()[['footy_match_id','footy_team_id','footy_opp_id','match_date_UTC',
                                'game_score','pace_score']]
    sb_game_scores = sb_data.copy()[['footy_match_id','footy_team_id','footy_opp_id','match_date_UTC','match_id','team_id','opp_team_id',
                                'game_score','pace_score']]

    game_scores = pd.concat([sb_game_scores, non_sb_scores], axis=0).sort_values(by='match_date_UTC').reset_index(drop=True)


    def prep_for_network(df):
        
        opp_data = df.copy().rename(columns={'id':'opp_id','opp_id':'id'})
        opp_data = opp_data.rename(columns={'game_score':'opp_game_score','pace_score':'opp_pace_score'})

        df = df.merge(opp_data[['match_date','footy_match_id','id','opp_id']+['opp_game_score','opp_pace_score']], how='left', on=['match_date','footy_match_id','id','opp_id'])
        # # just do two networks for now (later more)
        df['rtg_avg'] = (df['game_score'].copy() + (-1*df['opp_game_score'].copy()))/2
        df['pace_avg'] = (df['pace_score'].copy() + (df['opp_pace_score'].copy()))/2


        min_date = df.match_date.min()
        # # helper col
        df['rating_period'] = df['match_date'].copy().rank(method='dense').astype(int)
        df['date_since_inception'] = (df['match_date'].copy()-min_date).dt.days
        df['days_since_last'] = df['date_since_inception'].diff()
        df['game_no'] = df.groupby(['id'])['footy_match_id'].transform('cumcount')
        df['opp_game_no'] = df.groupby(['opp_id'])['footy_match_id'].transform('cumcount')
        df = df.drop_duplicates(subset=['footy_match_id']).reset_index(drop=True)
        
        ## only using statsbomb teams
        df['backup_team_id'] = df['id'].map(teams_footy2id)
        df['backup_opp_id'] = df['opp_id'].map(teams_footy2id)
        df['team_id'] = df['team_id'].fillna(df['backup_team_id'].copy())
        df['opp_team_id'] = df['opp_team_id'].fillna(df['backup_opp_id'].copy())
        df = df.drop(columns=['backup_team_id','backup_opp_id'])

        
        return df

    game_scores = game_scores.rename(columns={'footy_team_id':'id', 'footy_opp_id':'opp_id','match_date_UTC':'match_date'})
    game_scores = prep_for_network(game_scores)

    # create network
    df = game_scores.copy().dropna(subset=['team_id','opp_team_id','rtg_avg','pace_avg'])
    df['days_since_last'] = df['days_since_last'].fillna(1)# one case
    df = df.drop(columns=['id','opp_id'])
    teams = list(set(list(df.team_id.unique())+list(df.opp_team_id.unique())))
    num_teams = len(teams)
    teams.sort()
    team_map = {}
    for idx, team in enumerate(teams):
        team_map[team] = idx
        
    prank_mat = np.zeros((num_teams,num_teams))
    neg_prank_mat = np.zeros((num_teams,num_teams))
    pace_mat = np.ones((num_teams, num_teams))*2.5
    pace_std = np.ones((num_teams, num_teams))
    rating_periods = list(df.rating_period.unique())
    df.match_date = pd.to_datetime(df.match_date.copy())
    df = df.sort_values(by='match_date')


    def calc_ratings(protag_matrix):
        
        N = biggest_index = protag_matrix.shape[0]    
        d = 5e-4
        A = (d * protag_matrix + (1 - d) / N)
        v = np.repeat(1/biggest_index, biggest_index)
        for i in range(150):
            v = A@v
            norm = np.linalg.norm(v)
            v = v/norm
        
        return v

    history = []
    for rp in tqdm(rating_periods):
        rating_update = np.zeros((num_teams, num_teams))
        neg_rating_update = np.zeros((num_teams, num_teams))
        pace_update = np.zeros((num_teams, num_teams))
        pace_std_update = np.zeros((num_teams, num_teams))
        rp_data = df.copy().loc[df.rating_period==rp].reset_index(drop=True)
        days_since = rp_data.days_since_last.copy().max()
        if days_since < 1:
            days_since = 1
        
        # time decay the matrix
        prank_mat *= np.exp(-(1/150)*days_since)
        neg_prank_mat *= np.exp(-(1/150)*days_since)
        pace_mat *= np.exp(-(1/125)*days_since)
        pace_std *= np.exp(-(1/125)*days_since)
        
        ratings_vec = calc_ratings(prank_mat)
        neg_ratings_vec = calc_ratings(neg_prank_mat)
        
        pace_vec = calc_ratings(pace_mat.copy()/pace_std.copy())
        ratings_entr = entropy(prank_mat)
        pace_entr = entropy(pace_mat.copy()/pace_std.copy())
        for index, row in rp_data.iterrows():
            match_id = row['match_id']
            protag_id = row['team_id']
            antag_id = row['opp_team_id']
            protag_index = team_map[protag_id]
            antag_index = team_map[antag_id]
            ## grab ratings going into games
            protag_rating = ratings_vec[protag_index] 
            antag_rating = ratings_vec[antag_index]
            protag_neg_rating = neg_ratings_vec[protag_index] 
            antag_neg_rating = neg_ratings_vec[antag_index]
            protag_entr = ratings_entr[protag_index]
            antag_entr = ratings_entr[antag_index]
            protag_pace = pace_vec[protag_index]
            antag_pace = pace_vec[antag_index]
            protag_pentr = pace_entr[protag_index]
            antag_pentr = pace_entr[antag_index]
            history.append([match_id, protag_id, antag_id, protag_rating, antag_rating, protag_neg_rating,antag_neg_rating,protag_entr, antag_entr,
                            protag_pace, antag_pace, protag_pentr, antag_pentr])
            
            # update
            game_rating = row['rtg_avg']
            pace_rating = row['pace_avg']
            if np.isnan(game_rating):
                continue
            if np.isnan(pace_rating):
                continue
                
            if game_rating > 0:
                rating_update[protag_index][antag_index]+=np.abs(game_rating)
                neg_rating_update[antag_index][protag_index]+=np.abs(game_rating)
            else:
                rating_update[antag_index][protag_index]+=np.abs(game_rating)
                neg_rating_update[protag_index][antag_index]+=np.abs(game_rating)
            pace_update[protag_index][antag_index] += pace_rating
            pace_update[antag_index][protag_index] += pace_rating
            pace_std_update[protag_index][antag_index] += 1
            pace_std_update[antag_index][protag_index] += 1
            
        prank_mat+=rating_update
        neg_prank_mat+=neg_rating_update
        pace_mat += pace_update
        pace_std+= pace_std_update



    history = pd.DataFrame(history, columns=['match_id','id','opp_id','rating','opp_rating','neg_rating','neg_opp_rating','entropy','opp_entropy',
                                            'pace','opp_pace','pace_entropy','opp_pace_entropy'])
    history = history.merge(stf_schedule[['match_id','team_id','score','opp_score']].rename(columns={'team_id':'id'}),
                        how='left', on=['match_id','id'])

    history['score_diff'] = history['score'].copy()-history['opp_score'].copy()
    history['ratings_diff'] = history['rating'].copy()-history['opp_rating'].copy()
    history['ratings_v2'] = history['rating'].copy()-history['neg_rating'].copy()
    history['opp_ratings_v2'] = history['opp_rating'].copy()-history['neg_opp_rating'].copy()
    history['rtg_diff_v2'] = history['ratings_v2'].copy()-history['opp_ratings_v2'].copy()
    history['score_total'] = history['score'].copy()+history['opp_score'].copy()
    history['pace_comb'] = history['pace'].copy()+history['opp_pace'].copy()


    teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))


    def get_current_ratings():
        
        current_rtgs = []
        for protag_id,protag_index, in team_map.items():
            protag_index = team_map[protag_id]
            ## grab ratings going into games
            protag_rating = ratings_vec[protag_index] 
            protag_neg_rating = neg_ratings_vec[protag_index]
            protag_entr = ratings_entr[protag_index]
            protag_pace = pace_vec[protag_index]
            protag_pentr = pace_entr[protag_index]
            
            current_rtgs.append([protag_id, protag_rating, protag_neg_rating, protag_entr, protag_pace, protag_pentr])
        
        return pd.DataFrame(current_rtgs, columns=['team_id','rating','neg_rating','entropy','pace','pace_entropy'])
    # 
    to_save = history.copy().dropna(subset=['match_id']).reset_index(drop=True)
    to_save = to_save.drop(columns=['score','opp_score','score_diff','score_total'])
    to_save = to_save.rename(columns={'id':'team_id','opp_id':'opp_team_id'})
    to_save_opp = to_save.copy().reset_index(drop=True)
    to_save_opp.columns=['match_id','opp_team_id','team_id','opp_rating','rating','neg_opp_rating','neg_rating','opp_entropy','entropy','opp_pace','pace','opp_pace_entropy','pace_entropy','ratings_diff','opp_ratings_v2','ratings_v2','rtg_diff_v2','pace_comb']
    to_save_opp = to_save_opp[list(to_save)]
    to_save_opp['ratings_diff'] = to_save_opp['ratings_diff'].copy()*-1
    to_save_opp['rtg_diff_v2'] = to_save_opp['rtg_diff_v2'].copy()*-1
    to_save = pd.concat([to_save, to_save_opp], axis=0).reset_index(drop=True)
    to_save.sort_values(by=['match_id'])
    to_save = to_save.drop(columns=['opp_rating','rating','neg_opp_rating','neg_rating','ratings_diff'])
    current_ratings = get_current_ratings()
    opp_current_ratings = current_ratings.copy()
    opp_current_ratings.columns=['opp_team_id','opp_rating','neg_opp_rating','opp_entropy','opp_pace','opp_pace_entropy']
    upc = stf_schedule.copy().loc[stf_schedule['is_upcoming']==1].reset_index(drop=True)[['match_id','team_id','opp_team_id']]
    upc = upc.merge(current_ratings.copy(), how='left', on=['team_id'])
    upc = upc.merge(opp_current_ratings.copy(), how='left', on='opp_team_id')
    # upc['ratings_diff'] = upc['rating'].copy()-upc['opp_rating'].copy()
    upc['ratings_v2'] = upc['rating'].copy()-history['neg_rating'].copy()
    upc['opp_ratings_v2'] = upc['opp_rating'].copy()-history['neg_opp_rating'].copy()
    upc['rtg_diff_v2'] = upc['ratings_v2'].copy()-history['opp_ratings_v2'].copy()
    upc = upc.drop(columns=['opp_rating','rating','neg_opp_rating','neg_rating'])
    upc['pace_comb'] = upc['pace'].copy()+upc['opp_pace'].copy()
    to_save = pd.concat([to_save, upc], axis=0).reset_index(drop=True)

    to_save.to_csv(os.path.join(DROPBOX_PATH, 'team_ratings/grade_network.csv'), index=False)
    return

def gg3():

    # schedule, stf_schedule = load_schedules()
    # footy = load_footy()
    # teams_footy2id = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_footy2id'))
    # footy = prep_footy_data(footy.copy(), teams_footy2id)
    # df, missing = merge_game_ids(stf_schedule.copy(), footy.copy().rename(columns={'datetime_UTC':'footy_datetime_UTC'}), schedule)

    # df = processing_steps(df, schedule.copy())
    
    ## gg 3: prev matchup
    print("Creating team score/opp score game grades...")
    ### data prep ###
    teams = load_dict(os.path.join(DROPBOX_PATH, 'IDs/teams'))
    competitions = load_dict(os.path.join(DROPBOX_PATH, 'IDs/competitions'))
    
    schedule, stf_schedule = load_schedules()
    teams_id2footy = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_id2footy'))
    teams_footy2id = load_dict(os.path.join(DROPBOX_PATH, 'IDs/footy/teams_footy2id'))
    footy = load_footy()
    footy = prep_footy_data(footy.copy(), teams_footy2id)

    df, missing = merge_game_ids(stf_schedule.copy(), footy.copy().rename(columns={'datetime_UTC':'footy_datetime_UTC'}), schedule.copy())

    df = add_sb_metrics(df, stf_schedule)
    df = df.rename(columns={'id':'footy_match_id'})

    df = processing_steps(df, schedule.copy())
    sb_data = df.copy().loc[df['match_id'].notnull()].sort_values(by='datetime_UTC').reset_index(drop=True)
    non_sb_data = df.copy().loc[df['match_id'].isnull()].sort_values(by='datetime_UTC').reset_index(drop=True)
    
    def prep_sb_data(data):
        stat_decays = {
            'sb_team_goals':0.05,
            'sb_opp_goals':0.05,
            'sb_team_SOTs':0.06,
            'sb_opp_SOTs':0.06,
            'sb_team_obv':0.04,
            'sb_opp_obv':0.04

        }
        
        data = data.copy().dropna(subset=['sb_opp_shots','sb_team_xxG'])
        data['games_in_season'] = data.groupby(['team_id','season'])['match_id'].transform('count')
        data['szn_already_played'] = data.groupby(['team_id','season'])['match_id'].transform('cumcount')
        data['games_remaining'] = data['games_in_season'].copy()-data['szn_already_played'].copy()
        data = data.reset_index(drop=True)
        
        data = data.sort_values(by=['datetime_UTC']).reset_index(drop=True)
        for stat, decay in stat_decays.items():
            alpha=decay
            data = data.set_index('datetime_UTC')
            data['left_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().ewm(alpha=alpha).mean())
            # data['left_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().ewm(halflife=f'{halflife}D', times=pd.DatetimeIndex(x.index)).mean())
            # data['left_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().rolling(40).mean())
            data = data.reset_index()
            data = data.sort_values(by=['datetime_UTC'], ascending=False)
            data = data.set_index('datetime_UTC')
    #         data['right_league_HFA'] = data.groupby(['competition_id','is_home'])[stat].transform(lambda x: x.shift().ewm(alpha=alpha**2).mean())
            ## not sure why this gives worse results
            #data[f'adj_{stat}'] = data[stat].copy()-(data['right_league_HFA'].copy()-data['right_league_HFA'].mean())
            data['right_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().ewm(alpha=alpha).mean())
            #data['right_adj_target'] = data.groupby(['team_id'])[f'adj_{stat}'].transform(lambda x: x.shift().ewm(alpha=alpha).mean())

            data = data.reset_index()
            data = data.sort_values(by=['datetime_UTC']).reset_index(drop=True)
            data[f'{stat}_tgt'] = data[['left_target','right_target']].mean(axis=1)
            data[f'{stat}_fwd_tgt'] = data['right_target'].copy()
        
        return data

    sb_data = prep_sb_data(sb_data.copy())

    def prep_non_sb_data(data):
        stat_decays = {
            'sb_team_goals':0.05,
            'sb_opp_goals':0.05,
            'sb_team_SOTs':0.06,
            'sb_opp_SOTs':0.06
    #         'sb_team_obv':0.04,
    #         'sb_opp_obv':0.04

        }
        
        data = data.copy().dropna(subset=['team_goals','team_shotsOnTarget'])
        data['games_in_season'] = data.groupby(['team_id','season'])['footy_match_id'].transform('count')
        data['szn_already_played'] = data.groupby(['team_id','season'])['footy_match_id'].transform('cumcount')
        data['games_remaining'] = data['games_in_season'].copy()-data['szn_already_played'].copy()
        
        data = data.sort_values(by=['datetime_UTC']).reset_index(drop=True)
        for stat, decay in stat_decays.items():
            alpha = decay
    #         data = data.set_index('datetime_UTC')
            data['left_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().ewm(alpha=alpha).mean())
            # data['left_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().ewm(halflife=f'{halflife}D', times=pd.DatetimeIndex(x.index)).mean())
            # data['left_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().rolling(40).mean())
    #         data = data.reset_index()
            data = data.sort_values(by=['datetime_UTC'], ascending=False).reset_index(drop=True)
    #         data = data.set_index('datetime_UTC')
    #         data['right_league_HFA'] = data.groupby(['competition_id','is_home'])[stat].transform(lambda x: x.shift().ewm(alpha=alpha**2).mean())
            ## not sure why this gives worse results
            #data[f'adj_{stat}'] = data[stat].copy()-(data['right_league_HFA'].copy()-data['right_league_HFA'].mean())
            data['right_target'] = data.groupby(['team_id'])[stat].transform(lambda x: x.shift().ewm(alpha=alpha).mean())
            #data['right_adj_target'] = data.groupby(['team_id'])[f'adj_{stat}'].transform(lambda x: x.shift().ewm(alpha=alpha).mean())

            data = data.sort_values(by=['datetime_UTC']).reset_index(drop=True)
            data[f'{stat}_tgt'] = data[['left_target','right_target']].mean(axis=1)
            data[f'{stat}_fwd_tgt'] = data['right_target'].copy()
        
        # weird bug fix
        data['szn_already_played'] = data['games_in_season'].copy()-data['games_remaining'].copy()
        return data

    non_sb_data = prep_non_sb_data(non_sb_data.copy())


    stats = [
        'score_diff', 'xG_diff', 'shot_diff', 'sq_diff', 'sot_diff', 'score_total', 'xG_total', 'shot_total', 'sq_total', 'sot_total', 'sit_boost', 'pace', 'obv_pace', 'xxG_diff', 'pct_xxG_diff', 'obv_diff', 'pct_obv_diff', 'cgoal_skew', 'cgoal_kurt', 'cgoal_sum', 'cgoal_std', 'cconcede_skew', 'cconcede_kurt', 'cconcede_sum', 'cconcede_std', 'xxG_conversion', 'xG_conversion', 'opp_xG_conversion', 'opp_xxG_conversion', 'win_prob', 'draw_prob', 'opp_dt_eff', 'opp_mt_eff', 'opp_at_eff', 'opp_d_qual', 'opp_mid_qual', 'opp_atck_qual', 'team_dt_eff', 'team_mt_eff', 'team_at_eff', 'team_d_qual', 'team_mid_qual', 'team_atck_qual', 'opp_gk_pass_loc', 'opp_def_pass_loc', 'opp_mid_pass_loc', 'opp_atck_pass_loc', 'opp_gk_pass_angle', 'opp_def_pass_angle', 'opp_mid_pass_angle', 'opp_atck_pass_angle', 'opp_gk_pass_count', 'opp_def_pass_count', 'opp_mid_pass_count', 'opp_atck_pass_count', 'opp_gk_pass_dist', 'opp_def_pass_dist', 'opp_mid_pass_dist', 'opp_atck_pass_dist', 'opp_gk_pass_obv_vol', 'opp_def_pass_obv_vol', 'opp_mid_pass_obv_vol', 'opp_atck_pass_obv_vol', 'opp_gk_pass_obv_eff', 'opp_def_pass_obv_eff', 'opp_mid_pass_obv_eff', 'opp_atck_pass_obv_eff', 'team_gk_pass_loc', 'team_def_pass_loc', 'team_mid_pass_loc', 'team_atck_pass_loc', 'team_gk_pass_angle', 'team_def_pass_angle', 'team_mid_pass_angle', 'team_atck_pass_angle', 'team_gk_pass_count', 'team_def_pass_count', 'team_mid_pass_count', 'team_atck_pass_count', 'team_gk_pass_dist', 'team_def_pass_dist', 'team_mid_pass_dist', 'team_atck_pass_dist', 'team_gk_pass_obv_vol', 'team_def_pass_obv_vol', 'team_mid_pass_obv_vol', 'team_atck_pass_obv_vol', 'team_gk_pass_obv_eff', 'team_def_pass_obv_eff', 'team_mid_pass_obv_eff', 'team_atck_pass_obv_eff', 'opp_gk_carry_angle', 'opp_def_carry_angle', 'opp_mid_carry_angle', 'opp_atck_carry_angle', 'opp_gk_carry_count', 'opp_def_carry_count', 'opp_mid_carry_count', 'opp_atck_carry_count', 'opp_gk_carry_dist', 'opp_def_carry_dist', 'opp_mid_carry_dist', 'opp_atck_carry_dist', 'opp_gk_carry_vol', 'opp_def_carry_vol', 'opp_mid_carry_vol', 'opp_atck_carry_vol', 'opp_gk_carry_eff', 'opp_def_carry_eff', 'opp_mid_carry_eff', 'opp_atck_carry_eff', 'team_gk_carry_angle', 'team_def_carry_angle', 'team_mid_carry_angle', 'team_atck_carry_angle', 'team_gk_carry_dist', 'team_def_carry_dist', 'team_mid_carry_dist', 'team_atck_carry_dist', 'team_gk_carry_count', 'team_def_carry_count', 'team_mid_carry_count', 'team_atck_carry_count', 'team_gk_carry_vol', 'team_def_carry_vol', 'team_mid_carry_vol', 'team_atck_carry_vol', 'team_gk_carry_eff', 'team_def_carry_eff', 'team_mid_carry_eff', 'team_atck_carry_eff', 'opp_gk_dfn_pos', 'opp_def_dfn_pos', 'opp_mid_dfn_pos', 'opp_atck_dfn_pos', 'opp_gk_dfn_obv_vol', 'opp_def_dfn_obv_vol', 'opp_mid_dfn_obv_vol', 'opp_atck_dfn_obv_vol', 'opp_gk_dfn_obv_eff', 'opp_def_dfn_obv_eff', 'opp_mid_dfn_obv_eff', 'opp_atck_dfn_obv_eff', 'opp_gk_dfn_xxG_vol', 'opp_def_dfn_xxG_vol', 'opp_mid_dfn_xxG_vol', 'opp_atck_dfn_xxG_vol', 'team_gk_dfn_pos', 'team_def_dfn_pos', 'team_mid_dfn_pos', 'team_atck_dfn_pos', 'team_gk_dfn_obv_vol', 'team_def_dfn_obv_vol', 'team_mid_dfn_obv_vol', 'team_atck_dfn_obv_vol', 'team_gk_dfn_obv_eff', 'team_def_dfn_obv_eff', 'team_mid_dfn_obv_eff', 'team_atck_dfn_obv_eff', 'team_gk_dfn_xxG_vol', 'team_def_dfn_xxG_vol', 'team_mid_dfn_xxG_vol', 'team_atck_dfn_xxG_vol', 'team_weighted_fouls', 'opp_weighted_fouls', 'opp_corners', 'team_corners', 'team_crosses', 'opp_crosses', 'team_cross_pct', 'opp_cross_pct', 'team_pens', 'opp_pens', 'carry_lengths', 'gk_dist', 'fields_gained', 'fields_gained_comp', 'med_def_action', 'threat_pp', 'dthreat_pp', 'off_embed_0', 'off_embed_1', 'off_embed_2', 'off_embed_3', 'off_embed_4', 'off_embed_5', 'def_embed_0', 'def_embed_1', 'def_embed_2', 'def_embed_3', 'def_embed_4', 'def_embed_5', 'opp_poss', 'opp_ppp', 'opp_spp', 'team_poss', 'team_ppp', 'team_spp', 'team_poss_start', 'opp_poss_start', 'team_poss_len', 'opp_poss_len', 'team_poss_width', 'opp_poss_width', 'team_poss_time_sum', 'opp_poss_time_sum', 'team_poss_time_median', 'opp_poss_time_median', 'oxG_f3', 'txG_f3', 'pct_lead', 'pct_tied', 'pct_trail', 'man_adv', 'team_dx_sec', 'team_xxG_sec', 'opp_dx_sec', 'opp_xxG_sec', 'opp_d3_passes', 'opp_d3_comp%', 'opp_d3_fcomp%', 'opp_m3_passes', 'opp_m3_comp%', 'opp_m3_fcomp%', 'opp_a3_passes', 'opp_a3_comp%', 'opp_a3_fcomp%', 'team_d3_passes', 'team_d3_comp%', 'team_d3_fcomp%', 'team_m3_passes', 'team_m3_comp%', 'team_m3_fcomp%', 'team_a3_passes', 'team_a3_comp%', 'team_a3_fcomp%', 'opp_SOT%', 'team_save%', 'team_SOT%', 'opp_save%', 'team_XGOT/SOT', 'opp_XGOT/SOT', 'opp_end_d3', 'opp_end_m3', 'opp_end_a3', 'team_end_d3', 'team_end_m3', 'team_end_a3', 'team_switches', 'opp_switches', 'opp_d3_press', 'opp_m3_press', 'opp_a3_press', 'team_d3_press', 'team_m3_press', 'team_a3_press', 'opp_wide_poss', 'team_wide_poss',
    ]
    stats+=[
    'team_goals', 'opp_goals', 'attacks_recorded', 'team_yellow_cards', 'opp_yellow_cards', 'team_red_cards', 'opp_red_cards', 'team_shotsOnTarget', 'opp_shotsOnTarget', 'team_shotsOffTarget', 'opp_shotsOffTarget', 'team_shots', 'opp_shots', 'team_fouls', 'opp_fouls', 'team_possession', 'opp_possession', 'team_offsides', 'opp_offsides', 'team_dangerous_attacks', 'opp_dangerous_attacks', 'team_attacks', 'opp_attacks', 'team_xg', 'opp_xg', 'total_xg', 'team_penalties_won', 'opp_penalties_won', 'team_penalty_goals', 'opp_penalty_goals', 'team_penalty_missed', 'opp_penalty_missed', 'team_throwins', 'opp_throwins', 'team_freekicks', 'opp_freekicks', 'team_goalkicks', 'opp_goalkicks'
    ]
    stats+=[
        'goals_total', 'shots_total', 'shotsOnTarget_total', 'attacks_total', 'dangerous_attacks_total', 'xg_total'
    ]

    stats+=[
        'avg_potential','goals_diff', 'shots_diff', 'shotsOnTarget_diff', 'possession_diff', 'attacks_diff', 'dangerous_attacks_diff', 'xg_diff','no_home_away','is_home'
    ]

    # leaky for what happened in past
    ## but fixed this notebook
    stats+=[
        'odds_ft_1', 'odds_ft_x'
    ]

    # found to not be useful # these were useful for pace
    # 171           team_pens    0.000000
    # 172            opp_pens    0.000000
    # 283  team_penalties_won    0.000000
    for stat in ['opp_penalty_goals','team_penalty_goals','team_penalties_won','opp_penalties_won','opp_penalty_missed','attacks_recorded','goals_total']:
        stats.remove(stat)
        
    total_non_sb = len(non_sb_data)
    # drop cols where 93% nulls
    threshold = int(0.93*total_non_sb)
    footy_stats = stats.copy()
    to_drop = list(non_sb_data[stats].isnull().sum()[non_sb_data[stats].isnull().sum()>threshold].index)
    for td in to_drop:
        footy_stats.remove(td)
    non_sb_data = non_sb_data.drop(columns=to_drop)
    non_sb_data = non_sb_data.drop(columns=['team_id','opp_team_id','match_id','competition_id','is_upcoming']+['left_target','right_target'])

    def create_game_grades_non_sb(data, thresh=5):
        
        ## training data is a subset of all data, prefer mid season, because then the targets are stable
        train = data.copy().loc[data['games_remaining']>=thresh] ## will train right first
        train = train.copy().loc[train['szn_already_played']>=(thresh-3)].reset_index(drop=True) ## smaller here because it doesn't matter for one of the models
        
        print(f'{(len(train)/len(data))*100:.2f}% of data used for training')
        
        match_ids_used = list(train.copy().footy_match_id)
        remain = data.copy().loc[(data['games_remaining']<thresh)|(data['szn_already_played']<(thresh-3))].reset_index(drop=True)
        print(f'{(len(remain)/len(data))*100:.2f}% of data will be predicted OOS (bc didnt meet thresholds)')
        
        X = train.copy().sample(frac=1, random_state=17)
        
        non_sb_stats = ['sb_team_goals','sb_opp_goals','sb_team_SOTs','sb_opp_SOTs']        
        feat_importances = []
        feat_importances_fwd = []
        scores = {}
        scores_fwd = {}
        models = {}
        models_fwd = {}
        
        ## use kfold on historical data to train models
        hist_game_grades = None
        for stat in tqdm(non_sb_stats):
            fwd_tgt = stat+'_fwd_tgt'
            tgt = stat + '_tgt'
            stat_models = []
            stat_fwd_models = []
            stat_display = footy_stats.copy()
            stat_scores = []
            stat_fwd_scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=17)
            match_ids = []
            team_ids = []
            fwd_preds = []
            preds = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = X_train[tgt], X_test[tgt]
                y_train_fwd, y_test_fwd = X_train[fwd_tgt], X_test[fwd_tgt]
                X_train = X_train.copy()[footy_stats]
                match_ids.extend(X_test.footy_match_id)
                team_ids.extend(X_test.footy_team_id)
                X_test = X_test.copy()[footy_stats]
                fwd_model = catboost.CatBoostRegressor(verbose=False)
                fwd_model.fit(X_train, y_train_fwd)
                feat_importances_fwd.append(fwd_model.feature_importances_)
                stat_fwd_models.append(fwd_model)
                fwd_pred = fwd_model.predict(X_test)
                fwd_score = mean_squared_error(y_test_fwd, fwd_pred)
                fwd_preds.extend(fwd_pred)
                stat_fwd_scores.append(fwd_score)
                X_train = X_train.drop(columns=['odds_ft_x','odds_ft_1','avg_potential'])
                model = catboost.CatBoostRegressor(verbose=False)
                model.fit(X_train, y_train)
                feat_importances.append(model.feature_importances_)
                stat_models.append(model)
                pred = model.predict(X_test.drop(columns=['odds_ft_x','odds_ft_1','avg_potential']))
                score = mean_squared_error(y_test, pred)
                preds.extend(pred)
                stat_scores.append(score)

            scores[stat] = np.mean(stat_scores)
            scores_fwd[stat] = np.mean(stat_fwd_scores)
            models[stat] = stat_models
            models_fwd[stat] = stat_fwd_models
            stat_df = pd.DataFrame({
                'footy_match_id':match_ids,
                'footy_team_id':team_ids,
                f'{stat}_gg':preds,
                f'{stat}_fwd_gg':fwd_preds
            })
            
            ## record historical predictions
            if hist_game_grades is None:
                hist_game_grades = stat_df.copy()
            else:
                hist_game_grades = hist_game_grades.merge(stat_df, how='left', on=['footy_match_id','footy_team_id'])
        
        feat_importances_fwd = np.mean(np.array(feat_importances_fwd), axis=0)
        feat_importances_fwd = pd.DataFrame({
            'stats':stat_display,
            'importance': feat_importances_fwd
        }).sort_values(by='importance',ascending=False)

        feat_importances = np.mean(np.array(feat_importances), axis=0)
        stat_display.remove('odds_ft_x')
        stat_display.remove('odds_ft_1')
        stat_display.remove('avg_potential')
        feat_importances = pd.DataFrame({
            'stats':stat_display,
            'importance': feat_importances
        }).sort_values(by='importance',ascending=False)
        
        ## predict for non train games
        remaining_gg = pd.DataFrame({
            'footy_match_id':remain.footy_match_id,
            'footy_team_id':remain.footy_team_id
        })
        
        for stat in tqdm(non_sb_stats):
            fwd_stat_models = models_fwd[stat]
            stat_models = models[stat]
            fwd_cols = footy_stats.copy()
            reg_cols = fwd_cols.copy()
            reg_cols.remove('odds_ft_x')
            reg_cols.remove('odds_ft_1')
            reg_cols.remove('avg_potential')
            sm_preds =[]
            fsm_preds = []
            for sm in stat_models:
                sm_pred = sm.predict(remain[reg_cols])
                sm_preds.append(sm_pred)
            for fsm in fwd_stat_models:
                fsm_pred = fsm.predict(remain[fwd_cols])
                fsm_preds.append(fsm_pred)
            sm_preds = np.mean(np.array(sm_preds), axis=0)
            fsm_preds = np.mean(np.array(fsm_preds), axis=0)
            remaining_gg[f'{stat}_gg'] = sm_preds
            remaining_gg[f'{stat}_fwd_gg'] = fsm_preds
            
        ## commented out is just a check
    #     print(len(hist_game_grades))
    #     print(hist_game_grades[[f'{stat}_gg',f'{stat}_fwd_gg']].isnull().sum())
        hist_game_grades = pd.concat([hist_game_grades, remaining_gg], axis=0).reset_index(drop=True)
    #     print(len(hist_game_grades))
    #     print(hist_game_grades[[f'{stat}_gg',f'{stat}_fwd_gg']].isnull().sum())
        
        for stat_name, stat_models in models_fwd.items():
            for i, sm in enumerate(stat_models):
                model_path = os.path.join(DROPBOX_PATH, f'models/game_grades/fwd_non_{stat_name}_{i}')
                sm.save_model(model_path)
        for stat_name, _stat_models in models.items():
            for i, sm in enumerate(_stat_models):
                model_path = os.path.join(DROPBOX_PATH, f'models/game_grades/non_{stat_name}_{i}')
                sm.save_model(model_path)
            
            
        
        return hist_game_grades

    non_sb_grades = create_game_grades_non_sb(non_sb_data.copy())



    def create_game_grades_sb(data, thresh=5):
        
        ## training data is a subset of all data, prefer mid season, because then the targets are stable
        train = data.copy().loc[data['games_remaining']>=thresh] ## will train right first
        train = train.copy().loc[train['szn_already_played']>=(thresh-3)].reset_index(drop=True) ## smaller here because it doesn't matter for one of the models
        
        print(f'{(len(train)/len(data))*100:.2f}% of data used for training')
        
        
        remain = data.copy().loc[(data['games_remaining']<thresh)|(data['szn_already_played']<(thresh-3))].reset_index(drop=True)
        print(f'{(len(remain)/len(data))*100:.2f}% of data will be predicted OOS')
        
        X = train.copy().sample(frac=1, random_state=17)
        
        sb_stats = ['sb_team_goals','sb_opp_goals','sb_team_SOTs','sb_opp_SOTs','sb_team_obv','sb_opp_obv']        
        feat_importances = []
        feat_importances_fwd = []
        scores = {}
        scores_fwd = {}
        models = {}
        models_fwd = {}
        
        fwd_cols = stats.copy()
        reg_cols = fwd_cols.copy()
        reg_cols.remove('odds_ft_x')
        reg_cols.remove('odds_ft_1')
        reg_cols.remove('avg_potential')

        ## use kfold on historical data to train models
        hist_game_grades = None
        for stat in tqdm(sb_stats):
            fwd_tgt = stat+'_fwd_tgt'
            tgt = stat + '_tgt'
            stat_models = []
            stat_fwd_models = []
            stat_scores = []
            stat_fwd_scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=17)
            match_ids = []
            team_ids = []
            fwd_preds = []
            preds = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = X_train[tgt], X_test[tgt]
                y_train_fwd, y_test_fwd = X_train[fwd_tgt], X_test[fwd_tgt]
                X_train = X_train.copy()[fwd_cols]
                match_ids.extend(X_test.match_id)
                team_ids.extend(X_test.team_id)
                X_test = X_test.copy()[fwd_cols]
                fwd_model = catboost.CatBoostRegressor(verbose=False)
                fwd_model.fit(X_train, y_train_fwd)
                feat_importances_fwd.append(fwd_model.feature_importances_)
                stat_fwd_models.append(fwd_model)
                fwd_pred = fwd_model.predict(X_test)
                fwd_score = mean_squared_error(y_test_fwd, fwd_pred)
                fwd_preds.extend(fwd_pred)
                stat_fwd_scores.append(fwd_score)
                X_train = X_train[reg_cols]
                model = catboost.CatBoostRegressor(verbose=False)
                model.fit(X_train, y_train)
                feat_importances.append(model.feature_importances_)
                stat_models.append(model)
                pred = model.predict(X_test[reg_cols])
                score = mean_squared_error(y_test, pred)
                preds.extend(pred)
                stat_scores.append(score)

            scores[stat] = np.mean(stat_scores)
            scores_fwd[stat] = np.mean(stat_fwd_scores)
            models[stat] = stat_models
            models_fwd[stat] = stat_fwd_models
            stat_df = pd.DataFrame({
                'match_id':match_ids,
                'team_id':team_ids,
                f'{stat}_gg':preds,
                f'{stat}_fwd_gg':fwd_preds
            })
            
            ## record historical predictions
            if hist_game_grades is None:
                hist_game_grades = stat_df.copy()
            else:
                hist_game_grades = hist_game_grades.merge(stat_df, how='left', on=['match_id','team_id'])
                
        print(scores_fwd)
        print(scores)
        
        feat_importances_fwd = np.mean(np.array(feat_importances_fwd), axis=0)
        feat_importances_fwd = pd.DataFrame({
            'stats':fwd_cols,
            'importance': feat_importances_fwd
        }).sort_values(by='importance',ascending=False)

        feat_importances = np.mean(np.array(feat_importances), axis=0)
        feat_importances = pd.DataFrame({
            'stats':reg_cols,
            'importance': feat_importances
        }).sort_values(by='importance',ascending=False)
        
        ## predict for non train games
        remaining_gg = pd.DataFrame({
            'match_id':remain.match_id,
            'team_id':remain.team_id,
            'opp_team_id':remain.opp_team_id
        })
        
        for stat in tqdm(sb_stats):
            fwd_stat_models = models_fwd[stat]
            stat_models = models[stat]

            sm_preds =[]
            fsm_preds = []
            for sm in stat_models:
                sm_pred = sm.predict(remain[reg_cols])
                sm_preds.append(sm_pred)
            for fsm in fwd_stat_models:
                fsm_pred = fsm.predict(remain[fwd_cols])
                fsm_preds.append(fsm_pred)
            sm_preds = np.mean(np.array(sm_preds), axis=0)
            fsm_preds = np.mean(np.array(fsm_preds), axis=0)
            remaining_gg[f'{stat}_gg'] = sm_preds
            remaining_gg[f'{stat}_fwd_gg'] = fsm_preds
            
        for stat_name, stat_models in models_fwd.items():
            for i, sm in enumerate(stat_models):
                model_path = os.path.join(DROPBOX_PATH, f'models/game_grades/fwd_{stat_name}_{i}')
                sm.save_model(model_path)
        for stat_name, _stat_models in models.items():
            for i, sm in enumerate(_stat_models):
                model_path = os.path.join(DROPBOX_PATH, f'models/game_grades/{stat_name}_{i}')
                sm.save_model(model_path)
            
        ## commented out is just a check
        print(len(hist_game_grades))
        print(hist_game_grades[[f'{stat}_gg',f'{stat}_fwd_gg']].isnull().sum())
        hist_game_grades = pd.concat([hist_game_grades, remaining_gg], axis=0).reset_index(drop=True)
        print(len(hist_game_grades))
        print(hist_game_grades[[f'{stat}_gg',f'{stat}_fwd_gg']].isnull().sum())
        
        return hist_game_grades, feat_importances_fwd, feat_importances


    sb_game_grades, fi_fwd, fi_ = create_game_grades_sb(sb_data.copy())

    

    master = df.copy()[['datetime_UTC','competition_id','season','footy_match_id','footy_comp_name','footy_team_id','footy_opp_id','match_date_UTC','match_id','team_id','opp_team_id']]


    non_sb_grades.columns=['footy_match_id','footy_team_id']+['non_'+stat for stat in list(non_sb_grades)[2:]]

    if 'opp_team_id' in list(non_sb_grades):
        non_sb_grades = non_sb_grades.drop(columns=['opp_team_id'])
    if 'opp_team_id' in list(sb_game_grades):
        sb_game_grades = sb_game_grades.drop(columns=['opp_team_id'])
    master = master.merge(non_sb_grades, how='left', on=['footy_match_id','footy_team_id'])
    master = master.merge(sb_game_grades, how='left', on=['match_id','team_id'])
    master.drop_duplicates(subset=['footy_match_id','footy_team_id'])

    
    for col_name in ['non_sb_team_goals_gg', 'non_sb_team_goals_fwd_gg', 'non_sb_opp_goals_gg', 'non_sb_opp_goals_fwd_gg', 'non_sb_team_SOTs_gg', 'non_sb_team_SOTs_fwd_gg', 'non_sb_opp_SOTs_gg', 'non_sb_opp_SOTs_fwd_gg']:
        sb_col_name = copy(col_name).replace('non_', '')
        master[sb_col_name] = master[sb_col_name].fillna(master[col_name])
        master = master.drop(columns=[col_name])

    master.to_csv(os.path.join(DROPBOX_PATH, 'Statsbomb/game_grades/game_grades.csv'),index=False)

    gg = master.copy()
    del master
    import gc
    gc.collect()

    for gg_stat in ['sb_team_goals','sb_opp_goals','sb_team_SOTs','sb_opp_SOTs','sb_team_obv','sb_opp_obv']:
        stat_name = gg_stat.replace('sb_','')
        gg[stat_name] = 0.7*gg[gg_stat+'_fwd_gg'].copy() + 0.3*gg[gg_stat+'_gg'].copy()
        gg = gg.drop(columns=[gg_stat+'_fwd_gg',gg_stat+'_gg'])
    ## concat upcoming to help upcoming model
    gg = pd.concat([gg, stf_schedule.loc[stf_schedule['is_upcoming']==1][['datetime_UTC','competition_id','match_date_UTC','match_id','team_id','opp_team_id']]
    ], axis=0).reset_index(drop=True)

    
    gg[['prev_tgoals','prev_ogoals','prev_tSOTs','prev_oSOTs','prev_tobv','prev_oobv']] = gg.groupby(['team_id','opp_team_id'])[['team_goals','opp_goals','team_SOTs','opp_SOTs','team_obv','opp_obv']].transform(lambda x: x.shift().ewm(alpha=0.3).mean())
    model_ready = gg[['match_id','team_id']+['prev_tgoals','prev_ogoals','prev_tSOTs','prev_oSOTs','prev_tobv','prev_oobv']].dropna(subset=['match_id']).reset_index(drop=True)
    model_ready['match_id'] = model_ready['match_id'].astype(int)
    model_ready['team_id'] = model_ready['team_id'].astype(int)

    model_ready.to_csv(os.path.join(DROPBOX_PATH, 'model_ready/prev_matchup_data.csv'), index=False)
    return

def game_grades():
    jobs = []
    
    p1 = Process(target=gg1)
    jobs.append(p1)
    p1.start()
    p2 = Process(target=gg2)
    jobs.append(p2)
    p2.start()
    p3 = Process(target=gg3)
    jobs.append(p3)
    p3.start()
    # p4 = Process(target=update_game_state_vecs)
    # jobs.append(p4)
    # p4.start()
    # p5 = Process(target=download_odds)
    # jobs.append(p5)
    # p5.start()

    # checks to see if they are finished
    for job in jobs:
        job.join()
        time.sleep(1)
    return


if __name__ == '__main__':
    game_grades()
