

import os

import numpy as np
import pandas as pd

from gensim.models import FastText

from dotenv import load_dotenv

load_dotenv()

DROPBOX_PATH = os.environ.get('DROPBOX_PATH')


def preprocessing(df):
    
    # attack, midfield, defense, goalie
    df['unit'] = np.where(df['position_id']==1, 0, 1)
    df['unit'] = np.where(((df['position_id']>1)&(df['position_id']<=8)), 1, df['unit'].copy())
    df['unit'] = np.where(df['position_id'].copy().isin([9,10,11,12,13,14,15,16,18,19,20]), 2, df['unit'].copy())
    df['unit'] = np.where(df['position_id'].copy().isin([17,21,25,23,22,24]), 3, df['unit'].copy())

    df['third'] = np.where(df['x']<=40, 1, 0)
    df['third'] = np.where(((df['x']>40)&(df['x']<=80)), 2, df['third'].copy())
    df['third'] = np.where(df['x']>80, 3, df['third'].copy())

    # used for pace
    df['obv_total'] = df['obv_for_net'].copy()+df['obv_against_net'].copy()
    
    return df




def threat(df):
    
    stats, cols = [], []
    
    # game
    df['is_goal'] = df['team_goal'].copy()+df['opp_goal'].copy()
    df['is_shot'] = np.where(df['type_id']==16, 1, 0)
    df['is_SOT'] = np.where((df['type_id']==16)&(df['outcome_id'].isin([97,100,116])), 1, 0)

    # non pen
    non_pen = df.loc[df['shot_statsbomb_xg']!=0.76].groupby(['action_team']).agg({
        'shot_statsbomb_xg':['mean','sum'],
        'is_goal':['sum'],
        'is_shot':['sum'],
        'is_SOT':['sum']
    })

    # goals are backward for action team col
    t_shot_qual, t_shot_threat, o_goals, t_shots, t_sots = non_pen.loc[1].values
    o_shot_qual, o_shot_threat, t_goals, o_shots, o_sots = non_pen.loc[0].values

    # pace
    score_total = t_goals+o_goals
    xG_total = t_shot_threat+o_shot_threat
    shot_total = t_shots+o_shots
    sq_total = t_shot_qual+o_shot_qual
    sot_total = t_sots+o_sots
    
    # skill diff
    score_diff = (t_goals-o_goals)/(score_total+1)
    xG_diff = (t_shot_threat-o_shot_threat)/(xG_total+1)
    shot_diff = (t_shots-o_shots)/(shot_total+1)
    sq_diff = t_shot_qual-o_shot_qual
    sot_diff = (t_sots-o_sots)/(sot_total+1)
    
    stats.extend([score_diff, xG_diff, shot_diff, sq_diff, sot_diff, score_total, xG_total,shot_total,sq_total, sot_total])
    cols.extend(['score_diff', 'xG_diff', 'shot_diff', 'sq_diff', 'sot_diff', 'score_total', 'xG_total','shot_total','sq_total', 'sot_total'])
    
    ## xxG ##
    
    sit_boost = df.sit_boost.sum()
    pace = df.xxG_diff.abs().sum()
    obv_pace = df.obv_total_net.abs().sum()
    xxG_diff = df.xxG_diff.sum()
    obv_diff = df.obv_total_net.sum()
    pct_xxG_diff = xxG_diff/pace
    pct_obv_diff = df.obv_total_net.sum()/obv_pace
    cgoal_skew = df.cg.skew()
    cgoal_kurt = df.cg.kurt()
    cgoal_sum = df.cg.sum()
    cgoal_std = df.cg.std()
    cconcede_skew = df.cc.skew()
    cconcede_kurt = df.cc.kurt()
    cconcede_sum = df.cc.sum()
    cconcede_std = df.cc.std()
    
    xxG_conversion = t_shot_threat/cgoal_sum
    xG_conversion = t_goals/t_shot_threat
    opp_xxG_conversion = o_shot_threat/cconcede_sum
    opp_xG_conversion = o_goals/o_shot_threat
    
    stats.extend([
        sit_boost,pace,obv_pace,xxG_diff,pct_xxG_diff,obv_diff,pct_obv_diff,cgoal_skew,cgoal_kurt,cgoal_sum,cgoal_std,
        cconcede_skew,cconcede_kurt,cconcede_sum,cconcede_std,xxG_conversion,xG_conversion,
        opp_xG_conversion,opp_xxG_conversion
    ])
    cols.extend([
        'sit_boost','pace','obv_pace','xxG_diff','pct_xxG_diff','obv_diff','pct_obv_diff','cgoal_skew','cgoal_kurt','cgoal_sum','cgoal_std',
        'cconcede_skew','cconcede_kurt','cconcede_sum','cconcede_std','xxG_conversion','xG_conversion',
        'opp_xG_conversion','opp_xxG_conversion'
    ])
    
    
    ## win prob ##
    wp = df.prob_win.mean()
    dp = df.prob_draw.mean()
    stats.extend([wp, dp])
    cols.extend(['win_prob','draw_prob'])
    
    ## OBV ##
    
    # third_of_field quality
    third_q = df.groupby(['action_team','third']).agg({
        'obv_total_net':['mean','sum']
    }).reset_index()
    third_q.columns=['action_team','third','efficiency','volume']
    third_q =third_q.pivot(index='action_team', columns='third', values=['efficiency','volume'])
    
    stats.extend(list(third_q.loc[0].values))
    stats.extend(list(third_q.loc[1].values))
    cols.extend([
        'opp_dt_eff', 'opp_mt_eff', 'opp_at_eff', 'opp_d_qual', 'opp_mid_qual', 'opp_atck_qual',
        'team_dt_eff', 'team_mt_eff', 'team_at_eff', 'team_d_qual', 'team_mid_qual', 'team_atck_qual'
    ])
    
    # pass
    passes = df.copy().loc[df['type_id']==30].reset_index(drop=True)
    pass_stats = passes.groupby(['action_team','unit']).agg({
        'x':['mean'],
        'angle':['median', 'count'],
        'distance':['median'],
        'obv_total_net':['sum','mean']
    }).reset_index()

    pass_stats.columns=['action_team','unit','pass_loc','pass_angle','pass_count','pass_distance','pass_obv_volume','pass_efficiency']
    pass_stats = pass_stats.pivot(index='action_team', columns='unit', values=['pass_loc','pass_angle','pass_count','pass_distance','pass_obv_volume','pass_efficiency'])
    
    stats.extend(list(pass_stats.loc[0].values))
    stats.extend(list(pass_stats.loc[1].values))
    pass_cols = [
        'opp_gk_pass_loc', 'opp_def_pass_loc', 'opp_mid_pass_loc', 'opp_atck_pass_loc', 
        'opp_gk_pass_angle', 'opp_def_pass_angle', 'opp_mid_pass_angle', 'opp_atck_pass_angle', 
        'opp_gk_pass_count', 'opp_def_pass_count', 'opp_mid_pass_count', 'opp_atck_pass_count',
        'opp_gk_pass_dist', 'opp_def_pass_dist', 'opp_mid_pass_dist', 'opp_atck_pass_dist',
        'opp_gk_pass_obv_vol', 'opp_def_pass_obv_vol', 'opp_mid_pass_obv_vol', 'opp_atck_pass_obv_vol',
        'opp_gk_pass_obv_eff', 'opp_def_pass_obv_eff', 'opp_mid_pass_obv_eff', 'opp_atck_pass_obv_eff',
        'team_gk_pass_loc', 'team_def_pass_loc', 'team_mid_pass_loc', 'team_atck_pass_loc', 
        'team_gk_pass_angle', 'team_def_pass_angle', 'team_mid_pass_angle', 'team_atck_pass_angle', 
        'team_gk_pass_count', 'team_def_pass_count', 'team_mid_pass_count', 'team_atck_pass_count',
        'team_gk_pass_dist', 'team_def_pass_dist', 'team_mid_pass_dist', 'team_atck_pass_dist',
        'team_gk_pass_obv_vol', 'team_def_pass_obv_vol', 'team_mid_pass_obv_vol', 'team_atck_pass_obv_vol',
        'team_gk_pass_obv_eff', 'team_def_pass_obv_eff', 'team_mid_pass_obv_eff', 'team_atck_pass_obv_eff'
    ] 
    cols.extend(pass_cols)
#     print("passes", len(list(pass_stats.loc[0].values))*2, len(pass_cols))
    
    carries = df.copy().loc[df['type_id'].isin([14,43])].reset_index(drop=True) # and dribbles
    carry_stats = carries.groupby(['action_team','unit']).agg({
        'angle':['mean', 'count'],
        'distance':['mean'],
        'obv_total_net':['sum','mean']
    }).reset_index()

    carry_stats.columns=['action_team','unit','carry_angle','carry_count','carry_dist','carry_obv_volume','carry_efficiency']
    carry_stats = carry_stats.pivot(index='action_team', columns='unit', values=['carry_angle','carry_count','carry_dist','carry_obv_volume','carry_efficiency'])
    
    stats.extend(list(carry_stats.loc[0].values))
    stats.extend(list(carry_stats.loc[1].values))
    carry_cols = [
        'opp_gk_carry_angle', 'opp_def_carry_angle', 'opp_mid_carry_angle', 'opp_atck_carry_angle', 
        'opp_gk_carry_count', 'opp_def_carry_count', 'opp_mid_carry_count', 'opp_atck_carry_count',
        'opp_gk_carry_dist', 'opp_def_carry_dist', 'opp_mid_carry_dist', 'opp_atck_carry_dist',
        'opp_gk_carry_vol', 'opp_def_carry_vol', 'opp_mid_carry_vol', 'opp_atck_carry_vol', 
        'opp_gk_carry_eff', 'opp_def_carry_eff', 'opp_mid_carry_eff', 'opp_atck_carry_eff',
        'team_gk_carry_angle', 'team_def_carry_angle', 'team_mid_carry_angle', 'team_atck_carry_angle', 
        'team_gk_carry_dist', 'team_def_carry_dist', 'team_mid_carry_dist', 'team_atck_carry_dist',
        'team_gk_carry_count', 'team_def_carry_count', 'team_mid_carry_count', 'team_atck_carry_count',
        'team_gk_carry_vol', 'team_def_carry_vol', 'team_mid_carry_vol', 'team_atck_carry_vol', 
        'team_gk_carry_eff', 'team_def_carry_eff', 'team_mid_carry_eff', 'team_atck_carry_eff'
    ]
    cols.extend(carry_cols)
#     print("carry", len(list(carry_stats.loc[0].values))*2, len(carry_cols))
    
    ## defense ##
    dfn = df.copy().loc[df['type_id'].isin([3,4,6,9,10,17,21,22,23,39])].reset_index(drop=True)
    dfn_stats = dfn.groupby(['action_team','unit']).agg({
        'x':['mean'],
        'obv_total_net':['sum','mean'],
        'xxG_diff':['sum'] # also doing xxG because OBV not on all types of defensive actions
    }).reset_index()

    dfn_stats.columns=['action_team','unit','dfn_pos','dfn_vol','dfn_eff','dfn_xxG_vol']
    dfn_stats = dfn_stats.pivot(index='action_team', columns='unit', values=['dfn_pos','dfn_vol','dfn_eff','dfn_xxG_vol'])

    stats.extend(list(dfn_stats.loc[0].values))
    stats.extend(list(dfn_stats.loc[1].values))
    dfn_cols = [
        'opp_gk_dfn_pos', 'opp_def_dfn_pos', 'opp_mid_dfn_pos', 'opp_atck_dfn_pos',
        'opp_gk_dfn_obv_vol', 'opp_def_dfn_obv_vol', 'opp_mid_dfn_obv_vol', 'opp_atck_dfn_obv_vol',
        'opp_gk_dfn_obv_eff', 'opp_def_dfn_obv_eff', 'opp_mid_dfn_obv_eff', 'opp_atck_dfn_obv_eff',
        'opp_gk_dfn_xxG_vol', 'opp_def_dfn_xxG_vol', 'opp_mid_dfn_xxG_vol', 'opp_atck_dfn_xxG_vol',
        'team_gk_dfn_pos', 'team_def_dfn_pos', 'team_mid_dfn_pos', 'team_atck_dfn_pos',
        'team_gk_dfn_obv_vol', 'team_def_dfn_obv_vol', 'team_mid_dfn_obv_vol', 'team_atck_dfn_obv_vol',
        'team_gk_dfn_obv_eff', 'team_def_dfn_obv_eff', 'team_mid_dfn_obv_eff', 'team_atck_dfn_obv_eff',
        'team_gk_dfn_xxG_vol', 'team_def_dfn_xxG_vol', 'team_mid_dfn_xxG_vol', 'team_atck_dfn_xxG_vol',
    ] 
    cols.extend(dfn_cols)
#     print("defense", len(list(dfn_stats.loc[0].values))*2, len(dfn_cols))
    
    assert(len(stats)==len(cols))
    return stats, cols







def basic(game):
    
    game['next_action'] = game['type_id'].copy().shift(-1).fillna(0)
    
    # corners
    game['is_corner'] = np.where(game['secondary_type_id']==61, 1, 0)
    opp_corners, team_corners = game.groupby(['action_team'])['is_corner'].sum().values
    
    team_passes = game[(game['type_id']==30)&(game['action_team']==1)].copy()
    opp_passes = game[(game['type_id']==30)&(game['action_team']==0)].copy()
    
    team_switches = len(team_passes[(team_passes['x']<84) &(team_passes['next_action']==42) &(team_passes['dy'].abs()>30)&(~team_passes['secondary_type_id'].isin([62,63]))])
    opp_switches = len(opp_passes[(opp_passes['x']>36)&(opp_passes['next_action']==42) &(opp_passes['dy'].abs()>30)&(~opp_passes['secondary_type_id'].isin([62,63]))])
    
    #todo: maybe separate into low and high
    team_crosses = len(team_passes[(team_passes['x']>=84) &(team_passes['dy'].abs()>30)&(~team_passes['secondary_type_id'].isin([62,63]))])
    opp_crosses = len(opp_passes[(opp_passes['x']<=36) &(opp_passes['dy'].abs()>30)&(~opp_passes['secondary_type_id'].isin([62,63]))])
    
    team_cross_pct = len(team_passes[(team_passes['next_action']==42) &(team_passes['x']>=84) &(team_passes['dy'].abs()>30)&(~team_passes['secondary_type_id'].isin([62,63]))])/(team_crosses+1)
    opp_cross_pct = len(opp_passes[(opp_passes['next_action']==42) &(opp_passes['x']<=36) &(opp_passes['dy'].abs()>30)&(~opp_passes['secondary_type_id'].isin([62,63]))])/(opp_crosses+1)
    
    fouls_df = game.loc[game['type_id']==22]
    team_fouls = len(fouls_df[fouls_df['action_team']==1])
    opp_fouls = len(fouls_df[fouls_df['action_team']==0])
    
    team_foul_yellows = len(game[(game['action_team']==1)&(game['foul_committed_card_id'].isin([6,7]))])
    team_bad_yellows = len(game[(game['action_team']==1)&(game['bad_behaviour_card_id'].isin([6,7]))])
    opp_foul_yellows = len(game[(game['action_team']==0)&(game['foul_committed_card_id'].isin([6,7]))])
    opp_bad_yellows = len(game[(game['action_team']==0)&(game['bad_behaviour_card_id'].isin([6,7]))])

    team_yellows = team_foul_yellows + team_bad_yellows
    opp_yellows = opp_foul_yellows + opp_bad_yellows

    team_foul_sent_off = len(game[(game['action_team']==1)&(game['foul_committed_card_id'].isin([5,6]))])
    team_bad_sent_off = len(game[(game['action_team']==1)&(game['bad_behaviour_card_id'].isin([5,6]))])
    opp_foul_sent_off = len(game[(game['action_team']==0)&(game['foul_committed_card_id'].isin([5,6]))])
    opp_bad_sent_off = len(game[(game['action_team']==0)&(game['bad_behaviour_card_id'].isin([5,6]))])
    
    team_sent_off = team_foul_sent_off + team_bad_sent_off
    opp_sent_off = opp_foul_sent_off + opp_bad_sent_off

    team_pens = len(game[(game['shot_statsbomb_xg']==0.76)&(game['action_team']==1)])
    opp_pens = len(game[(game['shot_statsbomb_xg']==0.76)&(game['action_team']==0)])

    team_weighted = team_fouls +team_yellows+team_sent_off
    opp_weighted = opp_fouls+opp_yellows+opp_sent_off

    stats = [team_weighted, opp_weighted,opp_corners, team_corners,team_crosses,opp_crosses, team_cross_pct,opp_cross_pct,team_pens,opp_pens]
    cols = ['team_weighted_fouls','opp_weighted_fouls','opp_corners', 'team_corners','team_crosses','opp_crosses', 'team_cross_pct','opp_cross_pct','team_pens','opp_pens']
    
    assert(len(stats) == len(cols))
    return stats, cols






def add_game_clock(game):
    
    game['time'] = game['minute'].copy() + (game['second'].copy()/60)
    to_add = game.groupby(['period'])['time'].max().to_dict()
    to_add[0] = 45
    game['previous_period'] = game['period'].copy() - 1
    game['to_add'] = game['previous_period'].map(to_add)
    game['time'] = game['time'].copy() + game['to_add'].copy() - 45
    
    return game.drop(columns=['previous_period','to_add'])

def get_possession_speed(as_game, offense=True):
    
    # only offense or defense
    
    as_game['time_begin'] = as_game.groupby(['possession'])['time'].transform('min')
    as_game['time_end'] = as_game.groupby(['possession'])['time'].transform('max')
    if offense:
        as_game['x_small'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nsmallest(3).max())
        as_game['x_large'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nlargest(3).min())
        as_game['xxG_small'] = as_game.groupby(['possession'])['cg'].transform(lambda x: x.nsmallest(2).max())
        as_game['xxG_large'] = as_game.groupby(['possession'])['cg'].transform(lambda x: x.nlargest(2).min())
    else:
        as_game['x_small'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nsmallest(3).max())
        as_game['x_large'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nlargest(3).min())
        as_game['xxG_small'] = as_game.groupby(['possession'])['cc'].transform(lambda x: x.nsmallest(2).max())
        as_game['xxG_large'] = as_game.groupby(['possession'])['cc'].transform(lambda x: x.nlargest(2).min())
    as_game = as_game.drop_duplicates(subset=['possession'], keep='first').reset_index(drop=True)
    as_game = as_game[['id','poss_actions','time_begin','time_end','x_small','x_large','xxG_small','xxG_large']]
    
    if offense:
        as_game['poss_dx_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())*0.01/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
        as_game['poss_xxG_speed'] = (as_game['xxG_large'].copy()-as_game['xxG_small'].copy())*10/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
    else:
        as_game['poss_dx_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())*0.01/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
        as_game['poss_xxG_speed'] = (as_game['xxG_large'].copy()-as_game['xxG_small'].copy())*10/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
    # option to add as a per action basis as well
#     as_game['poss_action_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())/\
#     (as_game['poss_actions'])
#     as_game['poss_action_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())/\
#     (as_game['poss_actions'])
    
    if offense:
        as_game['side'] = 'offense'
    else:
        as_game['side'] = 'defense'
        
    return as_game

def advanced_tempo(game, team_id):
    
    # main function
    # both offense and defense
    
    game = add_game_clock(game)
    off = game.loc[game['possession_team_id']==team_id].reset_index(drop=True)
    dfn = game.loc[game['possession_team_id']!=team_id].reset_index(drop=True)

    off['poss_actions'] = off.groupby(['possession'])['type_id'].transform('count')
    dfn['poss_actions'] = dfn.groupby(['possession'])['type_id'].transform('count')

    off = off.loc[off['poss_actions']>=7].reset_index(drop=True)
    dfn = dfn.loc[dfn['poss_actions']>=7].reset_index(drop=True)
    
    off_ps = get_possession_speed(off, offense=True)
    dfn_ps = get_possession_speed(dfn, offense=False)
    
    poss_speed = pd.concat([off_ps, dfn_ps], axis=0).reset_index(drop=True)
    
    ofn_vecs = list(poss_speed.loc[poss_speed['side']=='offense'][['poss_dx_speed','poss_xxG_speed']].mean().values)
    dfn_vecs = list(poss_speed.loc[poss_speed['side']=='defense'][['poss_dx_speed','poss_xxG_speed']].mean().values)

    vecs = ofn_vecs + dfn_vecs
    cols = ['team_dx_sec','team_xxG_sec','opp_dx_sec','opp_xxG_sec']
    assert(len(vecs) == len(cols))
    return vecs, cols


 #### action embeddings ####   
fname = os.path.join(DROPBOX_PATH, 'models/action_embeds/skip_gram_5000.model')
model = FastText.load(fname)
wv_dict = model.wv.key_to_index
reverse = {v:k for k,v in wv_dict.items()}
embed_dim = 6

def map_vecs(x):
    return model.wv[x] 

def action_encode(df):
    return np.matrix(df['action_encoding'].apply(lambda x: map_vecs(x)).tolist())

def get_embeddings(game):

    embed_channels = []
    embed_X = action_encode(game)
    for i in range(embed_dim):
        col_name = f'action_embed_{i}'
        game[col_name] = embed_X[:, i]
        embed_channels.append(col_name)
        
    game = game.drop(columns=['action_encoding'])
    
    # also change to datetime
    game['timestamp'] = pd.to_datetime(game['timestamp'].copy())
    
    return game



def poss_stats_and_misc(game):
    
    unique_possessions = game.copy()[['match_id', 'possession', 'period', 'possession_team_id']].sort_values(['possession']).drop_duplicates(['match_id', 'possession']).reset_index(drop=True).sort_values(['match_id', 'possession'])
    unique_possessions['pos_shift'] = np.where(unique_possessions.groupby(['match_id', 'period'])['possession_team_id'].shift(1)==unique_possessions['possession_team_id'], np.nan,unique_possessions['possession'])
    unique_possessions['pos_shift'] = unique_possessions['pos_shift'].fillna(method="ffill")
    game = game.merge(unique_possessions[['match_id', 'possession', 'pos_shift']], how = 'left', on = ['match_id', 'possession'])

    possession_player_distance = game[(game['type_id']==43) & (game['action_team']==1) & (game['end_x']>20)].groupby(['match_id', 'player_id', 'team_id', 'pos_shift'])[['x', 'end_x']].agg(['min', 'max'])
    possession_player_distance.columns = ['min', 'max', 'min_end', 'max_end']
    possession_player_distance['length'] = possession_player_distance.apply(lambda x: max(x['max_end'], x['max']) - x['min'], axis = 1)
    # get max distance covered by a player for each possession
    possession_player_distance = possession_player_distance.sort_values('length', ascending = False).reset_index().drop_duplicates(['match_id', 'team_id', 'pos_shift'])
    # get game averages 
    game_average_lengths = possession_player_distance.length.sum()/11
    
    #misc
    gk_dist = game.loc[(game['secondary_type_id']==63)&(game['action_team']==1)]['distance'].median()
    if np.isnan(gk_dist):
        gk_dist = 38.10868453979492 # median over 80K games
    
    passes = game.copy().loc[(game['action_team']==1)&(game['type_id']==30)&(~game['secondary_type_id'].isin([61, 62, 63, 65]))]
    passes['remaining_field'] = (120 - passes['x'].copy()) + 10 
    # passes['']
    passes['fields_gained'] = passes['dx'].copy()/passes['remaining_field'].copy()
    fields_gained = passes.fields_gained.median()
    fields_gained_completed = passes.loc[passes['outcome_id']==0]['fields_gained'].sum()
    
    # attempt at defensive line height, can be improved
    med_def_action = game[(game['type_id'].isin([3,9,10,22,39]))&(game['action_team']==1)]['x'].median()
    
    # fixing error here
    game = get_embeddings(game)

    embeds = game.copy().groupby(['action_team','pos_shift']).agg({
        'cg':['max'],
        'cc':['max'],
        'action_embed_0':'mean',
        'action_embed_1':'mean',
        'action_embed_2':'mean',
        'action_embed_3':'mean',
        'action_embed_4':'mean',
        'action_embed_5':'mean'
    }).reset_index()
    offense = embeds.copy().loc[embeds['action_team']==1]
    defense = embeds.copy().loc[embeds['action_team']==0]

    offense.columns = [col[0] for col in list(offense)]
    defense.columns = [col[0] for col in list(defense)]
    threat_pp = offense.cg.mean()
    dthreat_pp = defense.cc.mean()

    embed_means = ['action_embed_0','action_embed_1','action_embed_2','action_embed_3','action_embed_4','action_embed_5']
    off_embeds = list(offense[embed_means].mean().values)
    def_embeds = list(defense[embed_means].mean().values)

    
    vecs = [game_average_lengths, gk_dist, fields_gained, fields_gained_completed, med_def_action]
    vecs.extend([threat_pp, dthreat_pp])
    vecs.extend(off_embeds)
    vecs.extend(def_embeds)
    cols = ['carry_lengths', 'gk_dist', 'fields_gained','fields_gained_comp', 'med_def_action']
    cols += ['threat_pp','dthreat_pp']
    cols = cols + ['off_embed_0','off_embed_1','off_embed_2','off_embed_3','off_embed_4','off_embed_5']
    cols = cols + ['def_embed_0','def_embed_1','def_embed_2','def_embed_3','def_embed_4','def_embed_5']
    
    
    
    # possession of possession, in other words, which possession belongs to each team
    poss_poss = game.groupby(['possession'])['action_team'].apply(lambda x: x.value_counts().sort_values(ascending=False).index[0]).reset_index()
    
    num_passes = len(game[game['type_id']==30])
    opp_passes = len(game[((game['type_id']==30)&(game['action_team']==0))])
    team_passes = len(game[((game['type_id']==30)&(game['action_team']==1))])
    opp_shots = len(game[((game['type_id']==16)&(game['action_team']==0))])
    team_shots = len(game[((game['type_id']==16)&(game['action_team']==1))])
    
    # total possessions for each team
    opp_poss = len(poss_poss) - poss_poss['action_team'].sum()
    team_poss = poss_poss['action_team'].sum()
    
    # passes per possession
    team_ppp = team_passes/team_poss
    opp_ppp = opp_passes/opp_poss

    # shots per possession
    team_spp = team_shots/team_poss
    opp_spp = opp_shots/opp_poss
    
    vecs.extend([opp_poss, opp_ppp, opp_spp, team_poss, team_ppp, team_spp])
    cols.extend(['opp_poss', 'opp_ppp', 'opp_spp','team_poss', 'team_ppp', 'team_spp'])
    
    game = add_game_clock(game)
    
    # adv possession
    # starts and ends
    poss_stats = game.groupby(['possession']).agg({
        'x':['first','last','min','max'],
        'y':['min','max'],
        'time':['min','max']
    }).reset_index()

    # first is irrelevant
    poss_poss = poss_poss[1:]
    poss_stats = poss_stats[1:]

    poss_stats.columns=['possession','x_first','x_last','x_min','x_max','y_min','y_max','time_min','time_max']
    
    poss_poss = pd.merge(poss_poss, poss_stats)
    
    # poss that enter attacking 3rd
    team_enter_in_A3 = len(poss_poss[(poss_poss.action_team==1)&(poss_poss.x_max >= 80)])
    opp_enter_in_A3 = len(poss_poss[(poss_poss.action_team==0)&(poss_poss.x_max <= 40)])
    
    xG_temp = game.loc[game['shot_statsbomb_xg']!=0.76].copy().groupby(['action_team']).agg({
        'shot_statsbomb_xg':['sum','mean']
    }).reset_index()

    xG_temp.columns=['action_team','sum','mean']
    opp_xG = xG_temp[xG_temp['action_team']==0]['sum'].values[0]
    opp_xG_mean = xG_temp[xG_temp['action_team']==0]['mean'].values[0]

    team_xG = xG_temp[xG_temp['action_team']==1]['sum'].values[0]
    team_xG_mean = xG_temp[xG_temp['action_team']==1]['mean'].values[0]
    
    txG_f3 = team_xG/(team_enter_in_A3 + 1)
    oxG_f3 = opp_xG/(opp_enter_in_A3 + 1)

    poss_stats['poss_length'] = np.abs(poss_stats['x_max'].copy()- poss_stats['x_min'].copy())
    poss_stats['poss_width'] = np.abs(poss_stats['y_max'].copy()- poss_stats['y_min'].copy())

    # poss_stats
    poss_stats['poss_time'] = poss_stats['time_max'].copy() - poss_stats['time_min'].copy()
#     poss_stats['poss_time'] = poss_stats['poss_time']

    poss_poss = pd.merge(poss_poss, poss_stats)
    
    # chopped
#     opp_poss, team_poss = poss_poss.groupby(['action_team'])['poss_time'].count().values

    opp_poss_start, team_poss_start = poss_poss.groupby(['action_team'])['x_first'].mean().values
    opp_poss_start = 120-opp_poss_start

    opp_poss_len, team_poss_len = poss_poss.groupby(['action_team'])['poss_length'].mean().values
    opp_poss_width, team_poss_width = poss_poss.groupby(['action_team'])['poss_width'].mean().values
    opp_poss_time_sum, team_poss_time_sum = poss_poss.groupby(['action_team'])['poss_time'].sum().values
    opp_poss_time_median, team_poss_time_median = poss_poss.groupby(['action_team'])['poss_time'].median().values

    vecs.extend([team_poss_start, opp_poss_start, team_poss_len, opp_poss_len,team_poss_width,opp_poss_width, 
                 team_poss_time_sum, opp_poss_time_sum, team_poss_time_median, opp_poss_time_median,
                oxG_f3, txG_f3])

    cols.extend(['team_poss_start','opp_poss_start','team_poss_len','opp_poss_len','team_poss_width','opp_poss_width',
             'team_poss_time_sum','opp_poss_time_sum','team_poss_time_median','opp_poss_time_median','oxG_f3', 'txG_f3'])
    
    game['is_leading'] = np.where(game['team_score']>game['opp_score'],1,0)
    game['is_leading'] = game['is_leading'].shift().fillna(0) # so it includes actual goal action

    game['is_trailing'] = np.where(game['team_score']<game['opp_score'],1,0)
    game['is_trailing'] = game['is_trailing'].shift().fillna(0) # so it includes actual goal action

    game['is_tied'] = np.where(game['team_score']==game['opp_score'],1,0)
    game['is_tied'] = game['is_tied'].shift().fillna(1) # so it includes actual goal action

    game_poss = game['possession'].max()
    lead = game[game['is_leading']==1].copy()
    tied = game[game['is_tied']==1].copy()
    trail = game[game['is_trailing']==1].copy()

    lead_poss = len(lead['possession'].unique())
    tied_poss = len(tied['possession'].unique())
    trail_poss = len(trail['possession'].unique())

    pct_lead = lead_poss/game_poss
    pct_tied = tied_poss/game_poss
    pct_trail = trail_poss/game_poss

    vecs.extend([pct_lead, pct_tied, pct_trail])
    cols.extend(['pct_lead','pct_tied','pct_trail'])

    assert(len(vecs) == len(cols))
    return vecs, cols 


def time_with_man_adv(temp):
    
    temp['man_adv'] = temp['team_strength'].copy() - temp['opp_strength'].copy()
    temp['timestamp'] = pd.to_datetime(temp['timestamp'])
    temp['time_elapsed'] = (temp['timestamp'].copy().diff().dt.seconds+(temp['timestamp'].copy().diff().dt.microseconds)/1e6)/60
    temp['time_elapsed'] = np.where(temp['time_elapsed']>1, 0, temp['time_elapsed'])
    temp['time_elapsed'] = np.where(temp['time_elapsed']<0, 0, temp['time_elapsed'])
    assert(temp['time_elapsed'].sum()>30)

    if temp['man_adv'].abs().sum() > 0:
        temp['twma'] = temp['man_adv'].copy() * temp['time_elapsed'].copy()
        vec = [temp['twma'].fillna(0).sum()]
    else:
        vec = [0]
        
    cols = ['man_adv']
    return vec, cols


def add_game_clock(game):
    
    game['time'] = game['minute'].copy() + (game['second'].copy()/60)
    to_add = game.groupby(['period'])['time'].max().to_dict()
    to_add[0] = 45
    game['previous_period'] = game['period'].copy() - 1
    game['to_add'] = game['previous_period'].map(to_add)
    game['time'] = game['time'].copy() + game['to_add'].copy() - 45
    
    return game.drop(columns=['previous_period','to_add'])

def get_possession_speed(as_game, offense=True):
    
    # only offense or defense
    
    as_game['time_begin'] = as_game.groupby(['possession'])['time'].transform('min')
    as_game['time_end'] = as_game.groupby(['possession'])['time'].transform('max')
    if offense:
        as_game['x_small'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nsmallest(3).max())
        as_game['x_large'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nlargest(3).min())
        as_game['xxG_small'] = as_game.groupby(['possession'])['cg'].transform(lambda x: x.nsmallest(2).max())
        as_game['xxG_large'] = as_game.groupby(['possession'])['cg'].transform(lambda x: x.nlargest(2).min())
    else:
        as_game['x_small'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nsmallest(3).max())
        as_game['x_large'] = as_game.groupby(['possession'])['x'].transform(lambda x: x.nlargest(3).min())
        as_game['xxG_small'] = as_game.groupby(['possession'])['cc'].transform(lambda x: x.nsmallest(2).max())
        as_game['xxG_large'] = as_game.groupby(['possession'])['cc'].transform(lambda x: x.nlargest(2).min())
    as_game = as_game.drop_duplicates(subset=['possession'], keep='first').reset_index(drop=True)
    as_game = as_game[['id','poss_actions','time_begin','time_end','x_small','x_large','xxG_small','xxG_large']]
    
    if offense:
        as_game['poss_dx_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())*0.01/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
        as_game['poss_xxG_speed'] = (as_game['xxG_large'].copy()-as_game['xxG_small'].copy())*10/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
    else:
        as_game['poss_dx_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())*0.01/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
        as_game['poss_xxG_speed'] = (as_game['xxG_large'].copy()-as_game['xxG_small'].copy())*10/\
        (as_game['time_end'].copy()-as_game['time_begin'].copy()+1)
    # option to add as a per action basis as well
#     as_game['poss_action_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())/\
#     (as_game['poss_actions'])
#     as_game['poss_action_speed'] = (as_game['x_large'].copy()-as_game['x_small'].copy())/\
#     (as_game['poss_actions'])
    
    if offense:
        as_game['side'] = 'offense'
    else:
        as_game['side'] = 'defense'
        
    return as_game

def advanced_tempo(game, team_id):
    
    # main function
    # both offense and defense
    
    game = add_game_clock(game)
    off = game.loc[game['possession_team_id']==team_id].reset_index(drop=True)
    dfn = game.loc[game['possession_team_id']!=team_id].reset_index(drop=True)

    off['poss_actions'] = off.groupby(['possession'])['type_id'].transform('count')
    dfn['poss_actions'] = dfn.groupby(['possession'])['type_id'].transform('count')

    off = off.loc[off['poss_actions']>=7].reset_index(drop=True)
    dfn = dfn.loc[dfn['poss_actions']>=7].reset_index(drop=True)
    
    off_ps = get_possession_speed(off, offense=True)
    dfn_ps = get_possession_speed(dfn, offense=False)
    
    poss_speed = pd.concat([off_ps, dfn_ps], axis=0).reset_index(drop=True)
    
    ofn_vecs = list(poss_speed.loc[poss_speed['side']=='offense'][['poss_dx_speed','poss_xxG_speed']].mean().values)
    dfn_vecs = list(poss_speed.loc[poss_speed['side']=='defense'][['poss_dx_speed','poss_xxG_speed']].mean().values)

    vecs = ofn_vecs + dfn_vecs
    cols = ['team_dx_sec','team_xxG_sec','opp_dx_sec','opp_xxG_sec']
    return vecs, cols

def patch_05022022(game):
    
    cols = []
    stats = []
    
    game['next_action'] = game['type_id'].copy().shift(-1).fillna(0)
    
    ## pass completions by third
    game['is_pass'] = np.where(game['type_id']==30, 1, 0)
    game['is_completed_pass'] = np.where(((game['type_id']==30)&(~game['outcome_id'].isin([9,74,75,76,77]))), 1, 0)
    game['is_forward_completed_pass'] = np.where(((game['type_id']==30)&(~game['outcome_id'].isin([9,74,75,76,77]))\
                                                  &(game['end_x']>game['x'])), 1, 0)

    pass_stats = game.copy().groupby(['action_team','third']).agg({
        'is_pass':'sum',
        'is_completed_pass':'sum',
        'is_forward_completed_pass':'sum'
    }).fillna(0)

    pass_stats['is_completed_pass'] = pass_stats['is_completed_pass']/pass_stats['is_pass'].copy()
    pass_stats['is_forward_completed_pass'] = pass_stats['is_forward_completed_pass']/pass_stats['is_pass'].copy()
    pass_stat_cols = ['opp_d3_passes','opp_d3_comp%','opp_d3_fcomp%',
     'opp_m3_passes','opp_m3_comp%','opp_m3_fcomp%',
     'opp_a3_passes','opp_a3_comp%','opp_a3_fcomp%',
     'team_d3_passes','team_d3_comp%','team_d3_fcomp%',
     'team_m3_passes','team_m3_comp%','team_m3_fcomp%',
     'team_a3_passes','team_a3_comp%','team_a3_fcomp%',
    ]
    pass_stat_vals = pass_stats.values.reshape(-1)
    stats.extend(pass_stat_vals)
    cols.extend(pass_stat_cols)
    
    
    ## extraneous shot on target
    game['is_shot'] = np.where((game['type_id']==16), 1, 0)
    game['is_SOT'] = np.where((game['type_id']==16)&(game['outcome_id'].isin([96,97,100,116])), 1, 0)
    game['is_saved'] = np.where((game['type_id']==16)&(game['outcome_id'].isin([100])), 1, 0)


    sot_extra = game.groupby(['action_team']).agg({
        'is_shot':'sum',
        'is_SOT':'sum',
        'is_saved':'sum'
    }).fillna(0.01)

    sot_extra['is_SOT'] = (sot_extra['is_SOT']/sot_extra['is_shot'].copy())
    sot_extra['is_saved'] = (sot_extra['is_saved']/sot_extra['is_shot'].copy())

    sot_cols = ['opp_shots','opp_SOT%','team_save%',
               'team_shots','team_SOT%','opp_save%']
    sot_stats = sot_extra.values.reshape(-1)

    SOT_stats = game.groupby(['action_team','is_SOT'])['shot_statsbomb_xg'].sum()
    SOT_counts = game.groupby(['action_team','is_SOT'])['shot_statsbomb_xg'].count()
    if len(SOT_stats) == 4:
        _, opp_XGOT, __, team_XGOT = SOT_stats.values
        _, opp_SOT, __, team_SOT = SOT_counts.values
    else:
        # in case no SOT
        try:
            opp_XGOT = SOT_stats.loc[0].loc[1]
            opp_SOT = SOT_counts.loc[0].loc[1]
        except:
            opp_XGOT = 0
            opp_SOT = 0.01
        try:
            team_XGOT = SOT_stats.loc[1].loc[1]
            team_SOT = SOT_counts.loc[1].loc[1]
        except:
            team_XGOT = 0
            team_SOT = 0.01

    sot_cols = list(sot_cols[1:3]) + list(sot_cols[4:])
    sot_stats = list(sot_stats[1:3]) + list(sot_stats[4:])
    sot_cols.extend(['team_XGOT/SOT','opp_XGOT/SOT'])
    sot_stats.extend([team_XGOT/team_SOT, opp_XGOT/opp_SOT])
    
    cols.extend(sot_cols)
    stats.extend(sot_stats)
    
    ### poss & switches
    game['poss_actions'] = game.groupby(['possession'])['type_id'].transform('count')
    game['poss_threshold'] = np.where(game['poss_actions']>=6, 1, 0)
    game['poss_start'] = game.groupby(['possession'])['x'].transform('first')
    game['poss_end'] = game.groupby(['possession'])['x'].transform('last')
    
    game['poss_top'] = game.groupby(['possession'])['y'].transform('max')
    game['poss_bottom'] = game.groupby(['possession'])['y'].transform('min')
    game['poss_width'] = game['poss_top'].copy()-game['poss_bottom'].copy()
    
    poss = game.copy().drop_duplicates(subset=['possession'])
    poss = poss.loc[poss['poss_threshold']==1].reset_index(drop=True)
    def cut(array, bins, labels, closed='right'):
        _bins = pd.IntervalIndex.from_tuples(bins, closed=closed)

        x = pd.cut(array, _bins)
        x.categories = labels # workaround for the bug
        return x
    labels = ['first', 'second', 'third']
    bins = [(0,40), (40,80), (80,120)]
    poss['poss_start_third'] = cut(poss['poss_start'].copy().values, bins=bins, labels=labels)
    poss['poss_end_third'] = cut(poss['poss_end'].copy().values, bins=bins, labels=labels)
    
    poss['is_wide'] = np.where(poss['poss_width']>54, 1, 0)
    opp_wide_poss = len(poss.loc[(poss['action_team']==0)&(poss['poss_width']>=54)])/len(poss.loc[poss['action_team']==0])
    team_wide_poss = len(poss.loc[(poss['action_team']==1)&(poss['poss_width']>=54)])/len(poss.loc[poss['action_team']==1])

    # would use groupby but need to handle zeros
    opp_end_a3=len(poss.loc[(poss['action_team']==0)&(poss['poss_end_third']=='first')])/len(poss.loc[poss['action_team']==0])
    opp_end_m3=len(poss.loc[(poss['action_team']==0)&(poss['poss_end_third']=='second')])/len(poss.loc[poss['action_team']==0])
    opp_end_d3=len(poss.loc[(poss['action_team']==0)&(poss['poss_end_third']=='third')])/len(poss.loc[poss['action_team']==0])

    team_end_d3=len(poss.loc[(poss['action_team']==1)&(poss['poss_end_third']=='first')])/len(poss.loc[poss['action_team']==1])
    team_end_m3=len(poss.loc[(poss['action_team']==1)&(poss['poss_end_third']=='second')])/len(poss.loc[poss['action_team']==1])
    team_end_a3=len(poss.loc[(poss['action_team']==1)&(poss['poss_end_third']=='third')])/len(poss.loc[poss['action_team']==1])

    team_passes = game[(game['type_id']==30)&(game['action_team']==1)].copy()
    opp_passes = game[(game['type_id']==30)&(game['action_team']==0)].copy()
    
    # forgot switches in game vecs
    team_switches = len(team_passes[(team_passes['x']<84) &(team_passes['next_action']==42) &(team_passes['dy'].abs()>30)&(~team_passes['secondary_type_id'].isin([62,63]))])
    opp_switches = len(opp_passes[(opp_passes['x']>36)&(opp_passes['next_action']==42) &(opp_passes['dy'].abs()>30)&(~opp_passes['secondary_type_id'].isin([62,63]))])
    
    game['is_pressure'] = np.where(game['type_id']==17, 1, 0)
    press = game.groupby(['action_team','third'])['is_pressure'].sum()
    opp_d3_press, opp_m3_press, opp_a3_press, team_d3_press, team_m3_press, team_a3_press = press.values

    stats.extend([opp_end_d3, opp_end_m3, opp_end_a3, team_end_d3, team_end_m3, team_end_a3,team_switches,opp_switches,
                opp_d3_press, opp_m3_press, opp_a3_press, team_d3_press, team_m3_press, team_a3_press,opp_wide_poss,team_wide_poss])
    cols.extend(['opp_end_d3', 'opp_end_m3', 'opp_end_a3', 'team_end_d3', 'team_end_m3', 'team_end_a3','team_switches','opp_switches',
                'opp_d3_press', 'opp_m3_press', 'opp_a3_press', 'team_d3_press', 'team_m3_press', 'team_a3_press','opp_wide_poss','team_wide_poss'])

    assert(len(stats)==len(cols))
    
    return stats, cols

def add_game_clock_2(game):

    """
    not efficient but ensures consistency for patch
    """
    
    game['time'] = game['minute'].copy() + (game['second'].copy()/60)
    to_add = game.groupby(['period'])['time'].max().to_dict()
    to_add[0] = 45
    game['previous_period'] = game['period'].copy() - 1
    game['to_add'] = game['previous_period'].map(to_add)
    game['time'] = game['time'].copy() + game['to_add'].copy() - 45
    game['difftime'] = game['time'].diff()
    return game.drop(columns=['previous_period','to_add'])

def patch_09052022(game, team_id):
    
    game = add_game_clock_2(game)
    # add "time in play"
    # I think the easiest way is to sum up time when actions are within 5.5 sec of each other
    # arbitrary threshold but less should be actual play and more should be time wasting (throw in, goal kick, etc)
    time_in_play = game[(game['difftime']>=0)&(game['difftime']<=0.055)]['difftime'].sum()
    dx_in_play = game[(game['difftime']>=0)&(game['difftime']<=0.055)]['dx'].abs().sum()
    
    # fixing bug on this before
    game['man_adv'] = game['team_strength'].copy() - game['opp_strength'].copy()
    if game['man_adv'].abs().sum() > 0:
        twma = game.copy().loc[game['man_adv']!=0].reset_index(drop=True)
        twma = twma.loc[(twma['difftime']>=0)&(twma['difftime']<=0.055)]
        twma['twma'] = twma['man_adv'].copy() * twma['difftime'].copy()
        stats = [twma['twma'].fillna(0).sum()]
    else:
        stats = [0]
        
    cols = ['man_adv_v2']

    game['open_play'] = np.where(game['play_pattern_id'].copy().isin([1,4,9,9,8,7]), 1, 0)
    game['counter'] = np.where(game['play_pattern_id'].copy().isin([6]), 1, 0)
    
    off = game.loc[game['possession_team_id']==team_id].reset_index(drop=True)
    dfn = game.loc[game['possession_team_id']!=team_id].reset_index(drop=True)

    # probably best to go in advanced tempo function
    field_tilt = off.groupby(['third'])['id'].count()[3]/(off.groupby(['third'])['id'].count()[3] + dfn.groupby(['third'])['id'].count()[1])

    game['poss_action_count'] = game.groupby(['possession'])['id'].transform('count')
    test = game.copy().loc[game['poss_action_count']>=7]
    poss_poss = test.groupby(['possession'])['action_team'].apply(lambda x: x.value_counts().sort_values(ascending=False).index[0]).reset_index()
    opp_poss = poss_poss.groupby(['action_team'])['possession'].count()[0]
    team_poss = poss_poss.groupby(['action_team'])['possession'].count()[1]
    team_pressures = len(game[((game['type_id']==17)&(game['action_team']==1))])
    opp_pressures = len(game[((game['type_id']==17)&(game['action_team']==0))])
    press_per_opp_poss, opp_press_per_opp_poss = team_pressures/opp_poss, opp_pressures/team_poss

    pct_open_play = off.loc[off['open_play']==1]['obv_for_net'].sum()/off['obv_for_net'].sum()
    pct_counter = off.loc[off['counter']==1]['obv_for_net'].sum()/off['obv_for_net'].sum()

    opp_pct_open_play = dfn.loc[dfn['open_play']==1]['obv_against_net'].sum()/dfn['obv_against_net'].sum()
    opp_pct_counter = dfn.loc[dfn['counter']==1]['obv_against_net'].sum()/dfn['obv_against_net'].sum()
    
    stats.extend([time_in_play, dx_in_play, field_tilt, press_per_opp_poss, opp_press_per_opp_poss,pct_open_play,pct_counter,opp_pct_open_play,opp_pct_counter ])
    cols.extend(['time_in_play','dx_in_play', 'field_tilt','press_per_opp_poss','opp_press_per_opp_poss','pct_open_play','pct_counter','opp_pct_open_play','opp_pct_counter'])
    return stats, cols




def STF2gvec(single_game):
    
    game_vec = []
    cols = []
    
    single_game = preprocessing(single_game)

    tvec, threat_cols = threat(single_game)
    bvec, basic_cols = basic(single_game)
    
    pvec, poss_cols = poss_stats_and_misc(single_game)
    man_adv, man_adv_cols = time_with_man_adv(single_game)
    tid = single_game['team_id'].values[0]
    at, at_cols = advanced_tempo(single_game, tid)
    p1, p1_cols = patch_05022022(single_game)
    p2, p2_cols = patch_09052022(single_game, tid)
    
    game_vec.extend(tvec+bvec+pvec+man_adv+at+p1+p2)
    cols.extend(threat_cols+basic_cols+poss_cols+man_adv_cols+at_cols+p1_cols+p2_cols)
    
    assert(len(game_vec) == len(cols))
    
    
    return game_vec, cols



