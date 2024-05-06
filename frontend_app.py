import streamlit as st
from bokeh.plotting import figure, show
from bokeh.io import push_notebook, show
from bokeh.models import ColumnDataSource, PointDrawTool
from bokeh.io import curdoc
from bokeh.resources import CDN
from bokeh.embed import file_html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import json_normalize

from gnn_model_inference import *

import base64
import requests
from PIL import Image
from io import BytesIO

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

teams_data = pd.read_csv('data/male_teams.csv')
teams_data = teams_data[['team_name', 'attack', 'defence']]

events = pd.read_json('3869685_events.json')
frames = pd.read_json('3869685_360_data.json')

events['pass_type'] = events['pass'].apply(lambda x: x['type']['name'] if isinstance(x, dict) and 'type' in x and 'name' in x['type'] else np.nan)
events['pass_shot_assist'] = events['pass'].apply(lambda x: x['shot_assist'] if isinstance(x, dict) and 'shot_assist' in x else False)
events['attack_team'] = events['team'].apply(lambda x: x['name'] if isinstance(x, dict) and 'name' in x else np.nan)

corners = events[events['pass_type']=='Corner']
corners_wframes = pd.merge(corners, frames, left_on='id', right_on='event_uuid', how='inner')

st_frame = corners_wframes[['event_uuid','visible_area','freeze_frame']]

df_exploded = st_frame.explode('freeze_frame')
df_normalized = json_normalize(df_exploded['freeze_frame'])
df_normalized['event_uuid'] = df_exploded['event_uuid'].values
df_normalized['visible_area'] = df_exploded['visible_area'].values
df_normalized[['x', 'y']] = pd.DataFrame(df_normalized['location'].tolist(), index=df_normalized.index)
df_normalized.drop(['location'], axis=1, inplace=True)

## Make a custom preset for the initial position of players
frame_idx = 20
frame_id = df_normalized.iloc[frame_idx].event_uuid

visible_area = np.array(df_normalized.iloc[frame_idx].visible_area).reshape(-1, 2)
visible_area2 = list(map(tuple, visible_area))
player_position_data = df_normalized[df_normalized.event_uuid == frame_id]

teammate_locs = player_position_data[player_position_data.teammate]
opponent_locs = player_position_data[~player_position_data.teammate]


# Create the Bokeh plot
def create_plot():
    p = figure(x_range=(0, 125), y_range=(0, 85), width=900, height=600)
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False

    image_url = ['https://i.ibb.co/gPtpTvf/football-pitch-horizontal.png']
    p.image_url(url=image_url, x=0, y=0, w=124, h=84, anchor="bottom_left")

    attack_players = ColumnDataSource(data={'x': teammate_locs['x'].tolist(), 'y': teammate_locs['y'].tolist()})
    defense_players = ColumnDataSource(data={'x': opponent_locs['x'].tolist(), 'y': opponent_locs['y'].tolist()})

    attack_nodes = p.scatter(x='x', y='y', source=attack_players, color='crimson', size=20)
    defense_nodes = p.scatter(x='x', y='y', source=defense_players, color='skyblue', size=20)

    draw_tool1 = PointDrawTool(renderers=[attack_nodes], empty_value='black')
    draw_tool2 = PointDrawTool(renderers=[defense_nodes], empty_value='black')
    p.add_tools(draw_tool1, draw_tool2)

    #p.patch(x=[x[0] for x in visible_area2], y=[y[1] for y in visible_area2], color="red", alpha=0.3)

    return p, attack_players, defense_players


image_path = 'soccer_net_logo.jpg'

image_base64 = get_image_base64(image_path)

# Create the HTML string with embedded base64 image
html_str = f"""
<div style="text-align: center;">
    <img src='data:image/png;base64,{image_base64}' width='300'/>
    <h1>Corners Tactic Board</h1>
</div>
"""
# Display the HTML in Streamlit
st.markdown(html_str, unsafe_allow_html=True)

#st.markdown('<h1 style="text-align: center;">Corner Tactics Board</h1>', unsafe_allow_html=True)

# Define team options for the dropdowns
team_names1 = ['Manchester United', 'Manchester City', 'Liverpool', 'Arsenal', 'Aston Villa']
team_names2 = ['Manchester City', 'Manchester United', 'Liverpool', 'Arsenal', 'Aston Villa']

# Create a row with two columns for the dropdowns
col1, col2 = st.columns(2)

# Attack team dropdown
with col1:
    selected_attack_team = st.selectbox('Attacking Team', team_names1)

# Defense team dropdown
with col2:
    selected_defense_team = st.selectbox('Defending Team', team_names2)

attack_rating = teams_data[teams_data['team_name'] == selected_attack_team]['attack'].values[0]
defense_rating = teams_data[teams_data['team_name'] == selected_defense_team]['defence'].values[0]

plot, attack_source, defense_source = create_plot()

col3, col4, col5 = st.columns([2.5, 0.8, 0.8])

with col3:
    st.write('Calculate the probability that this setup will lead to a goal')
with col4:
    calculate_pressed = st.button('Calculate')
with col5:
    reset_pressed = st.button('Reset')

if reset_pressed:
    st.rerun()

with col3:
    if calculate_pressed:
        attack_data = attack_source.data
        defense_data = defense_source.data
        probability = get_prediction(attack_data, defense_data, attack_rating, defense_rating)
        st.write(f'There is a {round(100*probability,2)}% chance that this setup will lead to a goal')

st.bokeh_chart(plot)
