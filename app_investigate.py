import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from pytz import timezone
from datetime import datetime, timedelta
from datetime import time as dt_time  # Importing 'time' with an alias
import plotly.graph_objects as go
import scipy.stats as stats
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

#set page config to wide first
st.set_page_config(layout="wide")

@st.cache_data
def get_df() :

    dt=datetime.now()
    url = "https://altoherobackend-staging.azurewebsites.net/api/v2.0/login/"

    payload = json.dumps({
        "username": "arbour_fom",
        "password": "password123"
    })

    headers = {"Content-Type" : "application/json"}

    response = requests.post(url, headers = headers, data=payload)
    access = json.loads(response.text)['access']

    start_date_str = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date_str = dt.strftime("%Y-%m-%d")#"2023-11-11"

    url = f"https://altoherobackend.azurewebsites.net/api/v2.0/work_orders/no-pagination/?start_date={start_date_str}&end_date={end_date_str}&limit=9999&site=30"

    headers = {
        'Authorization': f'Bearer {access}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    results = json.loads(response.text)
    return pd.DataFrame(results['results'], dtype = str)

df = get_df()

df_dnd = df[df['do_not_disturb']=='True']

def DataPreparation_DataFrame(df) :
    df.rename(columns = {"status":"Status", 
                         "cleaning_type":"Cleaning_type",
                         "started_at":"Started At", 
                         "cleaning_finished_at":"Cleaning Finished At", 
                         "end_at":"End At", 
                         "started_inspecting_at":"Started InspectingAt", 
                         "pause_continue_at":"Pause Continue At", 
                         "inspected_by":"inspected_by"}, inplace = True)
    
    def PauseTime_Calculation(pause_list_string):
        if pause_list_string == "0" :
            return 0
        else :
            pause_list = eval(pause_list_string)
            durations = []
            for i in pause_list:
                if i[0] != None and i[1] != None :
                    pause_time = datetime.strptime(i[0], '%Y-%m-%dT%H:%M:%S.%fZ')
                    continue_time = datetime.strptime(i[1], '%Y-%m-%dT%H:%M:%S.%fZ')
                    duration = (continue_time - pause_time).total_seconds()/60
                    durations.append(duration)
                return sum(durations)


    df = df[~df["Cleaning_type"].isnull()]
    df = df[df["do_not_disturb"] == "False"]

    df["Started At"] = pd.to_datetime(df["Started At"]) + timedelta(hours=7)
    df["created_at"] = pd.to_datetime(df["created_at"]) + timedelta(hours=7)
    df["Cleaning Finished At"] = pd.to_datetime(df["Cleaning Finished At"]) + timedelta(hours=7)
    df["Started InspectingAt"] = pd.to_datetime(df["Started InspectingAt"]) + timedelta(hours=7)
    df["End At"] = pd.to_datetime(df["End At"]) + timedelta(hours=7)
    df["Date"] = df["created_at"].dt.strftime("%Y/%m/%d")
    df["Time"] = df["Started At"].apply(lambda x : x.replace(day = 1, month = 1, year = 2000))

    df["Pause Continue At"].fillna("0", inplace = True)
    df["Cleaning_time"] = (df["Cleaning Finished At"]-df["Started At"]).dt.total_seconds()/60
    df["Pause_time"] = df["Pause Continue At"].apply(PauseTime_Calculation)
    df["Total_cleaning_time"] = df["Cleaning_time"] - df["Pause_time"]

    df["Room"] = df["room"].apply(lambda x : eval(x)["room_name"])
    
    df["assigned_to"].fillna(str({"first_name" : None}), inplace = True)
    df["Assigned To"] = df["assigned_to"].apply(lambda x : eval(x)["first_name"])

    df["Inspection time"] = (df["End At"] - df["Cleaning Finished At"]).dt.total_seconds() / 60
    df["inspection_duration"] = (df["End At"] - df["Started InspectingAt"]).dt.total_seconds() / 60

    site = eval(df["site"].iloc[0])["site_name"]


    df_roomtype = pd.read_csv("Arbour_Roomtype.csv", dtype = str)
    df = df.merge(df_roomtype, on = "Room", how = "left")

    if site == "Aster Hotel and Residence" :
        site = "Aster"
    ###
    if site == "Arbour Hotel and Residence" :
        site = "Arbour"

    if site == "Aster" :
        df["Floor"] = df["Room"].str.slice(start = 1, stop = 2)
    ###
    elif site == "Arbour" :
        df["Floor"] = df["Room"].apply(lambda x : x[0:2] if len(x) == 4 else " " + x[0:1])
        ###
        df["Room"] = df["Room"].apply(lambda x : " " + x if len(x) == 3 else x)
    else :
        df["Floor"] = df["Room"].str.slice(stop = 1)
    
    return df

df = DataPreparation_DataFrame(df)

def pivot_table(df):
    # st.subheader('pivot table: ')

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    with col1:
        image = Image.open('logo_hat2.jpg')
        st.image(image,use_column_width ='auto')

    with col2:
        metric2_placeholder = st.empty()
    with col3:
        metric3_placeholder = st.empty()
    with col4:
        metric4_placeholder = st.empty()
    with col5:
        metric5_placeholder = st.empty()
    with col6:
        metric6_placeholder = st.empty()
    with col7:
        metric7_placeholder = st.empty()
    with col8:
        metric8_placeholder = st.empty()
    with col9:
        metric9_placeholder = st.empty()

    col_headers = df.columns.tolist()

    default_columns = ['id', 'title', 'description', 'updated_at', 
                    'created_at', 'Started At', 'pause_at', 'continue_at', 
                    'Pause Continue At', 'End At',  
                    'Cleaning Finished At', 'Started InspectingAt', 
                    #'pause_description', #'finished_description',
                    'Status', 
                    #'assigned_to', 
                    #'assigned_by', #'created_by', 
                    #'room', 
                    'report_type', 
                    #'report_urgent', 'report_fix_type', 'progress', 
                    #'problem_description', 'problem_task_transfer', 
                    #'site', 'origin_site', 'assigned_department', 
                    #'attachments', 'schedule_start', 'schedule_end', 
                    'Cleaning_type', 
                    #'hq_report_type', 'expected_complete_date', 
                    'do_not_disturb', 'inspected_by', 
                    'Date', 'Time', 'Cleaning_time', 'Pause_time', 'Total_cleaning_time', 
                    'Room', 'Assigned To', 'Inspection time', 'Floor', 'Room_num']


    col1, col2 = st.columns([3,2])
    with col1:
        default_pivot_columns = ['Floor','Assigned To','Cleaning_type','Roomtype','Status']
        pivot_columns = st.multiselect('choose filter category (sequential filter)',
                                    col_headers,
                                    default= default_pivot_columns)
        
    with col2:
        date_select = st.multiselect('choose date(s)', df['Date'].unique().tolist(),default=df['Date'].unique().tolist()[0])


    df_pivot = df.copy()
    df_pivot = df_pivot[df_pivot['Date'].isin(date_select)]

    col1, col2 = st.columns([1,5])
    with col1:
        if pivot_columns:
            for columns_names in pivot_columns:
                column_filter = st.multiselect(str(columns_names), #label
                                            df_pivot[columns_names].unique().tolist(), #choices
                                            key =str(columns_names)) #key as it's name
                if column_filter:
                    df_pivot = df_pivot[df_pivot[columns_names].isin(column_filter)]


    with col2:

        quickview_columns = ['Room','Cleaning_type', 'Cleaning_time','Assigned To','Inspection time','inspection_duration','inspected_by',
                        'title','created_at', 'Started At', 'pause_at', 'continue_at', 
                        'Pause Continue At', 'End At',  
                        'Cleaning Finished At', 'Started InspectingAt',    
                        'id']


        col2a, col2b, col2c, col2d, col2e = st.columns([1,1,1,1,1])
        with col2a:
            column_name_filter = st.radio('column title options ',['Quick View','Full'])

        with col2b:
            #this one is lower bound for ct filter
            min_ct = st.number_input("minimum cleaning time (m)", value=df_pivot['Cleaning_time'].min(), placeholder="input min ct...")
        
        with col2c:
            #this on is upper bound for ct filter
            max_ct = st.number_input("maximum cleaning time (m)", value=df_pivot['Cleaning_time'].max(), placeholder="input max ct...")
        
        with col2d:
            #for scatter plot below
            color_by = st.selectbox('colour by',['Cleaning_type','Assigned To','Floor','Roomtype'])

        with col2e:    
            y_axis_choice = st.selectbox('y axis',['Cleaning_time','Inspection time','inspection_duration','Floor'])
              
        col2i, col2ii = st.columns([3,2])

        with col2i:
            # the actual filtering part for df_pivot
            df_pivot = df_pivot[(df_pivot['Cleaning_time'] >= min_ct) & (df['Cleaning_time'] <= max_ct)]

            if column_name_filter == 'Quick View':
                
                st.write(df_pivot[quickview_columns])

            
            else:
                choose_columns_to_display = st.multiselect('choose columns',df.columns.tolist(),default=df.columns.tolist())
                st.write(df_pivot[choose_columns_to_display])

        with col2ii:
            fig = px.scatter(df_pivot, 
                             x = 'Cleaning Finished At', #if live started at wont have time taken
                             y = y_axis_choice, 
                             color = color_by,
                             hover_name = 'Room',
                             hover_data=['Cleaning_time','Assigned To'],
                            #  size = 'Cleaning_time',
                             color_discrete_map = {'C/O': '#0068C9', 'OD': '#83C9FF', 'VC': '#FFABAB', 'DND' : '#FF2B2B'}
                             )
            fig.update_layout(
                            yaxis = dict(
                                tickmode = 'linear',
                                tick0 = 0,
                                dtick = 15
                                ),
                            xaxis=dict(showgrid=True)
                            )
            st.plotly_chart(fig, use_container_width=True)

    
#time to show some general statistics

    # c/o median ct
    a = df_pivot.loc[df_pivot['Cleaning_type'] == 'C/O', 'Cleaning_time'].median().round(1)
    b = df.loc[(df['Cleaning_type'] == 'C/O') & (df['Date'].isin(date_select)), 'Cleaning_time'].median().round(1)
    metric2_placeholder.metric(label="c/o ct (median)", 
            value = a,
            delta = str((-100*(1-(a/b))).round(1)) + '%',
            delta_color = 'inverse')  

    # c/o mean ct
    a = df_pivot.loc[df_pivot['Cleaning_type'] == 'C/O', 'Cleaning_time'].mean().round(1)
    b = df.loc[(df['Cleaning_type'] == 'C/O') & (df['Date'].isin(date_select)), 'Cleaning_time'].mean().round(1)
    metric3_placeholder.metric(label="c/o ct (mean)", 
            value = a,
            delta = str((-100*(1-(a/b))).round(1)) + '%',
            delta_color = 'inverse')
    

    # od median ct
    a = df_pivot.loc[df_pivot['Cleaning_type'] == 'OD', 'Cleaning_time'].median().round(1)
    b = df.loc[(df['Cleaning_type'] == 'OD') & (df['Date'].isin(date_select)), 'Cleaning_time'].median().round(1)
    metric4_placeholder.metric(label="od ct (median)", 
            value = a,
            delta = str((-100*(1-(a/b))).round(1)) + '%',
            delta_color = 'inverse')

    # od mean ct
    a = df_pivot.loc[df_pivot['Cleaning_type'] == 'OD', 'Cleaning_time'].mean().round(1)
    b = df.loc[(df['Cleaning_type'] == 'OD') & (df['Date'].isin(date_select)), 'Cleaning_time'].mean().round(1)
    metric5_placeholder.metric(label="od ct (mean)", 
            value = a,
            delta = str((-100*(1-(a/b))).round(1)) + '%',
            delta_color = 'inverse')
    

    # median inspection gap
    a = df_pivot['Inspection time'].median().round(1)
    b = df.loc[(df['Date'].isin(date_select)), 'Inspection time'].median().round(1)
    metric6_placeholder.metric(label="insp. gap (median)", 
                value = a,
                delta = str((-100*(1-(a/b))).round(1)) + '%',
                delta_color = 'inverse')

    # mean inspection gap
    a = df_pivot['Inspection time'].mean().round(1)
    b = df.loc[(df['Date'].isin(date_select)), 'Inspection time'].mean().round(1)
    metric7_placeholder.metric(label="insp. gap (mean)", 
                value = a,
                delta = str((-100*(1-(a/b))).round(1)) + '%',
                delta_color = 'inverse')
    

    # median inspection time
    a = df_pivot['inspection_duration'].median().round(3)
    b = df.loc[(df['Date'].isin(date_select)), 'inspection_duration'].median().round(1)
    metric8_placeholder.metric(label="insp. time (median)", 
                value = a,
                delta = str((-100*(1-(a/b))).round(1)) + '%',
                delta_color = 'inverse')

    # mean inspection time
    a = df_pivot['inspection_duration'].mean().round(1)
    b = df.loc[(df['Date'].isin(date_select)), 'inspection_duration'].mean().round(1)
    metric9_placeholder.metric(label="insp. time (mean)", 
                value = a,
                delta = str((-100*(1-(a/b))).round(1)) + '%',
                delta_color = 'inverse')

pivot_table(df)
st.write('--------------------------------------------------')

def time_slicer(df):
    df['Cleaning Finished At'] = pd.to_datetime(df['Cleaning Finished At'])
    df['Time'] = df['Cleaning Finished At'].dt.time

    # Create two time objects for the default values of the slider
    start_time_default = dt_time(7, 0)
    end_time_default = dt_time(20, 0)

    # Create a time range slider
    time_range = st.slider(
        "Select a time range",
        value=(start_time_default, end_time_default),
        format="HH:mm"
    )

    # Filter the DataFrame
    # filtered_df = df[df['Time'].between(time_range[0], time_range[1])]

    datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])

    def update_row_based_on_time(row, time_range):
        start_time, end_time = time_range
        for col in datetime_columns:
            if pd.notna(row[col]):
                col_time = row[col].time()
                if not (start_time <= col_time <= end_time):
                    row[col] = pd.NaT
        return row
    
    df = df.apply(update_row_based_on_time, time_range=time_range, axis=1)

    return df

dfs = time_slicer(df)


colchart1, colchart2, colchart3 = st.columns([2,2,3])
with colchart1:
    def burnUpChartCompareDates(df):
        date_choice = st.multiselect('select dates:',df['Date'].unique().tolist(), default = df['Date'].unique().tolist(),
                                    key = 'burnUpChartCompareDates')
        df = df[df['Date'].isin(date_choice)]

        #we create a timevalue column for cleaning finished at specifically to plot this
        df['cfa_time'] = df['Cleaning Finished At'].apply(lambda x : x.replace(day = 1, month = 1, year = 2000))

        df['Date_temp'] = pd.to_datetime(df['Date'])
        df.sort_values(by=['Date_temp', 'cfa_time'], inplace=True)
        df['queue'] = df.groupby('Date_temp').cumcount() + 1
        df['Cleaning_type_queue'] = df.groupby(['Date_temp', 'Cleaning_type']).cumcount() + 1
        df = df.drop(columns=['Date_temp'])
        df['Cleaning_time'] = df['Cleaning_time'].fillna(0)
        fig_burnup = px.scatter(df, x='cfa_time', y='queue', color='Date', hover_name='Room', hover_data=['Assigned To','Total_cleaning_time'])
        fig_burnup.update_layout(xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), paper_bgcolor='#FFFFFF')
        fig_burnup.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
        st.plotly_chart(fig_burnup, use_container_width=True)
    burnUpChartCompareDates(df)

with colchart2:

    def ctDoneBarChart(df):
        date_choice = st.multiselect('select dates:',df['Date'].unique().tolist(), default=df['Date'].unique().tolist()[0],
                                key = 'ctDoneBarChart')
        df = df[df['Date'].isin(date_choice)]
        
        df['total_work'] = df['Cleaning_type'].notna()
        df['total_cleaned'] = df['Cleaning Finished At'].notna()
        df['total_inspected'] = df['End At'].notna()
        room_status = df.groupby('Cleaning_type').agg({
            'total_cleaned': 'sum',
            'total_inspected': 'sum',
            'total_work': 'sum'
        }).reset_index()            

        melted_df = room_status.melt(id_vars='Cleaning_type', 
                                var_name='Category', 
                                value_name='Count')

        # melted_df
        # custom_order = ['PREK','DLXK','DLXT','PCVK','DCVK','PHCV','DHOV','TBSO','HSOV','PSWP']
        fig = px.bar (melted_df, x = 'Cleaning_type', y = 'Count', 
                        color='Category',barmode='group',
                        text_auto = True)
        # fig.update_xaxes(categoryorder='array', categoryarray=custom_order)
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig, use_container_width=True)        
    ctDoneBarChart(dfs)


with colchart3:
    def rtDoneBarChart(df):
        col1, col2 = st.columns([1,1])
        with col1:
            date_choice = st.multiselect('select dates:',df['Date'].unique().tolist(), default=df['Date'].unique().tolist()[0],
                                    key = 'rtDoneBarChart_date')
            df = df[df['Date'].isin(date_choice)]
        with col2:
            rt_choice = st.selectbox('select cleaning types:',['C/O','OD','All'], key = 'rtDoneBarChart_rt')

            if rt_choice == 'C/O':
                # Calculating "total_done_cleaned" and "total_inspected"
                df['total_work'] = df['Cleaning_type'] == 'C/O'
                df['total_cleaned'] = (df['Cleaning_type'] == 'C/O') & df['Cleaning Finished At'].notna()
                df['total_inspected'] = (df['Cleaning_type'] == 'C/O') & df['End At'].notna()
                room_status = df.groupby('Roomtype').agg({
                    'total_cleaned': 'sum',
                    'total_inspected': 'sum',
                    'total_work': 'sum'
                }).reset_index()

            elif rt_choice == 'OD':
                # Calculating "total_done_cleaned" and "total_inspected"
                df['total_work'] = df['Cleaning_type'] == 'OD'
                df['total_cleaned'] = (df['Cleaning_type'] == 'OD') & df['Cleaning Finished At'].notna()
                df['total_inspected'] = (df['Cleaning_type'] == 'OD') & df['End At'].notna()
                room_status = df.groupby('Roomtype').agg({
                    'total_cleaned': 'sum',
                    'total_inspected': 'sum',
                    'total_work': 'sum'
                }).reset_index()

            elif rt_choice == 'All':
                # Calculating "total_done_cleaned" and "total_inspected"
                df['total_work'] = df['Cleaning_type'].notna()
                df['total_cleaned'] = df['Cleaning Finished At'].notna()
                df['total_inspected'] = df['End At'].notna()
                room_status = df.groupby('Roomtype').agg({
                    'total_cleaned': 'sum',
                    'total_inspected': 'sum',
                    'total_work': 'sum'
                }).reset_index()            

        melted_df = room_status.melt(id_vars='Roomtype', 
                               var_name='Category', 
                               value_name='Count')
        
        # melted_df
        custom_order = ['PREK','DLXK','DLXT','PCVK','DCVK','PHCV','DHOV','TBSO','HSOV','PSWP']
        fig = px.bar (melted_df, x = 'Roomtype', y = 'Count', 
                      color='Category',barmode='group',
                      text_auto = True)
        fig.update_xaxes(categoryorder='array', categoryarray=custom_order)
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig, use_container_width=True)
    rtDoneBarChart(dfs)

# df_dnd[df_dnd['cleaning_type'].isin(['C/O','OD'])]

col1, col2, col3 = st.columns([1,1,1])

# with col1:
        # def dnd_display(df_dnd):
        #     df_dnd["created_at"]=pd.to_datetime(df["created_at"]) + timedelta(hours=7)
        #     df_dnd['Date']=df_dnd["created_at"].dt.strftime("%Y/%m/%d")
        #     df_dnd["Room"] = df_dnd["room"].apply(lambda x : eval(x)["room_name"])
        #     date_choice = st.multiselect('select dates:',df_dnd['Date'].dropna().unique().tolist(), default=df['Date'].unique().tolist()[0],
        #                     key = 'dnd_display')
        #     df_dnd = df_dnd[df_dnd['Date'].isin(date_choice)]

        #     dnd_to_diplay = st.multiselect('select dnd room to view:',df_dnd['Room'], default= df_dnd['Room'])

        #     # st.write('dnd today: ')
        #     # st.write(df_dnd['Room'])

        # dnd_display(df_dnd)

with col1:
    def inspection_trace (df):
        date_choice = st.multiselect('select dates:',df['Date'].unique().tolist(), default = df['Date'].unique().tolist(),
                                    key = 'inspection_trace')
        df = df[df['Date'].isin(date_choice)]

        #we create a timevalue column for cleaning finished at specifically to plot this
        df['cfa_time'] = df['Cleaning Finished At'].apply(lambda x : x.replace(day = 1, month = 1, year = 2000))

        df['Date_temp'] = pd.to_datetime(df['Date'])
        df.sort_values(by=['Date_temp', 'cfa_time'], inplace=True)
        df['queue'] = df.groupby('Date_temp').cumcount() + 1
        df['Cleaning_type_queue'] = df.groupby(['Date_temp', 'Cleaning_type']).cumcount() + 1
        df = df.drop(columns=['Date_temp'])
        df['Cleaning_time'] = df['Cleaning_time'].fillna(0)

        fig = go.Figure()

        # Add scatter plots with custom hover info
        fig.add_trace(go.Scatter(
            x=df['End At'],
            y=df['queue'],
            mode='markers',
            name='End Time',
            marker=dict(color='lightblue'),
            text=df['Room'] + '<br>' + df['Cleaning_type'] + '<br>' + df['Inspection time'].astype(str),
            hoverinfo='text'
        ))

        fig.add_trace(go.Scatter(
            x=df['End At'],
            y=df['queue'],
            mode='markers',
            name='CFA Time',
            marker=dict(color='salmon'),
            text=df['Room'] + '<br>' + df['Cleaning_type'] + '<br>' + df['Inspection time'].astype(str),
            hoverinfo='text'
        ))

        # Update layout
        fig.update_layout(
            title="Scatter plot of end_time vs cfa_time counters",
            xaxis_title="Time",
            yaxis_title="Counter",
            legend_title="Variable"
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    inspection_trace (df)


with col2:
    
    def burnUpChartCompareDates2(df):

        date_choice = st.multiselect('select dates:',df['Date'].unique().tolist(), default = df['Date'].unique().tolist(),
                                    key = 'dsburnUpChartCompareDates2')
        df = df[df['Date'].isin(date_choice)]


        rt_choice = st.multiselect('select rt:',df['Roomtype'].unique().tolist(), default = df['Roomtype'].unique().tolist(),
                                    key = 'rtburnUpChartCompareDates2')
        df = df[df['Roomtype'].isin(rt_choice)]            

        # rt_choice = st.multiselect('select dates:',df['Date'].unique().tolist(), default = df['Date'].unique().tolist(),
        #                             key = 'burnUpChartCompareDates2')
        # df = df[df['Date'].isin(date_choice)]

        #we create a timevalue column for cleaning finished at specifically to plot this
        df['cfa_time'] = df['Cleaning Finished At'].apply(lambda x : x.replace(day = 1, month = 1, year = 2000))

        df['Date_temp'] = pd.to_datetime(df['Date'])
        df.sort_values(by=['Date_temp', 'cfa_time'], inplace=True)
        df['queue'] = df.groupby('Date_temp').cumcount() + 1
        df['Cleaning_type_queue'] = df.groupby(['Date_temp', 'Cleaning_type']).cumcount() + 1
        df = df.drop(columns=['Date_temp'])
        df['Cleaning_time'] = df['Cleaning_time'].fillna(0)
        fig_burnup = px.scatter(df, x='cfa_time', y='queue', color='Date', hover_name='Room', hover_data=['Assigned To','Total_cleaning_time'])
        fig_burnup.update_layout(xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), paper_bgcolor='#FFFFFF')
        fig_burnup.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
        st.plotly_chart(fig_burnup, use_container_width=True)
    burnUpChartCompareDates2(df)





