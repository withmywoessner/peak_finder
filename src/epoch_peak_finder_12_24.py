import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
from dash.exceptions import PreventUpdate

import pandas as pd
app = Dash(__name__)
path = '/home/woess/mnt/c/Users/woess/Desktop/ANT18/ERN/ANT_18_comb_epoch_1_4.csv'
subject_df = pd.read_csv(path)
# subject_df["peak"] = ""
subject_df = subject_df.loc[subject_df["Subject"].str.contains("err")]
# lst = []
# rstr = '|'.join(lst)
# subject_df = subject_df[subject_df['Subject'].str.contains(rstr)]


channel_list = subject_df["Channel"]
channel_list = np.unique(channel_list)

file_list = subject_df["Subject"]
file_list = np.unique(file_list)

output_peak_columns= ["x", "y", "Peak", "File", "Channel", "Quality", "Review"]

#subject_df_init = subject_df.loc[subject_df["Subject"] == "7001_err"]

fig = px.scatter()
fig.update_layout(clickmode='event+select')
fig.update_layout(dragmode='select')

fig.update_layout(autotypenumbers='convert types')
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    #
    dcc.Graph(id='graph-with-slider',figure = fig, config={'modeBarButtonsToRemove': ["lasso2d"]}),
    dcc.Dropdown(options=channel_list, value=["FCz"], id='channel-dropdown',multi=True),
    dcc.Dropdown(options=file_list, value=file_list[0], id='file-dropdown'),
    html.Button(id="next-button", n_clicks=0, children="Next"),
    dcc.Dropdown(options=["1", "2", "3", "4"], value= "1", id='quality-dropdown'),
    dcc.Dropdown(options=["er0", "er1", "er2"], value= "er0", id='peak-number-dropdown'),
    html.Button(id="min-button", n_clicks=0, children="Find minimum"),
    html.Button(id="max-button", n_clicks=0, children="Find maximum"),
    html.Button(id='select-data-button', n_clicks=0, children='Select points'),
    dcc.Checklist(options=["Need Review"], id="review-box"),
    dcc.Checklist(options=["Select all channels"], id="select-all-box"),

    dcc.Slider(
        min= 1,
        max=15,
        marks=None,
        value=0,
        id='zoom-slider'
    ),
      html.Div([
        "Current Selection: "
    ]),
    html.Div(id='current-selection'),
      html.Div([
        "Selected: "
    ]),
    html.Div(id='selected-box'),
      html.Div([
        "Output: "
    ]),
    html.Div(id='peak-box'),
    html.Div(dash_table.DataTable(
                id='table',
                columns=(
                
                [{'id': name, 'name': name} for name in output_peak_columns]
                ),
                data=[
                    dict(Model=i, **{item: "" for item in output_peak_columns})
                    for i in range(1, 2000)
                ],
                editable=True,
                page_size=50,
                export_format='csv',
                export_headers='names'
            )
    )
])

#Keeps dropdown of channels updated per file
@app.callback(
    Output('channel-dropdown', 'options'),
    Input('file-dropdown', 'value')
)
def set_channel_options(file_name):
    subject_df_new = subject_df.loc[(subject_df['Subject'].isin([file_name]))]
    channel_list = subject_df_new["Channel"]
    channel_list = np.unique(channel_list)
    return channel_list
@app.callback(
    Output('file-dropdown', 'value'),
    Input('next-button', 'n_clicks'),
    State('file-dropdown', 'value'),
    State('file-dropdown', 'options'),
)
def change_file(button, current_file, file_list):
    return file_list[file_list.index(current_file)+1]
    


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('zoom-slider', 'value'),
    Input("channel-dropdown","value"),
    Input("file-dropdown","value"),
    State("current-selection","children"),
    #Input('select-all-button', "n_clicks")
    # Input('min-button', 'n_clicks'),
    # Input('max-button', 'n_clicks')
)
def update_graph(y_min, channel, file_name, current_selection,):
    callback_id = ctx.triggered_id if ctx.triggered_id else "Nothing-done"
    subject_df_new = (
        subject_df
        .loc[(subject_df['Channel'].isin(channel)) & 
        (subject_df['Subject'].isin([file_name]))])

    fig = px.line(
    subject_df_new, x="time_ms", y="value",
    facet_col="Channel", facet_col_wrap=3, height=500, width=1000, 
    facet_row_spacing=0.02,  markers=True
    )
    fig.update_layout(dragmode='select')
    fig = go.Figure(fig)
    fig.update_traces(marker_size=3, selected=dict(marker=dict(color="black")))

    #Old select all button that selects currently visible channels
    # if(callback_id == "select-all-button"):
    #      current_selection = json.loads(current_selection)
    #      fig.add_selection(x0= int(current_selection["x"][0]), y0=int(current_selection["y"][0]), 
    #      x1=int(current_selection["x"][1]), y1=int(current_selection["y"][1]), row="all", col="all")
    #      fig.update_yaxes(autorange="reversed") 
    #      return fig
    
 

    if(callback_id == "zoom-slider"):
        fig.update_yaxes(range=[5*y_min,-5*y_min])
    else:
         fig.update_yaxes(autorange="reversed") 
    
    return fig
@app.callback(
    Output('current-selection', 'children'),
    Input('graph-with-slider', 'selectedData'),
    State('graph-with-slider', 'figure'),
    #Add Input for channel options size to set 
)
def define_selection_box(selectedData, fig):
    if selectedData is None:
        raise PreventUpdate


    #Converts selection dict to format "{'x': [x1, x2], 'y': [y1, y2]}""
    selection_box = (selectedData["range"])
    
    for key in selection_box.copy().keys():
        if not(key == "x"): #Condition to not delete x point
            selection_box[key[0]] = selection_box[key]
            del selection_box[key]
    
    return (json.dumps(selection_box))#, str(fig))

@app.callback(
    Output('selected-box', 'children'),
    Input('min-button', 'n_clicks'),
    Input('max-button', 'n_clicks'),
    Input('select-data-button', 'n_clicks'),
    State('select-all-box', "value"),
    State('graph-with-slider', 'selectedData'),
    State('graph-with-slider', 'figure'),
    State("file-dropdown","value"),
    State('current-selection', 'children')
    #State('multi-one-select', "value")
    #Add Input for channel options size to set 
)
def select_points(min, max, select, select_all, selectedData, fig, file, current_select):
    callback_id = ctx.triggered_id if ctx.triggered_id else "Nothing-done"

    #Converts all selections to form [{'x': x1, 'y': y1 , 'Channel': "Example_channel"},...]
    selected_channels = list(fig["data"])
    select_chan_order =[]
    for diction in selected_channels:
        for key in diction.keys():
            if "Channel=" in str(diction[key]):
                
                #Appends channel name to list
                start_index = (diction[key].index("Channel=") + len("Channel="))
                end_index = diction[key].index("<br>")
                channel=diction[key][start_index:end_index]
                select_chan_order.append(channel)

    #Rename curveNumber to Channel
    if selectedData is None:
        raise PreventUpdate
    selected_points = list(selectedData["points"]) #Gets list of points as an array of dicts 
    for diction in selected_points:
        diction["curveNumber"] = select_chan_order[int(diction["curveNumber"])]
        diction["Channel"] = diction["curveNumber"]
        del diction["curveNumber"]
        del diction["pointNumber"]
        del diction["pointIndex"]
    
    if(select_all == ["Select all channels"]):
        
        current_select_box = json.loads(current_select)
        current_file_df = (
            subject_df
            .loc[(subject_df["Subject"] == file) & 
            (subject_df["time_ms"] >= current_select_box["x"][0]) &
            (subject_df["time_ms"] <= current_select_box["x"][1])]
            [["time_ms", "value", "Channel"]]
            .rename(columns={'time_ms': 'x', 'value': 'y'})
            .to_json(orient = 'records')
            )
        selected_points = (selected_points) + (json.loads(current_file_df))

    #Unselects points after max/min found 
    if (callback_id == "min-button" or callback_id == "max-button"):
        selected_points = ""

    return (json.dumps(selected_points))


@app.callback(
    Output('peak-box', 'children'),
    Output("table", "data"),
    Output("peak-number-dropdown", "value"),
    Input('min-button', 'n_clicks'),
    Input('max-button', 'n_clicks'),
    State('selected-box', 'children'),
    State("file-dropdown","value"),
    State("table", "data"),
    State("peak-number-dropdown", "value"),
    State("quality-dropdown", "value"),
    State("review-box", "value")
)
def find_peaks(min, max, selected_points, subject_file, table_data, peak_num, quality_rating, check_val):
    callback_id = ctx.triggered_id if ctx.triggered_id else "Nothing-done"
    #global output_peak_df
    if selected_points is None:
            raise PreventUpdate
    selected_points = json.loads(selected_points)
    df_selected_points = pd.DataFrame()
    df_table = pd.DataFrame.from_dict(table_data)
    for diction in list(selected_points):
        df_point = pd.DataFrame(diction, index=[0])
        df_selected_points = pd.concat([df_selected_points,df_point], ignore_index=True)
    df_selected_points["File"] = subject_file
    


    if(callback_id == "min-button"):
        df_selected_points_peak = df_selected_points.loc[df_selected_points.groupby(["Channel"])["y"].idxmin()]
    
    if(callback_id == "max-button"):
        df_selected_points_peak = df_selected_points.loc[df_selected_points.groupby(["Channel"])["y"].idxmax()]
    
    df_selected_points_peak["Review"] = ""
    if (check_val == ["Need Review"]):
        df_selected_points_peak["Review"] = check_val
    
    df_selected_points_peak["Peak"] = str(peak_num)
    df_selected_points_peak["Quality"] = str(quality_rating)
    df_table = pd.concat([df_selected_points_peak, df_table], ignore_index=True)
    
    #Changes peak num after max or min is found
    if(peak_num == "er0"):
        peak_num = "er1"
    else:
        if(peak_num == "er1"):
            peak_num = "er2"
        else:
            if(peak_num == "er2"):
                peak_num = "er0"
    
    # #Resets peak_num to 0 if file is changed
    # if (callback_id == "file-dropdown"):
    #     peak_num = "er0"
        
    
    return str(df_selected_points_peak), df_table.to_dict('records'), (peak_num)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, threaded=True)#debug = "True")

