import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np
import json
import datetime


######################################
# Change variables
path = "/home/woess/workspace/mnt/c/Users/woess/Desktop/ANT_coga_avg/coga_ant_avg_comb_2023-12-12_12-09.csv"

# peak_review = '/home/woess/current assignment/ANT_redone/ANT12/ANT12_new_peaks_long_2023-03-07_14-34.csv'

# .1-.250
# .250-.450
# .450-.8
con_ranges = {"Wc": [[0.05, 0.250], [0.200, 0.450], [0.450, 0.8]]}
con_peak_dropdown_options = {
    # 'Jc': ["N1", "P1", "N2", "P2"],
    # 'Ac': ["N1", "P1", "N2", "P2"],
    # 'Pc': ["N1", "P1", "N2", "P2"],
    "Wc": ["P1", "N1", "P2"]
}

plot_col_wrap = 3
channel_list_all = ["FZ", "CZ", "PZ", "P7", "P8"]
#
#######################################

app = Dash(__name__)
subject_df = pd.read_csv(path)
print(subject_df.head())
# Rname 'subject' column to 'Subject'
subject_df = subject_df.loc[subject_df["channel"].isin(channel_list_all)]
subject_df = subject_df[subject_df["subject"].str.contains("Wc") == True]
# Sort subject_df by Channel
# subject_df.sort_values(by=['Channel'], inplace=True)

# Create empty dataframe for peak values
peak_df = pd.DataFrame(
    ["x", "y", "peak", "file", "channel", "quality", "range", "review", "datetime"]
)

# peak_df = pd.read_csv(peak_review)
# #Rename 'subject' column to 'File'
# peak_df = peak_df.rename(columns={'Subject': 'File'})

# #Rename time_ms column to 'x'
# peak_df = peak_df.rename(columns={'time_ms': 'x'})

# #Rename value column to 'y'
# peak_df = peak_df.rename(columns={'value': 'y'})

# subject_df["peak"] = ""

# subject_df = subject_df[subject_df["Subject"].str.contains("p50_P50") == True]
# subject_df = subject_df.loc[subject_df["Subject"].str.slice(stop=5).astype(int) > 16184]

# ---------------------------Select subjects to be analyzed---------------------------
# time_diff_list = [16005,
#  16292]

# time_diff_list = [str(i) for i in time_diff_list]

# review_list = [
# 18227,
# 18014,
# 18033,
# 18036,
# 18042,
# 18043,
# 18047,
# 18048,
# 18049,
# 18068,
# 18079,
# 18109,
# 18117,
# 18118
# ]
# #Convert list to string
# review_list = [str(i) for i in review_list]

# # # #Find elements in review list that are not in the time_diff_list
# # # #------------------------------------------------------------------------------------
# # # time_diff_str = '|'.join(time_diff_list)
# review_str = '|'.join(review_list)

# subject_df = subject_df[(subject_df['Subject'].str.contains(time_diff_str) == False)
# subject_df = subject_df[(subject_df['Subject'].str.contains(review_str) == True)]
# ------------------------------------------------------------------------------------

_all = subject_df["channel"]
_all = np.unique(_all)

file_list = subject_df["subject"]
file_list = np.unique(file_list)

output_peak_columns = [
    "x",
    "y",
    "peak",
    "file",
    "channel",
    "quality",
    "range",
    "review",
    "datetime",
]

# subject_df_init = subject_df.loc[subject_df["Subject"] == "7001_err"]

fig = px.scatter()
fig.update_layout(clickmode="event+select")
fig.update_layout(dragmode="select")

fig.update_layout(autotypenumbers="convert types")
styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

app.layout = html.Div(
    [
        #
        dcc.Graph(
            id="graph-with-slider",
            figure=fig,
            config={"modeBarButtonsToRemove": ["lasso2d"]},
        ),
        dcc.Dropdown(
            options=_all, value=channel_list_all, id="channel-dropdown", multi=True
        ),
        dcc.Dropdown(options=file_list, value=file_list[0], id="file-dropdown"),
        html.Button(id="next-button", n_clicks=0, children="Next"),
        dcc.Dropdown(options=["1", "2", "3", "4"], value="1", id="quality-dropdown"),
        dcc.Dropdown(id="peak-number-dropdown"),
        html.Button(id="min-button", n_clicks=0, children="Find minimum"),
        html.Button(id="max-button", n_clicks=0, children="Find maximum"),
        html.Button(id="select-data-button", n_clicks=0, children="Select points"),
        html.Button(id="auto-button", n_clicks=0, children="auto"),
        dcc.Checklist(options=["Need Review"], id="review-box"),
        dcc.Checklist(options=["Select all channels"], id="select-all-box"),
        # dcc.Checklist(options=["Select all channels present"], id="select-present-box"),
        dcc.Checklist(options=["View Peaks"], id="view-peaks-box"),
        dcc.Slider(min=1, max=15, marks=None, value=0, id="zoom-slider"),
        html.Div(["Current Selection: "]),
        html.Div(id="current-selection"),
        html.Div(["Selected: "]),
        html.Div(id="selected-box"),
        html.Div(["Output: "]),
        html.Div(id="peak-box"),
        html.Div(
            dash_table.DataTable(
                id="table",
                columns=([{"id": name, "name": name} for name in output_peak_columns]),
                data=[
                    dict(Model=i, **{item: "" for item in output_peak_columns})
                    for i in range(1, 2000)
                ],
                editable=True,
                row_deletable=True,
                page_size=50,
                export_format="csv",
                export_headers="names",
            )
        ),
    ]
)


def _get_condition(file_name):
    file_name = file_name.split(".")[0]
    file_name = file_name.rsplit("_", 1)[1]
    return file_name


# Keeps dropdown of channels updated per file


@app.callback(
    Output("peak-number-dropdown", "options"),
    Output("peak-number-dropdown", "value"),
    Input("file-dropdown", "value"),
    Input("min-button", "n_clicks"),
    Input("max-button", "n_clicks"),
    State("peak-number-dropdown", "value"),
)
def set_peak_options(file_name, min, max, peak_num):
    callback_id = ctx.triggered_id if ctx.triggered_id else "Nothing-done"
    condition = _get_condition(file_name)
    if callback_id == "file-dropdown":
        return (
            con_peak_dropdown_options[condition],
            con_peak_dropdown_options[condition][0],
        )

    if (callback_id == "min-button") or (callback_id == "max-button"):
        peak_index = con_peak_dropdown_options[condition].index(peak_num)
        peak_list_length = len(con_peak_dropdown_options[condition])
        return (
            con_peak_dropdown_options[condition],
            con_peak_dropdown_options[condition][(peak_index + 1) % peak_list_length],
        )


# Keeps dropdown of channels updated per file
@app.callback(Output("channel-dropdown", "options"), Input("file-dropdown", "value"))
def set_channel_options(file_name):
    subject_df_new = subject_df.loc[(subject_df["subject"].isin([file_name]))]
    channel_list = subject_df_new["channel"]
    channel_list = np.unique(channel_list)
    return channel_list


@app.callback(
    Output("file-dropdown", "value"),
    Input("next-button", "n_clicks"),
    State("file-dropdown", "value"),
    State("file-dropdown", "options"),
)
def change_file(button, current_file, file_list):
    return file_list[file_list.index(current_file) + 1]


@app.callback(
    Output("graph-with-slider", "figure"),
    Input("zoom-slider", "value"),
    Input("channel-dropdown", "value"),
    Input("file-dropdown", "value"),
    Input("view-peaks-box", "value"),
    State("table", "data"),
    # State("current-selection","children"),
    # Input('select-all-button', "n_clicks")
    Input("peak-box", "children"),
)
def update_graph(y_min, channel, file_name, view_peak, table_data, peak_box):
    callback_id = ctx.triggered_id if ctx.triggered_id else "Nothing-done"
    df_table = pd.DataFrame.from_dict(table_data)
    print(subject_df["channel"].unique())
    subject_df_new = subject_df.loc[
        (subject_df["channel"].isin(channel))
        & (subject_df["subject"].isin([file_name]))
    ]

    peak_df_graph = df_table.loc[
        (df_table["channel"].isin(channel)) & (df_table["file"].isin([file_name]))
    ]

    condition = _get_condition(file_name)

    # Graph EEG
    fig = px.line(
        subject_df_new,
        x="time_ms",
        y="value",
        facet_col="channel",
        facet_col_wrap=plot_col_wrap,
        height=1000,
        width=2500,
        facet_row_spacing=0.02,
        markers=True,
    )
    fig.update_traces(marker_size=3, selected=dict(marker=dict(color="black")))
    # Add min and max points
    if view_peak == ["View Peaks"]:
        loop_channel = peak_df_graph["channel"].unique()
        # Add scatter to fig
        # col_comp = ((len(channel)+1) // plot_col_wrap) + 1
        print(f"sel chan: {channel}")
        for i, chan in enumerate(channel):
            row = i // plot_col_wrap + 1  # calculate current row
            col = i % plot_col_wrap + 1  # calculate current column
            print(chan, row, col)
            if chan in peak_df_graph["channel"].unique():
                scatter = go.Scatter(
                    x=peak_df_graph.loc[peak_df_graph["channel"] == chan]["x"],
                    y=peak_df_graph.loc[peak_df_graph["channel"] == chan]["y"],
                    mode="markers",
                    name="peak",
                    marker=dict(color="red", size=8),
                )
                # loop through rows and columns
                fig.add_trace(scatter, row=row, col=col)

    # # Add ranges for defining ranges of detection
    # for i, range in enumerate(con_ranges[condition]):
    #     if i % 2 == 0:
    #         color = "LightSalmon"
    #     else:
    #         color = "PaleTurquoise"
    #     fig.add_vrect(
    #         x0=range[0], x1=range[1],
    #         fillcolor=color, opacity=0.5,
    #         layer="below", line_width=0,
    #     )

    fig.update_layout(dragmode="select")
    # fig = go.Figure(fig)

    if callback_id == "zoom-slider":
        fig.update_yaxes(range=[5 * y_min, -5 * y_min])
    else:
        fig.update_yaxes(autorange="reversed")

    return fig


@app.callback(
    Output("current-selection", "children"),
    Input("graph-with-slider", "selectedData")
    # Add Input for channel options size to set
)
def define_selection_box(selectedData):
    if selectedData is None:
        raise PreventUpdate

    # selectedData = {"range": 0}
    # Converts selection dict to format "{'x': [x1, x2], 'y': [y1, y2]}""
    selection_box = selectedData["range"]

    for key in selection_box.copy().keys():
        if not (key == "x"):  # Condition to not delete x point
            selection_box[key[0]] = selection_box[key]
            del selection_box[key]

    return json.dumps(selection_box)  # , str(fig))


@app.callback(
    Output("selected-box", "children"),
    Input("min-button", "n_clicks"),
    Input("max-button", "n_clicks"),
    Input("select-data-button", "n_clicks"),
    State("select-all-box", "value"),
    State("graph-with-slider", "selectedData"),
    State("graph-with-slider", "figure"),
    State("file-dropdown", "value"),
    State("current-selection", "children")
    # State('multi-one-select', "value")
    # Add Input for channel options size to set
)
def select_points(
    min, max, select, select_all, selectedData, fig, file, current_select
):
    callback_id = ctx.triggered_id if ctx.triggered_id else "Nothing-done"

    # Converts all selections to form [{'x': x1, 'y': y1 , 'Channel': "Example_channel"},...]
    selected_channels = list(fig["data"])
    select_chan_order = []
    for diction in selected_channels:
        for key in diction.keys():
            if "channel=" in str(diction[key]):
                # Appends channel name to list
                start_index = diction[key].index("channel=") + len("channel=")
                end_index = diction[key].index("<br>")
                channel = diction[key][start_index:end_index]
                select_chan_order.append(channel)

    # Rename curveNumber to Channel
    if selectedData is None:
        raise PreventUpdate
    # Gets list of points as an array of dicts
    selected_points = list(selectedData["points"])
    for diction in selected_points:
        diction["curveNumber"] = select_chan_order[int(diction["curveNumber"])]
        diction["channel"] = diction["curveNumber"]
        del diction["curveNumber"]
        del diction["pointNumber"]
        del diction["pointIndex"]

    if select_all == ["Select all channels"]:
        current_select_box = json.loads(current_select)
        current_file_df = (
            subject_df.loc[
                (subject_df["subject"] == file)
                & (subject_df["time_ms"] >= current_select_box["x"][0])
                & (subject_df["time_ms"] <= current_select_box["x"][1])
            ][["time_ms", "value", "Channel"]]
            .rename(columns={"time_ms": "x", "value": "y"})
            .to_json(orient="records")
        )
        selected_points = (selected_points) + (json.loads(current_file_df))

    # Unselects points after max/min found
    if callback_id == "min-button" or callback_id == "max-button":
        selected_points = ""
    return json.dumps(selected_points)


@app.callback(
    Output("peak-box", "children"),
    Output("table", "data"),
    Input("min-button", "n_clicks"),
    Input("max-button", "n_clicks"),
    Input("auto-button", "n_clicks"),
    State("selected-box", "children"),
    State("file-dropdown", "value"),
    State("table", "data"),
    State("peak-number-dropdown", "value"),
    State("quality-dropdown", "value"),
    State("review-box", "value"),
    State("current-selection", "children"),
    State("channel-dropdown", "value"),
)
def find_peaks(
    min,
    max,
    auto,
    selected_points,
    subject_file,
    table_data,
    peak_num,
    quality_rating,
    check_val,
    current_select,
    channels,
):
    callback_id = ctx.triggered_id if ctx.triggered_id else "Nothing-done"
    temp_df = subject_df.loc[(subject_df["subject"] == subject_file)]
    temp_df = temp_df.loc[(temp_df["channel"].isin(channels))]

    # Rename column value to y
    temp_df = temp_df.rename(columns={"value": "y"})
    temp_df = temp_df.rename(columns={"time_ms": "x"})
    # Remove column Unnamed: 0
    temp_df = temp_df.drop(columns=["Unnamed: 0"])
    # Convert all numerical values to float
    temp_df = temp_df.astype(float, errors="ignore")
    # Convert Subject column to string
    temp_df = temp_df.astype({"subject": "string"})
    # Convert Channel column to string
    temp_df = temp_df.astype({"channel": "string"})

    # global output_peak_df
    if selected_points is None:
        raise PreventUpdate
    selected_points = json.loads(selected_points)
    df_selected_points = pd.DataFrame()
    df_table = pd.DataFrame.from_dict(table_data)
    for diction in list(selected_points):
        df_point = pd.DataFrame(diction, index=[0])
        df_selected_points = pd.concat(
            [df_selected_points, df_point], ignore_index=True
        )
    df_selected_points["file"] = subject_file
    condition = _get_condition(subject_file)
    print(condition)

    # Auto peak finder
    if callback_id == "auto-button":
        df_list = []
        for i, range in enumerate(con_ranges[condition]):
            auto_peak = temp_df.loc[
                (temp_df["x"] >= range[0]) & (temp_df["x"] <= range[1])
            ]

            if "N" in con_peak_dropdown_options[condition][i]:
                auto_peak = auto_peak.loc[
                    (auto_peak.groupby(["channel"])["y"].idxmin())
                ]
            if "P" in con_peak_dropdown_options[condition][i]:
                auto_peak = auto_peak.loc[
                    (auto_peak.groupby(["channel"])["y"].idxmax())
                ]

            auto_peak["range"] = str(range)
            auto_peak["peak"] = con_peak_dropdown_options[condition][i]
            auto_peak["file"] = subject_file
            print(auto_peak)
            df_list.append(auto_peak)
        df_selected_points_peak = pd.concat(df_list, ignore_index=True)

    if callback_id == "min-button":
        df_selected_points_peak = df_selected_points.loc[
            df_selected_points.groupby(["channel"])["y"].idxmin()
        ]

    if callback_id == "max-button":
        df_selected_points_peak = df_selected_points.loc[
            df_selected_points.groupby(["channel"])["y"].idxmax()
        ]

    # df_selected_points_peak["Review"] = ""
    if check_val == ["Need Review"]:
        df_selected_points_peak["review"] = check_val

    current_select_box = json.loads(current_select)
    if callback_id == "max-button" or callback_id == "min-button":
        df_selected_points_peak["range"] = str(current_select_box["x"])
        df_selected_points_peak["peak"] = str(peak_num)

    df_selected_points_peak["quality"] = str(quality_rating)
    df_selected_points_peak["datetime"] = str(datetime.datetime.now()) + " CST"

    df_table = pd.concat([df_selected_points_peak, df_table], ignore_index=True)

    return str(df_selected_points_peak), df_table.to_dict("records")


if __name__ == "__main__":
    app.run_server(debug=True, port=8052, threaded=True)  # debug = "True")
