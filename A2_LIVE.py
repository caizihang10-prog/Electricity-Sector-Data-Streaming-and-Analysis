from A2 import *

# Dash (dashboard)
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# Live Dashboard (Dash)
# ==========================================
def mqtt_subscribe():
    """Simple MQTT subscriber for debugging and bonus marks."""
    def on_connect(client, userdata, flags, rc):
        print(f" Connected to MQTT broker with result code {rc}")
        client.subscribe(f"{MQTT_TOPIC_PREFIX}/#")

    def on_message(client, userdata, msg):
        data = json.loads(msg.payload.decode())
        pd.DataFrame([data]).to_csv("mqtt_received.csv", mode="a", header=False)


    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print(f"🛰️ Subscribing to {MQTT_TOPIC_PREFIX}/# ...")
    client.loop_forever()

def run_dashboard():
    """Dashboard: Left = table, Right = interactive map with time series.
    Auto-refreshes from CSV every REFRESH_MS.
    """
    app = Dash(__name__)
    app.title = "NEM Live Facility Monitor"

    # ---------- Data loader ----------
    def load_latest():
        """Load the latest facility-level data from cached CSV."""
        if not os.path.exists(OUT_FACILITY_TS):
            return pd.DataFrame(columns=[
                "time","facility_code","facility_name","region","state","lat","lng",
                "fuel_tech","power_mw","emissions_tco2e_5m","price_aud_mwh","demand_mw"
            ])
        df = pd.read_csv(OUT_FACILITY_TS)
        df["timestamp"] = pd.to_datetime(df["time"])
        df = df.sort_values("timestamp", ascending=False)
        latest = df.groupby("facility_code").first().reset_index()
        return latest

    df_latest = load_latest()

    # ---------- Layout ----------
    app.layout = html.Div([
        html.H3("⚡ NEM Live Facility Monitor", style={"textAlign": "center"}),

        # Top section: dropdown + market summary
        html.Div([
            html.Div([
                html.Label("Filter by Fuel Type:"),
                dcc.Dropdown(
                    id='fuel-filter',
                    options=[],   # dynamically populated via callback
                    value=None,
                    placeholder="Select a fuel type",
                    style={'width': '300px'}
                )
            ], style={'display': 'inline-block', 'paddingRight': '20px'}),

            html.Div(id='market-summary', style={
                'display': 'inline-block',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'verticalAlign': 'top',
                'color': '#0074D9'
            })
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),

        # 🔄 Manual Refresh Button + Status Text
        html.Div([
            html.Button(
                '🔄 Refresh Now',
                id='manual-refresh',
                n_clicks=0,
                style={
                    'fontSize': '15px',
                    'padding': '8px 18px',
                    'backgroundColor': '#0074D9',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer'
                }
            ),
            html.Div(
                id='refresh-status',
                style={
                    'marginTop': '8px',
                    'color': '#0074D9',
                    'fontWeight': 'bold'
                }
            )
        ], style={'textAlign': 'center', 'marginBottom': '15px'}),

        # Middle section: left = table, right = map + time series
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='facility-table',
                    columns=[
                        {"name": "Facility Name", "id": "facility_name"},
                        {"name": "Region", "id": "region"},
                        {"name": "Fuel Tech", "id": "fuel_tech"},
                        {"name": "Power (MW)", "id": "power_mw", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Emissions (tCO2e/5m)", "id": "emissions_tco2e_5m", "type": "numeric", "format": {"specifier": ".2f"}},
                    ],
                    data=df_latest.to_dict("records"),
                    style_table={'overflowY': 'auto', 'height': '600px'},
                    style_cell={'textAlign': 'center', 'padding': '6px'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f2f2f2'},
                    page_size=20,
                )
            ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),

            html.Div([
                dcc.Graph(id='facility-map', style={"height": "500px"}),
                dcc.Graph(id='facility-timeseries', style={"height": "350px", "marginTop": "10px"})
            ], style={"width": "54%", "display": "inline-block", "paddingLeft": "10px"})
        ]),

        dcc.Interval(id='interval-update', interval=REFRESH_MS, n_intervals=0)
    ])

    # ---------- (0) Show refresh status ----------
    @app.callback(
        Output('refresh-status', 'children'),
        Input('manual-refresh', 'n_clicks'),
        prevent_initial_call=True
    )
    def show_refresh_status(n_clicks):
        ts = datetime.now().strftime('%H:%M:%S')
        return f" Refreshed at {ts}"

    # ---------- (1) Update dropdown and market summary ----------
    @app.callback(
        [Output('fuel-filter', 'options'),
        Output('market-summary', 'children')],
        [Input('interval-update', 'n_intervals'),
        Input('manual-refresh', 'n_clicks')]
    )
    def update_dropdown_and_summary(_n, _clicks):
        df = load_latest()
        if df.empty:
            return [], "No data available"

        fuel_options = [{'label': ft, 'value': ft} for ft in sorted(df['fuel_tech'].dropna().unique())]
        avg_price = df['price_aud_mwh'].mean(skipna=True)
        avg_demand = df['demand_mw'].mean(skipna=True)
        summary = f"Average Price: ${avg_price:.2f}/MWh | Avg Demand: {avg_demand:.1f} MW"
        return fuel_options, summary

    # ---------- (2) Update map ----------
    @app.callback(
        Output('facility-map', 'figure'),
        [Input('interval-update', 'n_intervals'),
        Input('manual-refresh', 'n_clicks'),   
        Input('fuel-filter', 'value')]
    )
    def update_map(_n, _clicks, selected_fuel):
        df = load_latest()
        if df.empty:
            fig = px.scatter_mapbox(lat=[-25], lon=[135], zoom=3, height=650, mapbox_style="carto-positron")
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            return fig

        if selected_fuel:
            df = df[df['fuel_tech'] == selected_fuel]
        df["point_size"] = df["power_mw"].apply(lambda x: max(x, 10))

        color_map = {
            "solar_utility": "#FFD700", "wind": "#32CD32", "hydro": "#1E90FF",
            "coal_black": "#4B4B4B", "gas_ccgt": "#FF8C00",
            "battery": "#8A2BE2", "bioenergy": "#9ACD32", "distillate": "#CD5C5C"
        }

        fig = px.scatter_mapbox(
            df,
            lat="lat", lon="lng", color="fuel_tech",
            hover_name="facility_name",
            hover_data={"region": True, "power_mw": True, "emissions_tco2e_5m": True, "fuel_tech": True},
            size="point_size", size_max=25, zoom=4.5, opacity=0.8,
            center={"lat": -25, "lon": 135}, mapbox_style="carto-positron",
            color_discrete_map=color_map
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=650, legend_title_text="Fuel Type")
        return fig

    # ---------- (3) Click map → show time series ----------
    @app.callback(
        Output("facility-timeseries", "figure"),
        [Input("facility-map", "clickData"),
        Input("manual-refresh", "n_clicks")]  
    )
    def show_timeseries(clickData, _clicks):
        if not os.path.exists(OUT_FACILITY_TS):
            return go.Figure()
        df = pd.read_csv(OUT_FACILITY_TS, parse_dates=["time"])

        if not clickData:
            fig = go.Figure()
            fig.update_layout(
                title="Click a facility to see its 7-day Power Trend",
                xaxis_title="Datetime", yaxis_title="Power (MW)",
                template="plotly_white"
            )
            return fig

        facility_name = clickData["points"][0]["hovertext"]
        df_fac = df[df["facility_name"] == facility_name]

        fig = go.Figure()
        if df_fac.empty:
            fig.update_layout(title=f"No data available for {facility_name}", template="plotly_white")
            return fig

        fig.add_trace(go.Scatter(x=df_fac["time"], y=df_fac["power_mw"],
                                mode="lines", name="Power (MW)",
                                line=dict(color="#1f77b4", width=2)))
        fig.update_layout(title=f"{facility_name} – 7-Day Power Trend",
                        xaxis_title="Datetime", yaxis_title="Power (MW)",
                        template="plotly_white")
        return fig


    # auto-open a browser tab shortly after server starts
    if AUTO_OPEN_BROWSER:
        def _open():
            time.sleep(1.0)
            webbrowser.open(f"http://{DASH_HOST}:{DASH_PORT}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"🚀 Dashboard on: http://{DASH_HOST}:{DASH_PORT}")
    app.run(host=DASH_HOST, port=DASH_PORT, debug=True)



# ---------- Dynamic Data Refresh Loop (COMMENTED OUT by default) ----------
def continuous_fetch(interval_sec=3600):
    """
    Re-fetch task12 data every interval to simulate near-real-time updates.
    WARNING: This consumes your daily 500-requests quota if set too small.
    """
    round_count = 1
    while True:
        print(f"\n==============================")
        print(f" FETCH ROUND {round_count} START ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"==============================")
        try:
            fetch_task12_data()
        except Exception as e:
            print(" Exception in fetch loop:", e)
        print(f" Waiting {interval_sec} seconds before next fetch…\n")
        time.sleep(interval_sec)
        round_count += 1


# ==========================================
# Main Entry
# ==========================================
# Uncomment to test MQTT once (publishes all facility records)
# mqtt_publish_once()
if __name__ == "__main__":
    # 1) Run Task 12 once (static batch for the assignment window)
    try:
        fetch_task12_data()
    except Exception as e:
        print(f" Task 12 fetch failed: {e}")
        # You may exit if initial data is mandatory
        # raise

    # 2) (Optional) Publish once to MQTT with the combined facility CSV
    # mqtt_publish_once(OUT_FACILITY_TS)

    # （optional）subscribe to MQTT for bonus if you want to run it together open another terminal and run
    # mqtt_subscribe()

    # 3) Start the live dashboard (auto-refresh reads the CSV written above)
    run_dashboard()

    # 4)Continuous modes (COMMENTED OUT by default)
    continuous_fetch(interval_sec=60)          # periodically re-fetch Task 12 data
    continuous_mqtt_publish(interval_sec=60)     # periodically re-publish MQTT

