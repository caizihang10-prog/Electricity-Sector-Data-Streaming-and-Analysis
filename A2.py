# ==========================================
# README
# ⚡ OpenElectricity A2 — Data Retrieval + (optional) MQTT + Live Dashboard
#First, set up your environment:
# Requirements:
# pip install -r requirements.txt
#########if you run into issues, try installing specific versions:#########
#like this #pip install dash==2.17.1 or pip install paho-mqtt==1.6.1
#some time numpty versions may cause issues. 
#after that, run this file first to fetch data:
#than run A2_live.py to launch the dashboard and MQTT publisher.
#HERE have another api key：oe_3ZSoFiK8jS7UByBSmd57f11o
# ==========================================

# One-file solution: Data Retrieval + (optional) MQTT + Live Dashboard
# - English column names in the dashboard
# - Dynamic loops are INCLUDED but COMMENTED OUT by default
# Run:  python A2.py than  python A2_live.py
#if you want to see more stations change QUERY_LIMIT to higher values like 10 or 20
#subscribe to MQTT for debugging/bonus if you want to run it together open another terminal and run
import os
import time
import threading
import webbrowser
import json
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

# OpenElectricity SDK
from openelectricity import OEClient
from openelectricity.types import DataMetric, MarketMetric, UnitStatusType

# MQTT (Task 3)
try:
    import paho.mqtt.client as mqtt
    print("✅ paho-mqtt imported successfully:", mqtt)
except ImportError as e:
    mqtt = None


OE_API_KEY = "oe_3ZetnYspTgxHDvkuu6TrANvc"

# ----------------------------- Configurable Parameters -----------------------------
# Query scope
QUERY_LIMIT = int(os.getenv("OE_QUERY_LIMIT", "5"))     # changable limit for facilities to query(1 for testing, higher for full)
INTERVAL = "5m"                                         # 5-minute interval
NETWORK = "NEM"                                         # NEM network
# Output file names
OUT_UNITS_TS = "nem_unit_5m_last7days.csv"
OUT_FACILITY_STATIC = "nem_facilities_static.csv"
OUT_FACILITY_TS = "nem_facility_5m_with_market_last7days.csv"        # dashboard + MQTT read this
OUT_FACILITY_TS_SIMPLE = "nem_facility_simple_5m_last7days.csv"
OUT_MARKET_RAW = "nem_market_region_5m_last7days.csv"
# Assignment window: 1–7 October 2025
ASSIGNMENT_START = datetime(2025, 10, 1, 0, 0, 0)
ASSIGNMENT_END   = datetime(2025, 10, 8, 0, 0, 0)


# MQTT (Task 3)
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "openelectricity/facilities"

# Dashboard
DASH_HOST = "127.0.0.1"
DASH_PORT = 8050
AUTO_OPEN_BROWSER = True
REFRESH_MS = 60 * 1000  # dashboard auto-refresh interval
# -----------------------------------------------------------------------------------


# ==========================================
# Helpers
# ==========================================
def now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def region_to_state(region: Optional[str]) -> Optional[str]:
    """Map NEM region to state code."""
    if not region:
        return None
    region = region.upper()
    table = {
        "NSW1": "NSW",
        "QLD1": "QLD",
        "VIC1": "VIC",
        "SA1":  "SA",
        "TAS1": "TAS",
    }
    return table.get(region)


# ==========================================
# Task 12 — Data Retrieval (REST via SDK)
# ==========================================
def fetch_task12_data():
    """
    Fetches:
      - per-facility power (5m)
      - per-facility CO2 emissions (5m)
      - optional per-region price & demand (5m)
    For one week in October 2025 (per assignment), then materializes CSV outputs.
    """
    print(f"[{now_hms()}] Script starting (Task 12)…")
    api_key = OE_API_KEY  # use direct constant
    client = OEClient(api_key=api_key)
    print("✅ API key loaded from code")

    # Fixed assignment window
    start_date = ASSIGNMENT_START
    end_date = ASSIGNMENT_END
    print("📅 Time window (assignment):", start_date, "→", end_date)

    api_calls = 0

    # Step 1b: Facilities (operating)
    print("Step 1b: Fetching OPERATING facilities in NEM…")
    facilities_resp = client.get_facilities(
        network_id=[NETWORK],
        status_id=[UnitStatusType.OPERATING],
    )
    api_calls += 1
    facilities = facilities_resp.data or []
    print(f"   ✅ Found {len(facilities)} facilities.")

    if not facilities:
        raise RuntimeError("No facilities returned. Abort.")

    # Collect static + unit metadata
    units_meta_rows: List[Dict[str, Any]] = []
    fac_static_rows: List[Dict[str, Any]] = []
    all_facility_codes: List[str] = []

    for fac in facilities:
        fac_code = getattr(fac, "code", None)
        fac_name = getattr(fac, "name", None)
        region = getattr(fac, "network_region", None)

        # Geo
        lat = lng = None
        loc = getattr(fac, "location", None)
        if loc is not None:
            lat = getattr(loc, "lat", None)
            lng = getattr(loc, "lng", None)

        # Determine main fuel by capacity sum
        units = getattr(fac, "units", None) or []
        fuel_cap: Dict[str, float] = {}
        unit_count = 0
        for u in units:
            unit_count += 1
            unit_code = getattr(u, "code", None)
            fueltech = getattr(u, "fueltech_id", None)
            fuel_str = getattr(fueltech, "value", str(fueltech)) if fueltech is not None else None
            cap = getattr(u, "capacity_registered", None)
            cap = float(cap) if cap is not None else 0.0
            if fuel_str:
                fuel_cap[fuel_str] = fuel_cap.get(fuel_str, 0.0) + cap

            units_meta_rows.append({
                "unit_code": unit_code,
                "facility_code": fac_code,
                "facility_name": fac_name,
                "region": region,
                "fuel_type": fuel_str,
                "lat": lat,
                "lng": lng,
            })

        if fuel_cap:
            main_fuel = max(fuel_cap.items(), key=lambda kv: kv[1])[0]
        else:
            fuels = sorted(set(
                r["fuel_type"] for r in units_meta_rows
                if r["facility_code"] == fac_code and r["fuel_type"]
            ))
            main_fuel = ",".join(fuels) if fuels else None

        fac_static_rows.append({
            "facility_code": fac_code,
            "facility_name": fac_name,
            "region": region,
            "state": region_to_state(region),
            "lat": lat,
            "lng": lng,
            "fuel_tech": main_fuel,
            "unit_count": unit_count,
        })
        all_facility_codes.append(fac_code)
    
    # Save facility static
    df_facilities_static = pd.DataFrame(fac_static_rows).drop_duplicates(subset=["facility_code"]).reset_index(drop=True)
    df_facilities_static.to_csv(OUT_FACILITY_STATIC, index=False)
    print(f"   ✅ Facilities static cached → {OUT_FACILITY_STATIC}")

    # De-dup unit meta
    df_units_meta = (
        pd.DataFrame(units_meta_rows)
        .dropna(subset=["unit_code", "facility_code"])
        .drop_duplicates(subset=["unit_code", "facility_code"])
        .reset_index(drop=True)
    )
    print(f"   ℹ️ Unit metadata rows: {len(df_units_meta)}")

    # Step 1c: facility data (power + emissions)
    print(f"Step 1c: Fetching {INTERVAL} power/emissions…")

    timeseries_rows: List[Dict[str, Any]] = []
    valid_facilities = []
    target_count = 5
    success_count = 0

    for fac in facilities:
        if success_count >= target_count:
            break

        fac_code = getattr(fac, "code", None)
        print(f"   Fetching {fac_code} ({success_count + 1}/{target_count}) …")

        try:
            resp = client.get_facility_data(
                network_code=NETWORK,
                facility_code=fac_code,
                metrics=[DataMetric.POWER, DataMetric.EMISSIONS],
                interval=INTERVAL,
                date_start=start_date,
                date_end=end_date,
            )
            api_calls += 1

            # ---- Parse and collect data ----
            for ts in resp.data or []:
                metric = getattr(ts, "metric", None)
                metric_str = getattr(metric, "value", str(metric)) if metric else None
                if not metric_str:
                    continue
                for result in ts.results or []:
                    unit_code = getattr(getattr(result, "columns", None), "unit_code", None)
                    for point in result.data or []:
                        ts_ = getattr(point, "timestamp", None)
                        val_ = getattr(point, "value", None)
                        if ts_ is None or val_ is None:
                            root = getattr(point, "root", None)
                            if root and len(root) >= 2:
                                ts_, val_ = root[0], root[1]
                        if ts_ is None or val_ is None:
                            continue
                        timeseries_rows.append({
                            "time": pd.to_datetime(ts_),
                            "facility_code": fac_code,
                            "unit_code": unit_code,
                            "metric": metric_str,
                            "value": val_,
                        })

            print(f"   ✅ Done: {fac_code}")
            valid_facilities.append(fac_code)
            success_count += 1
            time.sleep(0.25)

        except Exception as e:
            print(f"   ⚠️ Skipping {fac_code} due to API error: {e}")
            continue  # try next facility

    print(f"✅ Successfully fetched {success_count}/{target_count} facilities.")
    print(f"   Included: {', '.join(valid_facilities)}")


    # Step 1d: market price/demand (optional)
    print("Step 1d: Fetching market price/demand…")
    market_rows: List[Dict[str, Any]] = []
    try:
        market_resp = client.get_market(
            network_code=NETWORK,
            metrics=[MarketMetric.PRICE, MarketMetric.DEMAND],
            interval=INTERVAL,
            date_start=start_date,
            date_end=end_date,
            primary_grouping="network_region",
        )
        api_calls += 1
        for ts in market_resp.data or []:
            metric = getattr(ts, "metric", None)
            metric_str = getattr(metric, "value", str(metric)) if metric else None
            if not metric_str:
                continue
            for result in ts.results or []:
                # result.name like "metric_region_VIC1" → grab region suffix
                region = "Unknown"
                name = getattr(result, "name", "")
                if name and "_" in name:
                    region = name.split("_")[-1]
                for point in result.data or []:
                    ts_ = getattr(point, "timestamp", None)
                    val_ = getattr(point, "value", None)
                    if ts_ is None or val_ is None:
                        root = getattr(point, "root", None)
                        if root and len(root) >= 2:
                            ts_, val_ = root[0], root[1]
                    if ts_ is None or val_ is None:
                        continue
                    market_rows.append({"time": pd.to_datetime(ts_), "region": region, "metric": metric_str, "value": val_})
        print(f"   ℹ️ Market rows: {len(market_rows)}")
    except Exception as e:
        print(f"   ⚠️ Market fetch skipped/failed: {e}")

    print(f"[{now_hms()}] Fetch complete. (approx API calls: {api_calls})")

    # ---------------- Materialization ----------------
    # A) Unit-level timeseries + metadata
    print("Step 2a: Building unit-level timeseries…")
    df_units_final = pd.DataFrame()
    if timeseries_rows:
        df_raw = pd.DataFrame(timeseries_rows)
        df_pivot = (
            df_raw.pivot_table(index=["time", "facility_code", "unit_code"], columns="metric", values="value")
            .reset_index()
            .rename(columns={"power": "power_mw", "emissions": "emissions_tco2e_5m"})
        )
        for c in ["power_mw", "emissions_tco2e_5m"]:
            if c not in df_pivot.columns:
                df_pivot[c] = 0.0
        df_pivot[["power_mw", "emissions_tco2e_5m"]] = df_pivot[["power_mw", "emissions_tco2e_5m"]].fillna(0.0)

        cols_keep = ["unit_code", "facility_code", "facility_name", "region", "lat", "lng", "fuel_type"]
        df_units_meta_small = df_units_meta[cols_keep].drop_duplicates(subset=["unit_code", "facility_code"])
        df_units_final = (
            pd.merge(df_pivot, df_units_meta_small, on=["unit_code", "facility_code"], how="left", validate="m:1")
            .sort_values(["time", "facility_code", "unit_code"])
            .reset_index(drop=True)
        )
        df_units_final.to_csv(OUT_UNITS_TS, index=False)
        print(f"   ✅ Unit-level cached → {OUT_UNITS_TS}")
    else:
        print("   ⚠️ No unit-level timeseries built (no raw rows).")

    # B) Facility-level aggregation + market join
    print("Step 2b: Aggregating to facility-level & joining market…")
    if df_units_final.empty:
        print("   ⚠️ Unit-level empty; cannot aggregate facility-level.")
        df_facility_market = pd.DataFrame()
    else:
        df_units_final["state"] = df_units_final["region"].map(region_to_state)
        group_keys = ["time", "facility_code", "facility_name", "region", "state", "lat", "lng"]
        df_facility = df_units_final.groupby(group_keys, as_index=False)[["power_mw", "emissions_tco2e_5m"]].sum()

        if market_rows:
            df_market_raw = pd.DataFrame(market_rows)
            df_market = (
                df_market_raw.pivot_table(index=["time", "region"], columns="metric", values="value")
                .reset_index()
                .rename(columns={"price": "price_aud_mwh", "demand": "demand_mw"})
            )
            for c in ["price_aud_mwh", "demand_mw"]:
                if c not in df_market.columns:
                    df_market[c] = 0.0
            df_market[["price_aud_mwh", "demand_mw"]] = df_market[["price_aud_mwh", "demand_mw"]].fillna(0.0)

            df_facility_market = pd.merge(
                df_facility,
                df_market[["time", "region", "price_aud_mwh", "demand_mw"]],
                on=["time", "region"],
                how="left",
                validate="m:1",
            )
        else:
            df_facility_market = df_facility.copy()
            df_facility_market["price_aud_mwh"] = pd.NA
            df_facility_market["demand_mw"] = pd.NA

        # add fuel_tech from static
        df_facility_market = df_facility_market.merge(
            df_facilities_static[["facility_code", "fuel_tech"]],
            on="facility_code",
            how="left",
            validate="m:1",
        )

        # reorder columns
        df_facility_market = df_facility_market[
            [
                "time",
                "facility_code",
                "facility_name",
                "region",
                "state",
                "lat",
                "lng",
                "fuel_tech",
                "power_mw",
                "emissions_tco2e_5m",
                "price_aud_mwh",
                "demand_mw",
            ]
        ].sort_values(["time", "facility_code"])

      
        df_facility_market = df_facility_market[df_facility_market["power_mw"] > 0].copy()
        df_facility_market.to_csv(OUT_FACILITY_TS, index=False)
        print(f"   ✅ Facility-level (with market) cached → {OUT_FACILITY_TS}")


        # Simple 4-col file
        df_facility_simple = (
            df_facility_market[["time", "facility_name", "power_mw", "emissions_tco2e_5m"]]
            .rename(columns={"time": "timestamp", "power_mw": "power_generated_mw"})
            .sort_values(["timestamp", "facility_name"])
        )
        df_facility_simple.to_csv(OUT_FACILITY_TS_SIMPLE, index=False)
        print(f"   ✅ Facility SIMPLE cached → {OUT_FACILITY_TS_SIMPLE}")

    # C) market-only CSV
    if market_rows:
        pd.DataFrame(market_rows).to_csv(OUT_MARKET_RAW, index=False)
        print(f"   ✅ Market raw cached → {OUT_MARKET_RAW}")

    print(f"\n✅ [{now_hms()}] All Task 12 outputs ready.")

if __name__ == "__main__":
    fetch_task12_data()


# ==========================================
# Task 3 — MQTT Publisher
# ==========================================
def mqtt_publish_once(combined_csv: str = OUT_FACILITY_TS):
    """
    Publish facility-level combined dataset to MQTT (HiveMQ public broker).
    Topic: openelectricity/facilities/<region>/<facility_code>
    """
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    df = pd.read_csv(combined_csv)
    delays = []  # record real delays for verification

    print(f"✅ Connected to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
    for i, row in df.iterrows():
        start_time = time.time()

        try:
            payload = {
                "timestamp": str(row["time"]),
                "facility_code": str(row["facility_code"]),
                "facility_name": str(row["facility_name"]),
                "region": str(row["region"]),
                "state": str(row["state"]),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "fuel_tech": str(row["fuel_tech"]),
                "power_mw": float(row.get("power_mw", 0)),
                "emissions_tonne": float(row.get("emissions_tco2e_5m", 0)),
                "price": float(row.get("price_aud_mwh", 0)),
                "demand": float(row.get("demand_mw", 0)),
            }

            topic = f"{MQTT_TOPIC_PREFIX}/{payload['region']}/{payload['facility_code']}"
            client.publish(topic, json.dumps(payload))

            # Print with precise send timestamp
            print(f" {i+1:>4}/{len(df)} | Sent at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
                  f"| Record ts: {payload['timestamp']} | {payload['facility_name']} | {payload['power_mw']} MW")

        except Exception as e:
            print(f"⚠️ Error publishing record {i}: {e}")

        # Maintain exact 0.1s interval (including processing time)
        elapsed = time.time() - start_time
        remaining = 0.1 - elapsed
        if remaining > 0:
            time.sleep(remaining)
        delays.append(time.time() - start_time)

    # Show average delay for verification
    avg_delay = sum(delays) / len(delays) if delays else 0
    print(f"✅ Average publish delay: {avg_delay:.3f}s over {len(delays)} messages.")

    client.loop_stop()
    client.disconnect()


# ---------- Dynamic MQTT loop (COMMENTED OUT by default) ----------
def continuous_mqtt_publish(interval_sec=60, csv_path: str = OUT_FACILITY_TS):
    """Continuously publish CSV data to MQTT every interval."""
    round_count = 1
    while True:
        print(f"\n==============================")
        print(f" MQTT ROUND {round_count} START ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"==============================")
        try:
            mqtt_publish_once(csv_path)
        except Exception as e:
            print(" Exception in MQTT loop:", e)
            traceback.print_exc()
        print(f" Waiting {interval_sec} seconds before next publish…\n")
        time.sleep(interval_sec)
        round_count += 1


# # ==========================================
# # Live Dashboard (Dash)
# # ==========================================
# def mqtt_subscribe():
#     """Simple MQTT subscriber for debugging and bonus marks."""
#     def on_connect(client, userdata, flags, rc):
#         print(f" Connected to MQTT broker with result code {rc}")
#         client.subscribe(f"{MQTT_TOPIC_PREFIX}/#")

#     def on_message(client, userdata, msg):
#         data = json.loads(msg.payload.decode())
#         pd.DataFrame([data]).to_csv("mqtt_received.csv", mode="a", header=False)


#     client = mqtt.Client()
#     client.on_connect = on_connect
#     client.on_message = on_message
#     client.connect(MQTT_BROKER, MQTT_PORT, 60)
#     print(f"🛰️ Subscribing to {MQTT_TOPIC_PREFIX}/# ...")
#     client.loop_forever()

# def run_dashboard():
#     """Dashboard: Left = table, Right = interactive map with time series.
#     Auto-refreshes from CSV every REFRESH_MS.
#     """
#     app = Dash(__name__)
#     app.title = "NEM Live Facility Monitor"

#     # ---------- Data loader ----------
#     def load_latest():
#         """Load the latest facility-level data from cached CSV."""
#         if not os.path.exists(OUT_FACILITY_TS):
#             return pd.DataFrame(columns=[
#                 "time","facility_code","facility_name","region","state","lat","lng",
#                 "fuel_tech","power_mw","emissions_tco2e_5m","price_aud_mwh","demand_mw"
#             ])
#         df = pd.read_csv(OUT_FACILITY_TS)
#         df["timestamp"] = pd.to_datetime(df["time"])
#         df = df.sort_values("timestamp", ascending=False)
#         latest = df.groupby("facility_code").first().reset_index()
#         return latest

#     df_latest = load_latest()

#     # ---------- Layout ----------
#     app.layout = html.Div([
#         html.H3("⚡ NEM Live Facility Monitor", style={"textAlign": "center"}),

#         # Top section: dropdown + market summary
#         html.Div([
#             html.Div([
#                 html.Label("Filter by Fuel Type:"),
#                 dcc.Dropdown(
#                     id='fuel-filter',
#                     options=[],   # dynamically populated via callback
#                     value=None,
#                     placeholder="Select a fuel type",
#                     style={'width': '300px'}
#                 )
#             ], style={'display': 'inline-block', 'paddingRight': '20px'}),

#             html.Div(id='market-summary', style={
#                 'display': 'inline-block',
#                 'fontWeight': 'bold',
#                 'fontSize': '16px',
#                 'verticalAlign': 'top',
#                 'color': '#0074D9'
#             })
#         ], style={'textAlign': 'center', 'marginBottom': '10px'}),

#         # 🔄 Manual Refresh Button + Status Text
#         html.Div([
#             html.Button(
#                 '🔄 Refresh Now',
#                 id='manual-refresh',
#                 n_clicks=0,
#                 style={
#                     'fontSize': '15px',
#                     'padding': '8px 18px',
#                     'backgroundColor': '#0074D9',
#                     'color': 'white',
#                     'border': 'none',
#                     'borderRadius': '5px',
#                     'cursor': 'pointer'
#                 }
#             ),
#             html.Div(
#                 id='refresh-status',
#                 style={
#                     'marginTop': '8px',
#                     'color': '#0074D9',
#                     'fontWeight': 'bold'
#                 }
#             )
#         ], style={'textAlign': 'center', 'marginBottom': '15px'}),

#         # Middle section: left = table, right = map + time series
#         html.Div([
#             html.Div([
#                 dash_table.DataTable(
#                     id='facility-table',
#                     columns=[
#                         {"name": "Facility Name", "id": "facility_name"},
#                         {"name": "Region", "id": "region"},
#                         {"name": "Fuel Tech", "id": "fuel_tech"},
#                         {"name": "Power (MW)", "id": "power_mw", "type": "numeric", "format": {"specifier": ".2f"}},
#                         {"name": "Emissions (tCO2e/5m)", "id": "emissions_tco2e_5m", "type": "numeric", "format": {"specifier": ".2f"}},
#                     ],
#                     data=df_latest.to_dict("records"),
#                     style_table={'overflowY': 'auto', 'height': '600px'},
#                     style_cell={'textAlign': 'center', 'padding': '6px'},
#                     style_header={'fontWeight': 'bold', 'backgroundColor': '#f2f2f2'},
#                     page_size=20,
#                 )
#             ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),

#             html.Div([
#                 dcc.Graph(id='facility-map', style={"height": "500px"}),
#                 dcc.Graph(id='facility-timeseries', style={"height": "350px", "marginTop": "10px"})
#             ], style={"width": "54%", "display": "inline-block", "paddingLeft": "10px"})
#         ]),

#         dcc.Interval(id='interval-update', interval=REFRESH_MS, n_intervals=0)
#     ])

#     # ---------- (0) Show refresh status ----------
#     @app.callback(
#         Output('refresh-status', 'children'),
#         Input('manual-refresh', 'n_clicks'),
#         prevent_initial_call=True
#     )
#     def show_refresh_status(n_clicks):
#         ts = datetime.now().strftime('%H:%M:%S')
#         return f"✅ Refreshed at {ts}"

#     # ---------- (1) Update dropdown and market summary ----------
#     @app.callback(
#         [Output('fuel-filter', 'options'),
#         Output('market-summary', 'children')],
#         [Input('interval-update', 'n_intervals'),
#         Input('manual-refresh', 'n_clicks')]
#     )
#     def update_dropdown_and_summary(_n, _clicks):
#         df = load_latest()
#         if df.empty:
#             return [], "No data available"

#         fuel_options = [{'label': ft, 'value': ft} for ft in sorted(df['fuel_tech'].dropna().unique())]
#         avg_price = df['price_aud_mwh'].mean(skipna=True)
#         avg_demand = df['demand_mw'].mean(skipna=True)
#         summary = f"Average Price: ${avg_price:.2f}/MWh | Avg Demand: {avg_demand:.1f} MW"
#         return fuel_options, summary

#     # ---------- (2) Update map ----------
#     @app.callback(
#         Output('facility-map', 'figure'),
#         [Input('interval-update', 'n_intervals'),
#         Input('manual-refresh', 'n_clicks'),   
#         Input('fuel-filter', 'value')]
#     )
#     def update_map(_n, _clicks, selected_fuel):
#         df = load_latest()
#         if df.empty:
#             fig = px.scatter_mapbox(lat=[-25], lon=[135], zoom=3, height=650, mapbox_style="carto-positron")
#             fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
#             return fig

#         if selected_fuel:
#             df = df[df['fuel_tech'] == selected_fuel]
#         df["point_size"] = df["power_mw"].apply(lambda x: max(x, 10))

#         color_map = {
#             "solar_utility": "#FFD700", "wind": "#32CD32", "hydro": "#1E90FF",
#             "coal_black": "#4B4B4B", "gas_ccgt": "#FF8C00",
#             "battery": "#8A2BE2", "bioenergy": "#9ACD32", "distillate": "#CD5C5C"
#         }

#         fig = px.scatter_mapbox(
#             df,
#             lat="lat", lon="lng", color="fuel_tech",
#             hover_name="facility_name",
#             hover_data={"region": True, "power_mw": True, "emissions_tco2e_5m": True, "fuel_tech": True},
#             size="point_size", size_max=25, zoom=4.5, opacity=0.8,
#             center={"lat": -25, "lon": 135}, mapbox_style="carto-positron",
#             color_discrete_map=color_map
#         )
#         fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=650, legend_title_text="Fuel Type")
#         return fig

#     # ---------- (3) Click map → show time series ----------
#     @app.callback(
#         Output("facility-timeseries", "figure"),
#         [Input("facility-map", "clickData"),
#         Input("manual-refresh", "n_clicks")]  
#     )
#     def show_timeseries(clickData, _clicks):
#         if not os.path.exists(OUT_FACILITY_TS):
#             return go.Figure()
#         df = pd.read_csv(OUT_FACILITY_TS, parse_dates=["time"])

#         if not clickData:
#             fig = go.Figure()
#             fig.update_layout(
#                 title="Click a facility to see its 7-day Power Trend",
#                 xaxis_title="Datetime", yaxis_title="Power (MW)",
#                 template="plotly_white"
#             )
#             return fig

#         facility_name = clickData["points"][0]["hovertext"]
#         df_fac = df[df["facility_name"] == facility_name]

#         fig = go.Figure()
#         if df_fac.empty:
#             fig.update_layout(title=f"No data available for {facility_name}", template="plotly_white")
#             return fig

#         fig.add_trace(go.Scatter(x=df_fac["time"], y=df_fac["power_mw"],
#                                 mode="lines", name="Power (MW)",
#                                 line=dict(color="#1f77b4", width=2)))
#         fig.update_layout(title=f"{facility_name} – 7-Day Power Trend",
#                         xaxis_title="Datetime", yaxis_title="Power (MW)",
#                         template="plotly_white")
#         return fig


#     # auto-open a browser tab shortly after server starts
#     if AUTO_OPEN_BROWSER:
#         def _open():
#             time.sleep(1.0)
#             webbrowser.open(f"http://{DASH_HOST}:{DASH_PORT}")
#         threading.Thread(target=_open, daemon=True).start()

#     print(f"🚀 Dashboard on: http://{DASH_HOST}:{DASH_PORT}")
#     app.run(host=DASH_HOST, port=DASH_PORT, debug=True)



# # ---------- Dynamic Data Refresh Loop (COMMENTED OUT by default) ----------
# def continuous_fetch(interval_sec=3600):
#     """
#     Re-fetch task12 data every interval to simulate near-real-time updates.
#     WARNING: This consumes your daily 500-requests quota if set too small.
#     """
#     round_count = 1
#     while True:
#         print(f"\n==============================")
#         print(f"🔁 FETCH ROUND {round_count} START ({time.strftime('%Y-%m-%d %H:%M:%S')})")
#         print(f"==============================")
#         try:
#             fetch_task12_data()
#         except Exception as e:
#             print("❌ Exception in fetch loop:", e)
#         print(f"🕒 Waiting {interval_sec} seconds before next fetch…\n")
#         time.sleep(interval_sec)
#         round_count += 1


# # ==========================================
# # Main Entry
# # ==========================================
# # Uncomment to test MQTT once (publishes all facility records)
# # mqtt_publish_once()
# if __name__ == "__main__":
#     # 1) Run Task 12 once (static batch for the assignment window)
#     try:
#         fetch_task12_data()
#     except Exception as e:
#         print(f"❌ Task 12 fetch failed: {e}")
#         # You may exit if initial data is mandatory
#         # raise

#     # 2) (Optional) Publish once to MQTT with the combined facility CSV
#     # mqtt_publish_once(OUT_FACILITY_TS)

#     # （optional）subscribe to MQTT for debugging/bonus if you want to run it together open another terminal and run
#     # mqtt_subscribe()

#     # 3) Start the live dashboard (auto-refresh reads the CSV written above)
#     run_dashboard()

#     # 4) (Optional) Continuous modes (COMMENTED OUT by default)
#     continuous_fetch(interval_sec=60)          # periodically re-fetch Task 12 data
#     continuous_mqtt_publish(interval_sec=60)     # periodically re-publish MQTT
