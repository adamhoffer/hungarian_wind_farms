import os
import sqlite3
from zoneinfo import ZoneInfo
import pandas as pd
from datetime import datetime
import requests
import json
import glob
import numpy as np

import ast



# Folder setup
BRONZE_DIR = "data_lake/bronze"
SILVER_DIR = "data_lake/silver"
GOLD_DIR = "data_lake/gold"

for folder in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    os.makedirs(folder, exist_ok=True)


# Időjárás adatok lekérése, az összes lokációhoz, egyszerre.
def fetch_and_save_bronze():
    print("--- BRONZE RÉTEG: Adatgyűjtés indítása ---")


    # összes lokáció összegyűjtése (lat, lon)
    try:
        con = sqlite3.connect("data.db")
        cur = con.cursor()

        res = cur.execute("SELECT latitude, longitude FROM towers")
        loc_list = res.fetchall()

        con.close()

    except Exception as e:
        print(f"Hiba a beolvasás közben: {e}")
        return

    lat_list = [str(loc[0]) for loc in loc_list]
    lon_list = [str(loc[1]) for loc in loc_list]


    # HTTP link összeállítása
    latitudes = ",".join(lat_list)
    longitudes = ",".join(lon_list)
    api_url = f"""https://api.open-meteo.com/v1/forecast?latitude={latitudes}&longitude={longitudes}&hourly=windspeed_80m&wind_speed_unit=ms&timezone=auto"""


    # időjárásadatok lekérése
    timestamp = datetime.now(ZoneInfo("Europe/Budapest")).strftime("%Y-%m-%dT%H:%M:%S")
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Hiba az API hívás során. Státuszkód: {response.status_code}")
        return


    # nyers adatok elmentése tornyonként json fájlokba
    # Elmentjük a metaadatokat is
    try:
        raw_data = response.json()
        for i, obj in enumerate(raw_data, start=1):
            filename = f"{BRONZE_DIR}/tower_{i}.json"
            obj["meta_ingestion_timestamp"] = timestamp
            obj["meta_tower_id"] = i
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Hiba a lekért adat kiegészítése, mentése közben: {e}")
        return

    print(f"Időjárásadatok mentve, minden lokációhoz.")


# Időjárás adatok tisztítása, transzformázása.
def process_silver():
    print(f"\n--- EZÜST RÉTEG: Adattisztítás és idősorok kicsomagolása ---")

    # Megkeressük a json fájlokat
    json_files = glob.glob(f"{BRONZE_DIR}/*.json")
    if not json_files:
        print(f"Nem található feldolgozandő JSON fájl a {BRONZE_DIR} mappában")
        return

    # Feldolgozzuk a json fájlokat.
    all_hourly_records = []

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            tower_id = data.get("meta_tower_id")

            hourly_data = data.get("hourly", {})

            times = hourly_data.get("time", [])
            ws_80m = hourly_data.get("windspeed_80m", [])


            for i in range(len(times)):
                record = {
                    "tower_id" : tower_id,
                    "forecast_time" : times[i],
                    "windspeed_80m" : ws_80m[i],
                }
                all_hourly_records.append(record)

    # Sikerült-e adatot kinyerni?
    if not all_hourly_records:
        print("Nem sikerült rekordokat kinyerni a JSON fájlokból")
        return

    # Pandas DataFrame-be rendezzük az adatokat
    silver_df = pd.DataFrame(all_hourly_records)

    # Adattípusok beállítása
    silver_df["tower_id"] = silver_df["tower_id"].astype(int)
    silver_df["forecast_time"] = pd.to_datetime(silver_df["forecast_time"])
    silver_df["windspeed_80m"] = pd.to_numeric(silver_df["windspeed_80m"])

    # Adatok mentése
    output_path = f"{SILVER_DIR}/silver_wind_data.parquet"
    silver_df.to_parquet(output_path, index=False)

    print(f"Sikeresen transzformálva {len(silver_df)} db rekord")
    print(f" -> Mentve Parquet formátumban: {output_path}")


#
def process_gold():
    print("\n--- ARANY RÉTEG: Üzleti aggregáció és termelés-becslés ---")

    # 1) feldolgozott időjárásadatok megkeresése, betöltése
    silver_path = f"{SILVER_DIR}/silver_wind_data.parquet"
    if not os.path.exists(silver_path):
        print(f"Nem található a Silver Parquet fájl itt: {silver_path}")
        return

    weather_df = pd.read_parquet(silver_path)

    # 2) toronyadatok betöltése
    try:
        con = sqlite3.connect("data.db")
        cur = con.cursor()

        sql_query= ("SELECT tow.tower_id AS tower_id,\
                            turb.turbine_type_name,\
                            tow.location_name,\
                            tow.operator_or_licensee,\
                            turb.rated_power_mw,\
                            turb.cut_in_windspeed_mps,\
                            turb.cut_out_windspeed_mps,\
                            turb.power_curve\
                    FROM towers tow\
                    INNER JOIN turbine_data turb\
                        ON tow.turbine_type_id = turb.turbine_type_id"
                    )
        res = cur.execute(sql_query)
        columns = [col[0] for col in cur.description]
        r = res.fetchall()
        all_tower_data_df = pd.DataFrame(r, columns=columns)

        con.close()

    except Exception as e:
        print(f"Hiba a beolvasás közben: {e}")
        return


    # power_curve szöveges szótár átalakítása valódi Python dict-té (float kulcsokkal)
    def parse_power_curve(val):
        if isinstance(val, str):
            d = ast.literal_eval(val)
            return {float(k): float(v) for k, v in d.items()}
        elif isinstance(val, dict):
            return {float(k): float(v) for k, v in val.items()}
        return {}

    all_tower_data_df["power_curve"] = all_tower_data_df["power_curve"].apply(parse_power_curve)



    # 3) időjárásadatok összekapcsolása toronyadatokkal
    gold_df = pd.merge(
        weather_df,
        all_tower_data_df,
        on="tower_id",
        how="inner"
    )
    print(gold_df.head(170))


    # 4) termelés számítása tornyonként & óránként
    def calculate_power_kw(row):
        """
        Kiszámolja a termelt teljesítményt kW-ban az adott turbina power_curve szótára
        és a szélsebesség (mps) alapján, interpolációval.
        """
        ws = row["windspeed_80m"]
        cut_in = row["cut_in_windspeed_mps"]
        cut_out = row["cut_out_windspeed_mps"]
        pc = row["power_curve"]  # pl. {3.0: 0, 3.5: 42.2, 4.0: 93.3 ...}

        # 1. Leállási feltételek (cut-in alatt vagy cut-out felett/egyenlő)
        if pd.isna(ws) or ws < cut_in or ws >= cut_out or not pc:
            return 0.0

        # 2. Ha a szélsebesség pontosan benne van a power_curve kulcsai között
        if ws in pc:
            return float(pc[ws])

        # 3. Lineáris interpoláció a két legközelebbi szélsebesség-pont között
        speeds = sorted(pc.keys())

        # Ha lassabb a szél, mint a görbe legkisebb megadott pontja
        if ws < speeds[0]:
            return 0.0
        # Ha gyorsabb a szél, mint a görbe legnagyobb pontja (elérte a névleges teljesítményt)
        if ws > speeds[-1]:
            return float(pc[speeds[-1]])

        # Megkeressük a két szomszédos mérési pontot
        for i in range(len(speeds) - 1):
            if speeds[i] <= ws < speeds[i + 1]:
                x0, x1 = speeds[i], speeds[i + 1]
                y0, y1 = float(pc[x0]), float(pc[x1])
                # Lineáris interpolációs képlet
                power = y0 + (ws - x0) * (y1 - y0) / (x1 - x0)
                return power

        return 0.0


    # Termelés számítása
    print("Termelés számítása a teljesítménygörbék alapján...")
    gold_df["estimated_power_kw"] = gold_df.apply(calculate_power_kw, axis=1)

    # Átváltás Megawattba (1 MW = 1000 kW)
    gold_df["estimated_power_mw"] = gold_df["estimated_power_kw"] / 1000.0

    # Hatékonysági mutató (%)
    gold_df["capacity_factor_pct"] = np.where(
        gold_df["rated_power_mw"] > 0,
        (gold_df["estimated_power_mw"] / gold_df["rated_power_mw"]) * 100,
        0.0
    )


    # Eredmény mentése a Gold mappába
    del gold_df["power_curve"]

    output_path = f"{GOLD_DIR}/gold_power_forecast.parquet"
    gold_df.to_parquet(output_path, index=False)

    print(f"Sikeresen kiszámítva {len(gold_df)} termelési előrejelzés!")
    print(f" -> Mentve Gold Parquet formátumban: {output_path}")



if __name__ == "__main__":

    fetch_and_save_bronze()
    process_silver()
    process_gold()