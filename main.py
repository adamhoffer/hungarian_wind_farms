import os
import pandas as pd
from datetime import datetime
import requests
import json
import glob
import numpy as np



# Dictionary setup
bronze_dir = "data_lake/bronze"
silver_dir = "data_lake/silver"
gold_dir = "data_lake/gold"

for folder in [bronze_dir, silver_dir, gold_dir]:
    os.makedirs(folder, exist_ok=True)

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTM4zoFATcdTjO-RFcjh6pU8SEy31__RVUUMTfV1glSFoKqH3RAqb0AmeFpwKnpPB4v3Bqtia_wBPSP/pub?gid=0&single=true&output=csv"


def fetch_and_save_bronze():
    print("--- BRONZE RÉTEG: Adatgyűjtés indítása ---")

    # 1. Referencia adatok beolvasása a Google Sheets-ből
    try:
        turbines_df = pd.read_csv(SHEET_CSV_URL)
        print(f"Sikeresen beolvasva {len(turbines_df)} turbina adata a Google Sheets-ből.")
    except Exception as e:
        print(f"Hiba a Google Sheets beolvasásakor: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Időjárás lekérése a különböző koordinátákra
    for index, row in turbines_df.iterrows():
        turbine_id = row['turbine_id']
        lat = row['latitude']
        lon = row['longitude']

        print(f"Adatok lekérése a következőhöz: {turbine_id} {row['location_name_list']} {(lat,lon)}")

        # API
        # api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={appid}&units=metric"
        api_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=windspeed_10m,windspeed_80m,winddirection_80m"

        response = requests.get(api_url)

        if response.status_code == 200:
            raw_data = response.json()

            # Elmentjük a metaadatokat is JSON-be
            raw_data["meta_turbine_id"] = turbine_id
            raw_data["meta_ingestion_timestamp"] = timestamp

            # Fájl mentése bronz mappába
            filename = f"{bronze_dir}/{turbine_id}_{timestamp}.json"
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=4)

            print(f"Mentve: {filename}")

        else:
            print(f"Hiba az API hívás során a {turbine_id} szélerőmű parknál. Státuszkód: {response.status_code}")


def process_silver():
    print(f"\n--- EZÜST RÉTEG: Adattisztítás és idősorok kicsomagolása ---")

    # Megkeressük a json fájlokat
    json_files = glob.glob(f"{bronze_dir}/*.json")
    if not json_files:
        print(f"Nem található feldolgozandő JSON fájl a {bronze_dir} mappában")
        return

    # Feldolgozzuk a json fájlokat.
    all_hourly_records = []

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            turbine_id = data.get("meta_turbine_id")
            ing_timestamp = data.get("meta_ingestion_timestamp")

            hourly_data = data.get("hourly", {})

            times = hourly_data.get("time", [])
            ws_80m = hourly_data.get("windspeed_80m", [])
            wd_80m = hourly_data.get("winddirection_80m", [])


            for i in range(len(times)):
                record = {
                    "turbine_id" : turbine_id,
                    "ingestion_timestamp" : ing_timestamp,
                    "forecast_time" : times[i],
                    "windspeed_80m" : ws_80m[i],
                    "winddirection_80m_deg" : wd_80m[i]
                }
                all_hourly_records.append(record)

    if not all_hourly_records:
        print("Nem sikerült rekordokat kinyerni a JSON fájlokból")
        return

    # Pandas DataFrame-be rendezzük az adatokat
    silver_df = pd.DataFrame(all_hourly_records)

    # Adattípusok beállítása
    silver_df["turbine_id"] = silver_df["turbine_id"].astype(str)
    silver_df["ingestion_timestamp"] = pd.to_datetime(silver_df["ingestion_timestamp"], format="%Y%m%d_%H%M%S")
    silver_df["forecast_time"] = pd.to_datetime(silver_df["forecast_time"])
    silver_df["windspeed_80m"] = pd.to_numeric(silver_df["windspeed_80m"])
    silver_df["winddirection_80m_deg"] = pd.to_numeric(silver_df["winddirection_80m_deg"])

    # Adatok mentése
    output_path = f"{silver_dir}/silver_wind_data.parquet"
    silver_df.to_parquet(output_path, index=False)

    print(f"Sikeresen transzformálva {len(silver_df)} db rekord")
    print(f" -> Mentve Parquet formátumban: {output_path}")


def process_gold():
    print("\n--- ARANY RÉTEG: Üzleti aggregáció és termelés-becslés ---")

    silver_path = f"{silver_dir}/silver_wind_data.parquet"
    if not os.path.exists(silver_path):
        print(f"Nem található a Silver Parquet fájl itt: {silver_path}")
        return

    # 1. silver adatok beolvasása
    silver_df = pd.read_parquet(silver_path)

    # 2. Törzsadatok (Master Data) beolvasása Google Sheets-ből
    try:
        windfarms_master = pd.read_csv(SHEET_CSV_URL)
        windfarms_master["turbine_id"] = windfarms_master["turbine_id"].astype(str)
        windfarms_master["max_capacity_mw"] = pd.to_numeric(windfarms_master["max_capacity_mw"])
    except Exception as e:
        print(f"Hiba a törzsadatok beolvasásakor: {e}")
        return

    # 3. Adatok összekapcsolása (join)
    gold_df = pd.merge(
        silver_df,
        windfarms_master[["turbine_id", "location_name_list", "max_capacity_mw"]],
        on="turbine_id",
        how="inner"
    )

    # 4. Termelés becslése
    ws = gold_df["windspeed_80m"]
    cap = gold_df["max_capacity_mw"]

    # Teljesítmény-görbe összerakása
    conditions = [
        (ws <= 10) | (ws > 90), # Leállás: túl gyenge vagy túl erős szél
        (10 < ws) & (ws <= 45), # Részleges termelés
        (45 < ws) & (ws <= 90) # Max kapacitással termel, egyre kifordítottabb lapátokkal
    ]

    values = [
        0, # 0 MW
        cap * ( (ws - 10) / (45 - 10) ) ** 3, # Köbös skálázás
        cap # Max kapacitással termelés
    ]

    gold_df["estimated_power_mw"] = np.select(conditions, values, default=0)

    # Hatékonyság mutaó (%)
    gold_df["capacity_factor_pct"] = (gold_df["estimated_power_mw"] / gold_df["max_capacity_mw"]) * 100

    # Eredmény mentése a Gold mappába
    output_path = f"{gold_dir}/gold_power_forecast.parquet"
    gold_df.to_parquet(output_path, index=False)

    print(f"Sikeresen kiszámítva {len(gold_df)} termelési előrejelzés!")
    print(f" -> Mentve Gold Parquet formátumban: {output_path}")



if __name__ == "__main__":

    fetch_and_save_bronze()
    process_silver()
    process_gold()
