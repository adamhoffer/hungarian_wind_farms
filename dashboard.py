import logging
from datetime import datetime

import os
import main

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# LOGGING
# 1) log setup
logging.basicConfig(
    filename='dashboard_access.log',
    level=logging.INFO,
    encoding='utf-8'
)

# 2) get log info
user_agent = st.context.headers.get("User-Agent", "Ismeretlen eszköz")
time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 3) write into logfile
logging.info(f"[{time_now}] Új megtekintés! Eszköz: {user_agent}")


# GET DATA
if not os.path.exists("data_lake/gold/gold_power_forecast.parquet"):
    with st.spinner("Adatok frissítése az API-ból... Ez eltarthat egy percig."):
        main.fetch_and_save_bronze()
        main.process_silver()
        main.process_gold()


# SET PAGE
st.set_page_config(page_title="Magyar Szélerőmű Előrejelzés", layout="wide")
st.title("💨 Magyarországi Szélerőmű Parkok - 7 Napos Termelés-előrejelzés")


# Gold adatok beolvasása
@st.cache_data
def load_data():
    df = pd.read_parquet("data_lake/gold/gold_power_forecast.parquet")
    return df


df = load_data()

# 1. Szűrő a szélerőművekre
selected_farms = st.multiselect(
    "Válassz ki szélerőműveket az elemzéshez:",
    options=df["location_name_list"].unique(),
    default=df["location_name_list"].unique()[:4]
)

filtered_df = df[df["location_name_list"].isin(selected_farms)]

if filtered_df.empty:
    st.warning("Kérlek, válassz ki legalább egy szélerőművet!")
else:
    # 2. Összesített idősor kiszámítása (Időpont szerint szummázzuk a kiválasztottakat)
    total_power_df = (
        filtered_df.groupby("forecast_time")["estimated_power_mw"]
        .sum()
        .reset_index()
    )

    # 3. KPI kártyák dinamikusan a kiválasztott parkokra
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kiválasztott Parkok", f"{filtered_df['turbine_id'].nunique()} db")

    selected_max_cap = filtered_df.groupby('turbine_id')['max_capacity_mw'].first().sum()
    col2.metric("Kiválasztott Beépített Kapacitás", f"{selected_max_cap:.1f} MW")

    max_total_power = total_power_df["estimated_power_mw"].max()
    col3.metric("Max. Együttes Várható Csúcs", f"{max_total_power:.1f} MW")

    avg_total_power = total_power_df["estimated_power_mw"].mean()
    col4.metric("Átlagos Összesített Termelés", f"{avg_total_power:.1f} MW")

    st.markdown("---")

    # 4. Diagram elkészítése Plotly-val
    # 4.a. Egyéni parkok vonalai
    fig = px.line(
        filtered_df,
        x="forecast_time",
        y="estimated_power_mw",
        color="location_name_list",
        title="Becsült Áramtermelés (MW) az elkövetkező 7 napban",
        labels={
            "forecast_time": "Időpont",
            "estimated_power_mw": "Becsült Termelés (MW)",
            "location_name_list": "Helyszín"
        }
    )

    # 4.b. ÖSSZESÍTETT termelés hozzáadása vastag, fekete/szaggatott vonalként
    fig.add_trace(
        go.Scatter(
            x=total_power_df["forecast_time"],
            y=total_power_df["estimated_power_mw"],
            mode="lines",
            name="<b>⚡ ÖSSZESÍTETT TERMELÉS</b>",
            line=dict(color="black", width=4, dash="dash")
        )
    )

    # Diagram finomhangolása
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 5. Adattáblázat megtekintése
    with st.expander("Nyers Gold Adatok Megtekintése"):
        st.dataframe(filtered_df)
