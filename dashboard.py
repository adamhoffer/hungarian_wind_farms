import os
import main
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# GET DATA
if not os.path.exists("data_lake/gold/gold_power_forecast.parquet"):
    with st.spinner("Adatok frissítése az API-ból... Ez eltarthat egy percig."):
        main.fetch_and_save_bronze()
        main.process_silver()
        main.process_gold()


# SET PAGE
st.set_page_config(page_title="Magyar Szélerőmű Előrejelzés", layout="wide")
st.title("Magyarországi Szélerőművek - 7 Napos Termelés-előrejelzés")


# Gold adatok beolvasása
@st.cache_data
def load_data():
    df = pd.read_parquet("data_lake/gold/gold_power_forecast.parquet")
    return df

df = load_data()


# 1. Szűrő a szélerőművekre
all_locations = sorted(df["location_name"].dropna().unique())
selected_locations = st.multiselect(
    "Válassz ki településeket az elemzéshez:",
    options=all_locations,
    default=all_locations
)

filtered_df = df[df["location_name"].isin(selected_locations)]

if filtered_df.empty:
    st.warning("Kérlek, válassz ki legalább egy települést!")


# Egyedi helyszínek koordinátáinak és beépített kapacitásának kinyerése
map_df = df.groupby("location_name").agg({
    "latitude": "first",
    "longitude": "first",
    "rated_power_mw": "sum"
}).reset_index()

# Státusz és szín/méret beállítása a kiválasztástól függően
map_df["Kijelölve"] = map_df["location_name"].apply(
    lambda x: "Kiválasztva" if x in selected_locations else "Nincs kiválasztva"
)

# Színkódolás: Kiválasztott -> Kék/Környezetfüggő, Nem kiválasztott -> Szürke
color_map = {
    "Kiválasztva": "#00CC96",      # Élénk zöld/kék
    "Nincs kiválasztva": "#A0A0A0"  # Szürke
}

fig_map = px.scatter_map(
    map_df,
    lat="latitude",
    lon="longitude",
    color="Kijelölve",
    color_discrete_map=color_map,
    size=map_df["Kijelölve"].apply(lambda x: 14 if x == "Kiválasztva" else 8),
    hover_name="location_name",
    hover_data={"rated_power_mw": ":.1f MW", "latitude": False, "longitude": False, "Kijelölve": False},
    zoom=6.8,
    center={"lat": 47.1625, "lon": 19.5033}, # Magyarország közepe
    map_style="carto-positron", # Letisztult, szép térkép stílus
    title="Szélerőművek elhelyezkedése Magyarországon"
)

fig_map.update_layout(
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
)

st.plotly_chart(fig_map, use_container_width=True)


# 2. Összesített idősor kiszámítása (Időpont szerint szummázzuk a kiválasztottakat)
total_power_df = (
    filtered_df.groupby("forecast_time")["estimated_power_mw"]
    .sum()
    .reset_index()
)

# 3. KPI kártyák dinamikusan a kiválasztott parkokra
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tornyok száma", f"{filtered_df['tower_id'].nunique()} db")

selected_max_cap = filtered_df.groupby('tower_id')['rated_power_mw'].first().sum()
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
    color="location_name",
    title="Becsült Áramtermelés (MW) az elkövetkező 7 napban",
    labels={
        "forecast_time": "Időpont",
        "estimated_power_mw": "Becsült Termelés (MW)",
        "location_name": "Helyszín"
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
