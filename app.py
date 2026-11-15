import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from scipy.stats import poisson

st.set_page_config(page_title="Simulátor Tipsport Extraligy", layout="wide")

# ==========================================
# KROK 1: NAČTENÍ A VYČIŠTĚNÍ DAT
# ==========================================
@st.cache_data
def nacti_a_vycisti_data(zdroj_dat):
    try:
        df = pd.read_csv(zdroj_dat, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(zdroj_dat, "seek"):
            zdroj_dat.seek(0)
        df = pd.read_csv(zdroj_dat, sep=";", encoding="cp1250")
    
    slovnik_tymu = {
        "HC Oceláři Třinec": "Třinec",
        "HC DYNAMO PARDUBICE": "Pardubice",
        "HC Sparta Praha": "Sparta",
        "HC KOMETA BRNO": "Kometa Brno",
        "Mountfield HK, a.s.": "Hradec Králové",
        "Bílí Tygři Liberec": "Liberec",
        "HC ŠKODA PLZEŇ": "Plzeň",
        "BANES Motor České Budějovice": "Č. Budějovice",
        "HC VÍTKOVICE RIDERA": "Vítkovice",
        "BYD Energie Karlovy Vary": "Karlovy Vary",
        "BK Mladá Boleslav": "Ml. Boleslav",
        "HC Litvínov": "Litvínov",
        "HC Olomouc": "Olomouc",
        "Rytíři Kladno": "Kladno"
    }
    
    df["Domácí"] = df["Domácí"].str.strip().replace(slovnik_tymu)
    df["Hosté"] = df["Hosté"].str.strip().replace(slovnik_tymu)
    vsechny_tymy = sorted(pd.concat([df["Domácí"], df["Hosté"]]).unique())
    
    odehrane = df[df["Stav"].str.contains(":", na=False)].copy()
    budouci = df[~df["Stav"].str.contains(":", na=False)].copy()
    
    def parsuj_vysledek(stav):
        match = re.match(r"(\d+):(\d+)(p|sn)?", str(stav).strip())
        if match:
            g_dom = int(match.group(1))
            g_hos = int(match.group(2))
            typ = match.group(3) if match.group(3) else "60"
            return pd.Series([g_dom, g_hos, typ])
        return pd.Series([None, None, None])

    odehrane[["G_Dom", "G_Hos", "Typ_Konce"]] = odehrane["Stav"].apply(parsuj_vysledek)
    return odehrane, budouci, vsechny_tymy

# ==========================================
# KROK 2A: ELO A POISSON
# ==========================================
def vypocitej_elo(odehrane_df, vsechny_tymy, zaklad=1500, k_faktor=20, vyhoda_domaci=50):
    elo = {tym: float(zaklad) for tym in vsechny_tymy}
    for _, radek in odehrane_df.iterrows():
        dom, hos = radek["Domácí"], radek["Hosté"]
        g_dom, g_hos, typ = radek["G_Dom"], radek["G_Hos"], radek["Typ_Konce"]
        ocek_dom = 1 / (1 + 10 ** ((elo[hos] - (elo[dom] + vyhoda_domaci)) / 400))
        if typ == "60":
            skut_dom = 1.0 if g_dom > g_hos else 0.0
        else:
            skut_dom = 0.75 if g_dom > g_hos else 0.25
        zmena = k_faktor * (skut_dom - ocek_dom)
        elo[dom] += zmena
        elo[hos] -= zmena
    return elo

def vypocitej_poisson_parametry(odehrane_df, vsechny_tymy):
    prumer_dom = odehrane_df["G_Dom"].mean()
    prumer_hos = odehrane_df["G_Hos"].mean()
    parametry = {}
    for tym in vsechny_tymy:
        dom_z = odehrane_df[odehrane_df["Domácí"] == tym]
        hos_z = odehrane_df[odehrane_df["Hosté"] == tym]
        utok_dom = (dom_z["G_Dom"].mean() / prumer_dom) if len(dom_z) > 0 else 1.0
        obrana_dom = (dom_z["G_Hos"].mean() / prumer_hos) if len(dom_z) > 0 else 1.0
        utok_hos = (hos_z["G_Hos"].mean() / prumer_hos) if len(hos_z) > 0 else 1.0
        obrana_hos = (hos_z["G_Dom"].mean() / prumer_dom) if len(hos_z) > 0 else 1.0
        parametry[tym] = {
            "utok": (utok_dom + utok_hos) / 2,
            "obrana": (obrana_dom + obrana_hos) / 2
        }
    return parametry, prumer_dom, prumer_hos

# ==========================================
# KROK 4: UŽIVATELSKÉ ROZHRANÍ (STREAMLIT)
# ==========================================
st.title("🏒 Matematický simulátor Tipsport Extraligy")

with st.sidebar:
    st.header("1. Vstupní data")
    nahrany_soubor = st.file_uploader("Volitelně: Nahraj novější CSV soubor", type=["csv"])
    
    st.header("2. Parametry simulace")
    pocet_sim = st.slider("Počet simulací sezóny", min_value=1000, max_value=20000, value=5000, step=1000)
    vyhoda_domaci = st.slider("Bonifikace domácího prostředí (body Elo)", min_value=0, max_value=100, value=50, step=5)

if nahrany_soubor is not None:
    aktivni_zdroj = nahrany_soubor
    st.sidebar.success("✅ Používám tvůj nahraný soubor.")
elif os.path.exists("export.csv"):
    aktivni_zdroj = "export.csv"
    st.sidebar.info("ℹ️ Používám výchozí data z GitHubu (export.csv).")
else:
    aktivni_zdroj = None

if aktivni_zdroj is not None:
    odehrane, budouci, vsechny_tymy = nacti_a_vycisti_data(aktivni_zdroj)
    
    st.subheader(f"📊 Aktuální reálná tabulka (odehráno {len(odehrane)} z {len(odehrane) + len(budouci)} zápasů)")
    aktualni_tab = vytvor_aktualni_tabulku(odehrane, vsechny_tymy)
    
    # Vizuální oddělení pásem v aktuální tabulce (pod 4., 12. a 13. řádkem)
    def styl_aktualni_tabulky(df):
        styly = pd.DataFrame("", index=df.index, columns=df.columns)
        for i in range(len(df)):
            poradi = df.iloc[i]["Pořadí"]
            if poradi in:
                styly.iloc[i, :] = "border-bottom: 3px solid #ff4b4b;"
        return styly

    st.dataframe(
        aktualni_tab.style.apply(styl_aktualni_tabulky, axis=None),
        use_container_width=True,
        height=530
    )
    
    st.divider()
    st.subheader("🔮 Simulace konečného pořadí podle jednotlivých modelů")
    
    # Vizuální oddělení sloupců (4.|5., 12.|13., 13.|14.) a barevné odlišení pásem
    def zobraz_matici(df_matice):
        sloupce_1_4 = ["1.", "2.", "3.", "4."]
        sloupce_5_12 = [f"{m}." for m in range(5, 13)]
        sloupce_pozic = [f"{m}." for m in range(1, 15)]
        sloupce_pasem = ["1–4 (ČF)", "5–12 (Předkolo)", "13 (Konec)", "14 (Baráž)"]
        
        styler = (
            df_matice.style
            .background_gradient(subset=sloupce_1_4, cmap="Greens", vmin=0, vmax=60)
            .background_gradient(subset=sloupce_5_12, cmap="Blues", vmin=0, vmax=60)
            .background_gradient(subset=["13."], cmap="Oranges", vmin=0, vmax=60)
            .background_gradient(subset=["14."], cmap="Reds", vmin=0, vmax=60)
            .background_gradient(subset=["1–4 (ČF)"], cmap="Greens", vmin=0, vmax=100)
            .background_gradient(subset=["14 (Baráž)"], cmap="Reds", vmin=0, vmax=100)
            .format("{:.1f}", subset=["Ø Body"] + sloupce_pozic + sloupce_pasem)
            .set_properties(subset=["4.", "12.", "13."], **{"border-right": "3px solid #ff4b4b"})
        )
        
        st.dataframe(styler, use_container_width=True, height=530)

    tab_elo, tab_poisson, tab_power, tab_custom = st.tabs([
        "📈 Model 1: Elo rating", 
        "🥅 Model 2: Poisson (gólový)", 
        "⚖️ Model 3: Power Index", 
        "🎛️ Model 4: Vlastní síly týmů"
    ])
    
    # --- ZÁLOŽKA 1: ELO ---
    with tab_elo:
        elo_sily = vypocitej_elo(odehrane, vsechny_tymy, vyhoda_domaci=vyhoda_domaci)
        with st.expander("Zobrazit vypočítané Elo ratingy týmů (výchozí = 1500)"):
            df_elo_sily = pd.DataFrame(
                [(t, round(elo_sily[t], 1)) for t in aktualni_tab.index],
                columns=["Tým", "Elo rating"]
            ).set_index("Tým")
            st.dataframe(df_elo_sily.T)
            
        sance_elo = [sance_zapas_rating(elo_sily, r["Domácí"], r["Hosté"], vyhoda_domaci) for _, r in budouci.iterrows()]
        matice_elo = spust_simulaci_matice(aktualni_tab, budouci, sance_elo, pocet_sim)
        zobraz_matici(matice_elo)

    # --- ZÁLOŽKA 2: POISSON ---
    with tab_poisson:
        pois_params, prum_dom, prum_hos = vypocitej_poisson_parametry(odehrane, vsechny_tymy)
        with st.expander("Zobrazit útočnou a obrannou sílu týmů (1.00 = ligový průměr)"):
            df_pois_sily = pd.DataFrame(
                [(t, round(pois_params[t]["utok"], 2), round(pois_params[t]["obrana"], 2)) for t in aktualni_tab.index],
                columns=["Tým", "Síla útoku (více = lépe)", "Propustnost obrany (méně = lépe)"]
            ).set_index("Tým")
            st.dataframe(df_pois_sily.T)
            
        sance_pois = [sance_zapas_poisson(pois_params, prum_dom, prum_hos, r["Domácí"], r["Hosté"]) for _, r in budouci.iterrows()]
        matice_pois = spust_simulaci_matice(aktualni_tab, budouci, sance_pois, pocet_sim)
        zobraz_matici(matice_pois)

    # --- ZÁLOŽKA 3: POWER INDEX ---
    with tab_power:
        power_sily = vypocitej_power_index(odehrane, vsechny_tymy)
        with st.expander("Zobrazit vypočítaný Power Index týmů (kombinace bodů a skóre)"):
            df_pow_sily = pd.DataFrame(
                [(t, round(power_sily[t], 1)) for t in aktualni_tab.index],
                columns=["Tým", "Power Index"]
            ).set_index("Tým")
            st.dataframe(df_pow_sily.T)
            
        sance_pow = [sance_zapas_rating(power_sily, r["Domácí"], r["Hosté"], vyhoda_domaci) for _, r in budouci.iterrows()]
        matice_pow = spust_simulaci_matice(aktualni_tab, budouci, sance_pow, pocet_sim)
        zobraz_matici(matice_pow)

    # --- ZÁLOŽKA 4: VLASTNÍ SÍLY ---
    with tab_custom:
        st.info("Uprav sílu týmů na škále 0 až 100 (výchozí hodnoty odpovídají modelu Elo). Po úpravě klikni na tlačítko níže.")
        
        vychozi_custom = pd.DataFrame({
            "Tým": list(aktualni_tab.index),
            "Vlastní síla (0-100)": [int(np.clip(round((elo_sily[t] - 1200) / 6), 1, 100)) for t in aktualni_tab.index]
        })
        
        upravena_tab = st.data_editor(
            vychozi_custom,
            column_config={
                "Tým": st.column_config.TextColumn(disabled=True),
                "Vlastní síla (0-100)": st.column_config.NumberColumn(min_value=1, max_value=100, step=1)
            },
            hide_index=True,
            use_container_width=True
        )
        
        if st.button("🚀 Spustit simulaci s vlastními silami", type="primary"):
            custom_dict = {
                radek["Tým"]: 1200.0 + (radek["Vlastní síla (0-100)"] * 6.0)
                for _, radek in upravena_tab.iterrows()
            }
            sance_cust = [sance_zapas_rating(custom_dict, r["Domácí"], r["Hosté"], vyhoda_domaci) for _, r in budouci.iterrows()]
            matice_cust = spust_simulaci_matice(aktualni_tab, budouci, sance_cust, pocet_sim)
            zobraz_matici(matice_cust)

else:
    st.info("👈 Nahraj CSV soubor v levém panelu (nebo ulož soubor export.csv do repozitáře na GitHubu).")
