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
# KROK 2B: POWER INDEX A ŠANCE ZÁPASŮ
# ==========================================
def vypocitej_power_index(odehrane_df, vsechny_tymy):
    power = {}
    for tym in vsechny_tymy:
        dom_z = odehrane_df[odehrane_df["Domácí"] == tym]
        hos_z = odehrane_df[odehrane_df["Hosté"] == tym]
        zapasy = len(dom_z) + len(hos_z)
        if zapasy == 0:
            power[tym] = 1500.0
            continue
        gf = dom_z["G_Dom"].sum() + hos_z["G_Hos"].sum()
        ga = dom_z["G_Hos"].sum() + hos_z["G_Dom"].sum()
        pyth = (gf ** 2.15) / ((gf ** 2.15) + (ga ** 2.15)) if (gf + ga) > 0 else 0.5
        
        body = 0
        for _, r in dom_z.iterrows():
            if r["G_Dom"] > r["G_Hos"]: body += 3 if r["Typ_Konce"] == "60" else 2
            else: body += 1 if r["Typ_Konce"] != "60" else 0
        for _, r in hos_z.iterrows():
            if r["G_Hos"] > r["G_Dom"]: body += 3 if r["Typ_Konce"] == "60" else 2
            else: body += 1 if r["Typ_Konce"] != "60" else 0
            
        bodova_uspesnost = body / (zapasy * 3)
        kombinovane_skore = 0.5 * bodova_uspesnost + 0.5 * pyth
        power[tym] = 1000.0 + (kombinovane_skore * 1000.0)
    return power

def sance_zapas_rating(rating_dict, dom, hos, vyhoda_domaci=50):
    rozdil_dom = (rating_dict[dom] + vyhoda_domaci) - rating_dict[hos]
    p_dom_zaklad = 1 / (1 + 10 ** (-rozdil_dom / 400))
    vyrovnanost = 1 - abs(p_dom_zaklad - 0.5) * 2
    p_remiza = 0.14 + (0.10 * vyrovnanost)
    p_dom_3b = p_dom_zaklad * (1 - p_remiza)
    p_hos_3b = (1 - p_dom_zaklad) * (1 - p_remiza)
    rozdil_cisty = rating_dict[dom] - rating_dict[hos]
    p_dom_ot = 1 / (1 + 10 ** (-rozdil_cisty / 400))
    return p_dom_3b, p_remiza * p_dom_ot, p_remiza * (1 - p_dom_ot), p_hos_3b

def sance_zapas_poisson(params, prumer_dom, prumer_hos, dom, hos):
    lambda_dom = params[dom]["utok"] * params[hos]["obrana"] * prumer_dom
    lambda_hos = params[hos]["utok"] * params[dom]["obrana"] * prumer_hos
    max_golu = 10
    p_dom = [poisson.pmf(i, lambda_dom) for i in range(max_golu)]
    p_hos = [poisson.pmf(i, lambda_hos) for i in range(max_golu)]
    matice = np.outer(p_dom, p_hos)
    p_dom_3b = np.sum(np.tril(matice, -1))
    p_remiza = np.sum(np.diag(matice))
    p_hos_3b = np.sum(np.triu(matice, 1))
    celkem = p_dom_3b + p_remiza + p_hos_3b
    p_dom_3b /= celkem; p_remiza /= celkem; p_hos_3b /= celkem
    p_dom_ot = lambda_dom / (lambda_dom + lambda_hos)
    return p_dom_3b, p_remiza * p_dom_ot, p_remiza * (1 - p_dom_ot), p_hos_3b

# ==========================================
# KROK 3: AKTUÁLNÍ TABULKA A MONTE CARLO
# ==========================================
def vytvor_aktualni_tabulku(odehrane_df, vsechny_tymy):
    tabulka = {
        tym: {"Z": 0, "V": 0, "VP": 0, "PP": 0, "P": 0, "GF": 0, "GA": 0, "Body": 0}
        for tym in vsechny_tymy
    }
    for _, r in odehrane_df.iterrows():
        dom, hos = r["Domácí"], r["Hosté"]
        g_dom, g_hos, typ = int(r["G_Dom"]), int(r["G_Hos"]), r["Typ_Konce"]
        tabulka[dom]["Z"] += 1; tabulka[hos]["Z"] += 1
        tabulka[dom]["GF"] += g_dom; tabulka[dom]["GA"] += g_hos
        tabulka[hos]["GF"] += g_hos; tabulka[hos]["GA"] += g_dom
        
        if g_dom > g_hos:
            if typ == "60":
                tabulka[dom]["V"] += 1; tabulka[dom]["Body"] += 3; tabulka[hos]["P"] += 1
            else:
                tabulka[dom]["VP"] += 1; tabulka[dom]["Body"] += 2; tabulka[hos]["PP"] += 1; tabulka[hos]["Body"] += 1
        else:
            if typ == "60":
                tabulka[hos]["V"] += 1; tabulka[hos]["Body"] += 3; tabulka[dom]["P"] += 1
            else:
                tabulka[hos]["VP"] += 1; tabulka[hos]["Body"] += 2; tabulka[dom]["PP"] += 1; tabulka[dom]["Body"] += 1
                
    df_tab = pd.DataFrame.from_dict(tabulka, orient="index")
    df_tab["Rozdíl"] = df_tab["GF"] - df_tab["GA"]
    df_tab = df_tab.sort_values(by=["Body", "Rozdíl", "GF"], ascending=False)
    df_tab.insert(0, "Pořadí", range(1, len(df_tab) + 1))
    return df_tab

def spust_simulaci_matice(aktualni_tab_df, budouci_df, sance_zapasu_list, pocet_simulaci=10000):
    tymy = list(aktualni_tab_df.index)
    tym_na_idx = {tym: i for i, tym in enumerate(tymy)}
    n_tymu = len(tymy)
    n_zapasu = len(budouci_df)
    
    start_body = aktualni_tab_df["Body"].values
    start_rozdil = aktualni_tab_df["Rozdíl"].values
    
    celkove_body = np.tile(start_body, (pocet_simulaci, 1)).astype(float)
    celkove_rozdil = np.tile(start_rozdil, (pocet_simulaci, 1)).astype(float)
    
    dom_idx = [tym_na_idx[t] for t in budouci_df["Domácí"]]
    hos_idx = [tym_na_idx[t] for t in budouci_df["Hosté"]]
    nahoda = np.random.rand(pocet_simulaci, n_zapasu)
    
    for z in range(n_zapasu):
        d, h = dom_idx[z], hos_idx[z]
        p1, p2, p3, p4 = sance_zapasu_list[z]
        r = nahoda[:, z]
        
        m_d3 = r < p1
        celkove_body[m_d3, d] += 3; celkove_rozdil[m_d3, d] += 2; celkove_rozdil[m_d3, h] -= 2
        
        m_d2 = (r >= p1) & (r < p1 + p2)
        celkove_body[m_d2, d] += 2; celkove_body[m_d2, h] += 1; celkove_rozdil[m_d2, d] += 1; celkove_rozdil[m_d2, h] -= 1
        
        m_h2 = (r >= p1 + p2) & (r < p1 + p2 + p3)
        celkove_body[m_h2, h] += 2; celkove_body[m_h2, d] += 1; celkove_rozdil[m_h2, h] += 1; celkove_rozdil[m_h2, d] -= 1
        
        m_h3 = r >= p1 + p2 + p3
        celkove_body[m_h3, h] += 3; celkove_rozdil[m_h3, h] += 2; celkove_rozdil[m_h3, d] -= 2

    skore_pro_razeni = celkove_body + (celkove_rozdil * 0.001) + np.random.uniform(0, 0.0001, celkove_body.shape)
    poradi_idx = np.argsort(-skore_pro_razeni, axis=1)
    
    umisteni = np.empty_like(poradi_idx)
    radky = np.arange(pocet_simulaci)[:, None]
    umisteni[radky, poradi_idx] = np.arange(1, n_tymu + 1)
    
    vysledky = []
    for i, tym in enumerate(tymy):
        pozice = umisteni[:, i]
        radek_dat = {
            "Tým": tym,
            "Akt. body": int(start_body[i]),
            "Ø Body": round(np.mean(celkove_body[:, i]), 1)
        }
        for m in range(1, n_tymu + 1):
            radek_dat[f"{m}."] = round(np.mean(pozice == m) * 100, 1)
            
        radek_dat["1–4 (ČF)"] = round(np.mean(pozice <= 4) * 100, 1)
        radek_dat["5–12 (Předkolo)"] = round(np.mean((pozice >= 5) & (pozice <= 12)) * 100, 1)
        radek_dat["13 (Konec)"] = round(np.mean(pozice == 13) * 100, 1)
        radek_dat["14 (Baráž)"] = round(np.mean(pozice == 14) * 100, 1)
        vysledky.append(radek_dat)
        
    df_matice = pd.DataFrame(vysledky).set_index("Tým")
    return df_matice
# ==========================================
# KROK 4: UŽIVATELSKÉ ROZHRANÍ (STREAMLIT)
# ==========================================
st.title("🏒 Matematický simulátor Tipsport Extraligy")

with st.sidebar:
    st.header("1. Vstupní data")
    nahrany_soubor = st.file_uploader("Volitelně: Nahraj novější CSV soubor", type=["csv"])
    
    st.header("2. Parametry simulace")
    pocet_sim = st.slider("Počet simulací sezóny", min_value=1000, max_value=20000, value=10000, step=1000)
    vyhoda_domaci = st.slider("Bonifikace domácího prostředí (body Elo)", min_value=0, max_value=100, value=50, step=5)
    pocet_nejblizsich = st.slider("Počet nejbližších zápasů v predikci", min_value=1, max_value=14, value=7)

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
    st.dataframe(aktualni_tab, use_container_width=True, height=530)
    
    st.divider()
    st.subheader("🔮 Simulace konečného pořadí a predikce nejbližších zápasů")
    
    def zobraz_matici(df_matice):
        df_viz = df_matice.copy()
        df_viz.insert(df_viz.columns.get_loc("5."), " ", "")
        df_viz.insert(df_viz.columns.get_loc("13."), "  ", "")
        df_viz.insert(df_viz.columns.get_loc("14."), "   ", "")
        
        sloupce_1_4 = ["1.", "2.", "3.", "4."]
        sloupce_5_12 = [f"{m}." for m in range(5, 13)]
        sloupce_pozic = [f"{m}." for m in range(1, 15)]
        sloupce_pasem = ["1–4 (ČF)", "5–12 (Předkolo)", "13 (Konec)", "14 (Baráž)"]
        
        styler = (
            df_viz.style
            .background_gradient(subset=sloupce_1_4, cmap="Greens", vmin=0, vmax=60)
            .background_gradient(subset=sloupce_5_12, cmap="Blues", vmin=0, vmax=60)
            .background_gradient(subset=["13."], cmap="Oranges", vmin=0, vmax=60)
            .background_gradient(subset=["14."], cmap="Reds", vmin=0, vmax=60)
            .background_gradient(subset=["1–4 (ČF)"], cmap="Greens", vmin=0, vmax=100)
            .background_gradient(subset=["14 (Baráž)"], cmap="Reds", vmin=0, vmax=100)
            .format("{:.1f}", subset=["Ø Body"] + sloupce_pozic + sloupce_pasem)
            .set_properties(subset=[" ", "  ", "   "], **{"background-color": "#e0e0e0", "width": "8px"})
        )
        st.dataframe(styler, use_container_width=True, height=530)

    def zobraz_nejblizsi_zapasy(budouci_df, sance_list, pocet):
        st.markdown(f"#### 📅 Predikce nejbližších {pocet} zápasů (1 – X – 2 po 60 minutách)")
        vyber = budouci_df.head(pocet)
        radky_zapasu = []
        for idx_listu, (_, r) in enumerate(vyber.iterrows()):
            p_d3, p_d2, p_h2, p_h3 = sance_list[idx_listu]
            p_remiza = p_d2 + p_h2
            datum = r["Začátek"] if "Začátek" in r else ""
            radky_zapasu.append({
                "Termín": datum,
                "Domácí": r["Domácí"],
                "Hosté": r["Hosté"],
                "Výhra domácích (1) %": round(p_d3 * 100, 1),
                "Remíza (X) %": round(p_remiza * 100, 1),
                "Výhra hostů (2) %": round(p_h3 * 100, 1),
                "Celková šance Domácí (vč. prodl.) %": round((p_d3 + p_d2) * 100, 1)
            })
        df_zap = pd.DataFrame(radky_zapasu)
        st.dataframe(
            df_zap.style
            .background_gradient(subset=["Výhra domácích (1) %", "Výhra hostů (2) %"], cmap="Blues", vmin=15, vmax=70)
            .format("{:.1f}", subset=["Výhra domácích (1) %", "Remíza (X) %", "Výhra hostů (2) %", "Celková šance Domácí (vč. prodl.) %"]),
            hide_index=True,
            use_container_width=True
        )
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
        zobraz_nejblizsi_zapasy(budouci, sance_elo, pocet_nejblizsich)

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
        zobraz_nejblizsi_zapasy(budouci, sance_pois, pocet_nejblizsich)

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
        zobraz_nejblizsi_zapasy(budouci, sance_pow, pocet_nejblizsich)

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
            zobraz_nejblizsi_zapasy(budouci, sance_cust, pocet_nejblizsich)

else:
    st.info("👈 Nahraj CSV soubor v levém panelu (nebo ulož soubor export.csv do repozitáře na GitHubu).")
