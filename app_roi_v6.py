# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from pathlib import Path

# === CONFIG ===
from pathlib import Path
APP_DIR = Path(__file__).parent
ANALYSIS_CSV = APP_DIR / "data" / "getaround_x_autoscout_analysis.csv"
COMPAT_STATUS_CSV = APP_DIR / "data" / "compat_status_by_model.csv"


PAGE_TITLE = "ROI Getaround — Simulateur (statut compatibilité)"
PAGE_ICON  = "🚗"

# === Colonnes numériques attendues dans l'analyse croisée ===
NUM_COLS = [
    "annee_i","n_total",
    "prix_min_w","prix_max_w","prix_moyen_w","prix_median_w",
    "prix_min_estime","prix_max_estime","prix_moyen_estime","prix_median_estime",
    "prix_moyen_final","estimation_eur_f",
    "roi_brut_mensuel","payback_mois","taux_rentabilite",
    "presence_marche_fuel","presence_marche_total"
]

def load_analysis(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        st.error(f"Fichier introuvable : {path}")
        st.stop()
    df = pd.read_csv(path, dtype=str)
    # num
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    # texte minimales
    for c in ["ville","marque","modele","annee","carburant_n","tranche_km","Type","SousType","statut"]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].astype(str)
    df["carburant_n"] = df["carburant_n"].fillna("").str.lower()
    df["annee_i"] = pd.to_numeric(df["annee_i"], errors="coerce")

    # références prix (fallback)
    df["prix_min_use"]    = df["prix_min_w"].where(df["prix_min_w"]>0,    df["prix_min_estime"])
    df["prix_max_use"]    = df["prix_max_w"].where(df["prix_max_w"]>0,    df["prix_max_estime"])
    df["prix_moyen_use"]  = df["prix_moyen_w"].where(df["prix_moyen_w"]>0,df["prix_moyen_estime"])
    df["prix_median_use"] = df["prix_median_w"].where(df["prix_median_w"]>0,df["prix_median_estime"])
    df["prix_ref"]        = df["prix_moyen_final"].where(df["prix_moyen_final"]>0, df["prix_moyen_use"])
    return df

def load_compat_status(path: str) -> pd.DataFrame:
    """
    Lit le fichier compat_status_by_model.csv (sorti par build_compat_status.py).
    Attend au minimum: marque, modele, annee_i, Type, estimation_eur_f, compatible, direction
    """
    if not Path(path).exists():
        st.warning("Fichier compatibilité introuvable — statut affiché comme '—'.")
        return pd.DataFrame(columns=["marque","modele","annee_i","Type","compatible"])
    df = pd.read_csv(path, dtype=str)
    # normalisation
    for c in ["marque","modele","Type"]:
        if c in df.columns: df[c] = df[c].astype(str)
    if "annee_i" in df.columns:
        df["annee_i"] = pd.to_numeric(df["annee_i"], errors="coerce")
    if "compatible" in df.columns:
        # garder 0/1 entiers
        df["compatible"] = pd.to_numeric(df["compatible"], errors="coerce").fillna(0).astype(int)
    else:
        df["compatible"] = 0
    return df

def compute_kpis(revenus_mensuels, prix_achat):
    if prix_achat is None or prix_achat <= 0 or revenus_mensuels is None or revenus_mensuels <= 0:
        return np.nan, np.nan, np.nan
    roi = revenus_mensuels / prix_achat
    payback = prix_achat / revenus_mensuels
    renta5 = (revenus_mensuels * 60) / prix_achat
    return roi, payback, renta5

def price_position(price, pmin, pmed, pmean, pmax):
    parts = []
    if pd.notna(pmin):  parts.append(f"min: {int(round(pmin))}€")
    if pd.notna(pmed):  parts.append(f"médian: {int(round(pmed))}€")
    if pd.notna(pmean): parts.append(f"moyen: {int(round(pmean))}€")
    if pd.notna(pmax):  parts.append(f"max: {int(round(pmax))}€")
    band = " · ".join(parts)
    msg = f"Fourchette AutoScout — {band}" if band else "Pas de référence de prix disponible."
    if pd.notna(pmin) and price < pmin:
        state = "✅ Très bon deal (sous min)"
    elif pd.notna(pmed) and price <= pmed:
        state = "✅ Bon deal (≤ médian)"
    elif pd.notna(pmean) and price <= pmean:
        state = "🟨 Correct (≤ moyen)"
    elif pd.notna(pmax) and price <= pmax:
        state = "⚠️ Cher (proche du max)"
    else:
        state = "⚠️ Au-dessus des bornes connues"
    return msg, state

def plot_price_ruler(user_price, pmin, pmed, pmean, pmax):
    fig, ax = plt.subplots(figsize=(9, 1.8), dpi=170)
    xs = [x for x in [pmin, pmed, pmean, pmax] if pd.notna(x)]
    if xs:
        lo, hi = min(xs + [user_price]), max(xs + [user_price])
        pad = max(300, 0.05*(hi - lo))
        ax.hlines(1, lo - pad, hi + pad, linewidth=8, alpha=0.2)
        labels = [("min", pmin), ("médian", pmed), ("moyen", pmean), ("max", pmax)]
        y_above, y_below, toggle = 1.10, 0.84, True
        for name, val in labels:
            if pd.notna(val):
                ax.plot([val], [1], marker="o")
                ytxt = y_above if toggle else y_below
                ax.annotate(f"{name}\n{int(round(val))}€", (val,1), xytext=(val, ytxt),
                            textcoords="data", ha="center", fontsize=9,
                            arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5))
                toggle = not toggle
        ax.plot([user_price], [1], marker="D")
        ax.annotate(f"ton prix\n{int(round(user_price))}€", (user_price,1), xytext=(user_price, 1.24),
                    textcoords="data", ha="center", fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.7))
        ax.set_ylim(0.7,1.35); ax.set_yticks([])
        ax.set_xlabel("Référence de prix AutoScout (€)")
    else:
        ax.text(0.5,0.5,"Références de prix indisponibles.", ha="center", va="center")
        ax.axis("off")
    st.pyplot(fig, use_container_width=True)

def generate_pdf_bytes(context: dict) -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 — résumé & KPI
        plt.figure(figsize=(11.69, 8.27))
        plt.axis('off')
        title = f"Fiche d'achat — {context['marque']} {context['modele']} {context['annee']} ({context['carburant']}, {context['km']})"
        if context.get("ville"):
            title += f" — {context['ville']}"
        lines = [
            f"Revenus mensuels (Getaround): {int(round(context['revenus']))} €" if pd.notna(context['revenus']) else "Revenus mensuels: —",
            f"Prix moyen marché (AutoScout): {int(round(context['prix_marche']))} €" if pd.notna(context['prix_marche']) else "Prix moyen marché: —",
            f"Ton prix d'achat: {int(round(context['prix_user']))} €",
            f"ROI mensuel: {context['roi']:.3f} x" if pd.notna(context['roi']) else "ROI mensuel: —",
            f"Payback: {context['payback']:.1f} mois" if pd.notna(context['payback']) else "Payback: —",
            f"Rentabilité 5 ans: {context['renta5']:.2f} x" if pd.notna(context['renta5']) else "Rentabilité 5 ans: —",
            f"Présence marché (annonces): {int(context['presence'])}",
            f"Compatibilité (année): {'Compatible ✅' if context['compatible']==1 else 'Non compatible ❌' if context['compatible']==0 else '—'}",
        ]
        plt.text(0.03, 0.92, title, fontsize=18, fontweight='bold')
        plt.text(0.03, 0.76, "\n".join(lines), fontsize=12)
        # petit tableau de prix
        table = [
            ["min", f"{int(round(context['pmin']))} €"] if pd.notna(context['pmin']) else ["min","—"],
            ["médian", f"{int(round(context['pmed']))} €"] if pd.notna(context['pmed']) else ["médian","—"],
            ["moyen", f"{int(round(context['pmean']))} €"] if pd.notna(context['pmean']) else ["moyen","—"],
            ["max", f"{int(round(context['pmax']))} €"] if pd.notna(context['pmax']) else ["max","—"],
        ]
        the_table = plt.table(cellText=table, colLabels=["Référence", "€"], loc='lower left', cellLoc='left', colLoc='left', bbox=[0.03, 0.05, 0.25, 0.25])
        the_table.auto_set_font_size(False); the_table.set_fontsize(10)
        pdf.savefig(); plt.close()

        # Page 2 — règle de prix
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        plt.close(fig)  # on utilisera la même fonction que dans l'app mais en mode fig séparé
        # re-créer un ruler spécifique pour le PDF
        fig2, ax2 = plt.subplots(figsize=(11.69, 4))
        xs = [x for x in [context["pmin"], context["pmed"], context["pmean"], context["pmax"]] if pd.notna(x)]
        if xs:
            lo, hi = min(xs + [context["prix_user"]]), max(xs + [context["prix_user"]])
            pad = max(300, 0.05*(hi - lo))
            ax2.hlines(1, lo - pad, hi + pad, linewidth=12, alpha=0.2)
            labels = [("min", context["pmin"]), ("médian", context["pmed"]), ("moyen", context["pmean"]), ("max", context["pmax"])]
            y_above, y_below, toggle = 1.10, 0.84, True
            for name, val in labels:
                if pd.notna(val):
                    ax2.plot([val], [1], marker="o")
                    ytxt = y_above if toggle else y_below
                    ax2.annotate(f"{name}\n{int(round(val))}€", (val,1), xytext=(val, ytxt),
                                 textcoords="data", ha="center", fontsize=11)
                    toggle = not toggle
            ax2.plot([context["prix_user"]], [1], marker="D")
            ax2.annotate(f"ton prix\n{int(round(context['prix_user']))}€", (context["prix_user"],1), xytext=(context["prix_user"], 1.24),
                         textcoords="data", ha="center", fontsize=12, fontweight="bold")
            ax2.set_ylim(0.7,1.35); ax2.set_yticks([]); ax2.set_xlabel("Référence de prix AutoScout (€)")
        else:
            ax2.text(0.5,0.5,"Références de prix indisponibles.", ha="center", va="center"); ax2.axis("off")
        pdf.savefig(fig2); plt.close(fig2)

    return buf.getvalue()

# === APP ===
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(PAGE_TITLE)
st.caption("Basé sur l’analyse croisée Getaround × AutoScout24 + statut de compatibilité agrégé.")

df = load_analysis(ANALYSIS_CSV)
compat_df = load_compat_status(COMPAT_STATUS_CSV)

# Barre d’options globales
left, right = st.columns([3,2])
with left:
    lez_only = st.checkbox("Filtrer LEZ-friendly (Essence & année ≥ 2020)", value=False)
with right:
    if "ville" in df.columns and df["ville"].replace("", np.nan).notna().any():
        villes = ["Toutes"] + sorted(df["ville"].dropna().unique().tolist())
        pick_ville = st.selectbox("Ville", villes, index=0)
    else:
        pick_ville = "Toutes"

# Appliquer filtres globaux
df_filt = df.copy()
if pick_ville != "Toutes":
    df_filt = df_filt[df_filt["ville"] == pick_ville]
if lez_only:
    df_filt = df_filt[(df_filt["carburant_n"].str.contains("essence")) & (df_filt["annee_i"] >= 2020)]
if df_filt.empty:
    st.warning("Aucune donnée après application des filtres Ville/LEZ. Filtre(s) ignoré(s) temporairement.")
    df_filt = df.copy()

# Sélecteurs hiérarchiques
def safe_unique(series):
    vals = sorted([v for v in series.dropna().unique().tolist() if str(v).strip()!=""])
    return vals

col1, col2, col3 = st.columns(3)
with col1:
    marques = safe_unique(df_filt["marque"])
    pick_marque = st.selectbox("Marque", marques)
with col2:
    modeles = safe_unique(df_filt.loc[df_filt["marque"]==pick_marque, "modele"])
    pick_modele = st.selectbox("Modèle", modeles)
with col3:
    annees = df_filt.loc[(df_filt["marque"]==pick_marque)&(df_filt["modele"]==pick_modele), "annee_i"]
    annees = sorted(pd.to_numeric(annees, errors="coerce").dropna().astype(int).unique().tolist())
    pick_annee = st.selectbox("Année", annees)

col4, col5 = st.columns(2)
with col4:
    carbus_raw = safe_unique(df_filt.loc[(df_filt["marque"]==pick_marque)&(df_filt["modele"]==pick_modele), "carburant_n"])
    preferred = ["essence","diesel","hybride","electrique"]
    carbus = [c for c in preferred if c in carbus_raw] + [c for c in carbus_raw if c not in preferred]
    if lez_only and "essence" in carbus:
        carbus = ["essence"]
    pick_carb = st.selectbox("Carburant", carbus)
with col5:
    base_tranches = ["0-50 000 km","50-100 000 km","100-150 000 km"]
    tr_presentes = safe_unique(df_filt.loc[
        (df_filt["marque"]==pick_marque)&(df_filt["modele"]==pick_modele)&
        (df_filt["annee_i"]==pick_annee)&(df_filt["carburant_n"]==pick_carb),
        "tranche_km"
    ])
    tranches = list(dict.fromkeys(base_tranches + tr_presentes))
    pick_km = st.selectbox("Tranche kilométrique", tranches)

# Récup ligne ; fallback tranche proche si absente
row = df_filt[
    (df_filt["marque"]==pick_marque)&
    (df_filt["modele"]==pick_modele)&
    (df_filt["annee_i"]==pick_annee)&
    (df_filt["carburant_n"]==pick_carb)&
    (df_filt["tranche_km"]==pick_km)
]
if row.empty:
    candidates = ["50-100 000 km","0-50 000 km","100-150 000 km"] + tr_presentes
    fallback_row, fb_choice = pd.DataFrame(), None
    for tr in candidates:
        tmp = df_filt[
            (df_filt["marque"]==pick_marque)&
            (df_filt["modele"]==pick_modele)&
            (df_filt["annee_i"]==pick_annee)&
            (df_filt["carburant_n"]==pick_carb)&
            (df_filt["tranche_km"]==tr)
        ]
        if not tmp.empty:
            fallback_row, fb_choice = tmp, tr
            break
    if fallback_row.empty:
        st.warning("Combinaison introuvable pour cette clé (même avec fallback).")
        st.stop()
    else:
        row = fallback_row
        st.info(f"La tranche '{pick_km}' n'existe pas pour cette clé. Affichage sur '{fb_choice}'.")
        pick_km = fb_choice

r = row.iloc[0]

# === Statut de compatibilité (lecture du CSV binaire) ===
comp_now = None
if not compat_df.empty:
    subc = compat_df[
        (compat_df["marque"].str.lower() == str(pick_marque).lower()) &
        (compat_df["modele"].str.lower() == str(pick_modele).lower()) &
        (compat_df["annee_i"] == int(pick_annee))
    ]
    if not subc.empty:
        comp_now = int(subc.iloc[0]["compatible"])

st.markdown("---")
subtitle_bits = [pick_marque, pick_modele, str(pick_annee), pick_carb, pick_km]
if pick_ville != "Toutes": subtitle_bits.append(pick_ville)
st.subheader(" — ".join(subtitle_bits))

colA, colB, colC, colD = st.columns(4)
with colA: st.metric("Revenus mensuels (GA)", f"{int(round(r['estimation_eur_f']))} €")
with colB: st.metric("Prix moyen marché (AS)", f"{int(round(r['prix_moyen_final']))} €")
with colC: st.metric("Présence marché", f"{int(r['presence_marche_fuel']) if pd.notna(r['presence_marche_fuel']) else 0}")
with colD:
    label = "—" if comp_now is None else ("Compatible ✅" if comp_now==1 else "Non compatible ❌")
    st.metric("Compatibilité (année)", label)

# Saisie prix & KPI
default_price = int(r["prix_moyen_final"]) if pd.notna(r["prix_moyen_final"]) and r["prix_moyen_final"]>0 else 10000
user_price = st.number_input("Ton prix d’achat (€)", min_value=0, step=250, value=default_price)
roi, payback, renta5 = compute_kpis(float(r["estimation_eur_f"]), float(user_price))

colE, colF, colG = st.columns(3)
with colE: st.metric("ROI mensuel", f"{roi:.3f} x" if pd.notna(roi) else "—")
with colF: st.metric("Payback", f"{payback:.1f} mois" if pd.notna(payback) else "—")
with colG: st.metric("Renta 5 ans", f"{renta5:.2f} x" if pd.notna(renta5) else "—")

# Bande de prix
pmin, pmed, pmean, pmax = r["prix_min_use"], r["prix_median_use"], r["prix_moyen_use"], r["prix_max_use"]
msg, state = price_position(user_price, pmin, pmed, pmean, pmax)
st.info(msg); st.write(state)
plot_price_ruler(user_price, pmin, pmed, pmean, pmax)

# === Export PDF (reprend le statut) ===
st.markdown("### 📄 Export")
ctx = {
    "ville": (r["ville"] if isinstance(r["ville"], str) else ""),
    "marque": pick_marque, "modele": pick_modele, "annee": pick_annee,
    "carburant": pick_carb, "km": pick_km,
    "revenus": float(r["estimation_eur_f"]) if pd.notna(r["estimation_eur_f"]) else np.nan,
    "prix_marche": float(r["prix_moyen_final"]) if pd.notna(r["prix_moyen_final"]) else np.nan,
    "prix_user": float(user_price),
    "roi": roi, "payback": payback, "renta5": renta5,
    "presence": float(r["presence_marche_fuel"]) if pd.notna(r["presence_marche_fuel"]) else 0.0,
    "compatible": comp_now if comp_now is not None else -1,
    "pmin": pmin, "pmed": pmed, "pmean": pmean, "pmax": pmax,
}
if st.button("📄 Exporter cette fiche en PDF"):
    pdf_bytes = generate_pdf_bytes(ctx)
    st.download_button(
        label="Télécharger la fiche PDF",
        data=pdf_bytes,
        file_name=f"fiche_{pick_marque}_{pick_modele}_{pick_annee}_{pick_carb}_{pick_km.replace(' ','')}.pdf",
        mime="application/pdf"
    )

# Détails tabulaires
with st.expander("Détails AutoScout & Getaround"):
    disp_cols = [
        "ville","marque","modele","annee_i","carburant_n","tranche_km",
        "estimation_eur_f",
        "prix_min_w","prix_median_w","prix_moyen_w","prix_max_w",
        "prix_min_estime","prix_median_estime","prix_moyen_estime","prix_max_estime",
        "prix_moyen_final","presence_marche_fuel","presence_marche_total",
        "roi_brut_mensuel","payback_mois","taux_rentabilite","statut"
    ]
    for c in disp_cols:
        if c not in row.columns: row[c] = np.nan
    st.dataframe(row[disp_cols])
