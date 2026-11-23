import json
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _salva_grafico(fig, filename):
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, filename)
    try:
        fig.savefig(target, dpi=150, bbox_inches="tight")
        return target
    except OSError:
        fallback = os.path.join(tempfile.gettempdir(), filename)
        fig.savefig(fallback, dpi=150, bbox_inches="tight")
        return fallback
    finally:
        plt.close(fig)


def assegna_priorita(row):
    if row["Shipment_Status"] == "Delayed" and row["Logistics_Delay_Reason"] != "Non specificata":
        return "P1 Azione operativa immediata"
    if row["Logistics_Delay_Reason"] == "Non specificata" or (
        row["Shipment_Status"] == "Delivered" and row["Logistics_Delay"] == 1
    ):
        return "P2 Correzione buco dati e processo"
    return "P3 Ottimizzazione monitoraggio"


def audit_qualita_dati(log: pd.DataFrame) -> dict:
    attesa_tot = max(float(log["Waiting_Time"].sum()), 1.0)
    mask_non_spec = log["Logistics_Delay_Reason"] == "Non specificata"
    mask_del_late = (log["Shipment_Status"] == "Delivered") & (log["Logistics_Delay"] == 1)
    mask_anomalia_tot = mask_non_spec | mask_del_late

    attesa_non_spec = float(log.loc[mask_non_spec, "Waiting_Time"].sum())
    attesa_del_late = float(log.loc[mask_del_late, "Waiting_Time"].sum())
    attesa_p2_unita = float(log.loc[mask_anomalia_tot, "Waiting_Time"].sum())

    return {
        "record_non_specificati": int(mask_non_spec.sum()),
        "attesa_non_specificati": int(attesa_non_spec),
        "record_delivered_in_ritardo": int(mask_del_late.sum()),
        "attesa_delivered_in_ritardo": int(attesa_del_late),
        "record_p2_unici": int(mask_anomalia_tot.sum()),
        "attesa_p2_unita": int(attesa_p2_unita),
        "quota_attesa_cieca_pct": round(attesa_p2_unita / attesa_tot * 100.0, 2),
    }


def analizza_buchi_per_mezzo(log: pd.DataFrame) -> pd.DataFrame:
    if "Asset_ID" not in log.columns:
        return pd.DataFrame()
    df = log.copy()
    df["is_non_spec"] = (df["Logistics_Delay_Reason"] == "Non specificata").astype(int)
    df["is_del_late"] = ((df["Shipment_Status"] == "Delivered") & (df["Logistics_Delay"] == 1)).astype(int)
    df["is_p2"] = ((df["is_non_spec"] == 1) | (df["is_del_late"] == 1)).astype(int)
    df["attesa_p2"] = np.where(df["is_p2"] == 1, df["Waiting_Time"], 0.0)

    buchi = (
        df.groupby("Asset_ID", as_index=True, observed=False)
        .agg(
            eventi_totali=("Waiting_Time", "count"),
            record_non_spec=("is_non_spec", "sum"),
            record_delivered_ritardo=("is_del_late", "sum"),
            record_p2=("is_p2", "sum"),
            attesa_p2_totale=("attesa_p2", "sum"),
            attesa_totale=("Waiting_Time", "sum"),
        )
        .reset_index()
    )
    buchi = buchi[buchi["eventi_totali"] > 0].sort_values("attesa_p2_totale", ascending=False).reset_index(drop=True)
    buchi["quota_attesa_p2_pct"] = np.where(
        buchi["attesa_totale"] > 0, buchi["attesa_p2_totale"] / buchi["attesa_totale"] * 100.0, 0.0
    )
    return buchi


def analizza_picco_pomeridiano_mezzi(log: pd.DataFrame) -> pd.DataFrame:
    if "Asset_ID" not in log.columns:
        return pd.DataFrame()
    pom = log[log["fascia_oraria"] == "Pomeriggio (1217)"].copy()
    if pom.empty:
        return pd.DataFrame()
    attesa_pom = max(float(pom["Waiting_Time"].sum()), 1.0)
    res = (
        pom.groupby("Asset_ID", as_index=True, observed=False)
        .agg(
            eventi_pom=("Waiting_Time", "count"),
            ritardi_pom=("Logistics_Delay", "sum"),
            attesa_pom=("Waiting_Time", "sum"),
            attesa_media_pom=("Waiting_Time", "mean"),
        )
        .reset_index()
    )
    res = res[res["eventi_pom"] > 0].sort_values("attesa_pom", ascending=False).reset_index(drop=True)
    res["tasso_ritardo_pom_pct"] = res["ritardi_pom"] / res["eventi_pom"] * 100.0
    res["quota_attesa_pom_pct"] = res["attesa_pom"] / attesa_pom * 100.0
    return res


def costruisci_matrice_mezzo_fascia(log: pd.DataFrame) -> pd.DataFrame:
    if "Asset_ID" not in log.columns:
        return pd.DataFrame()
    pivot = log.pivot_table(
        index="Asset_ID",
        columns="fascia_oraria",
        values="Waiting_Time",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    pivot["totale_mezzo"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("totale_mezzo", ascending=False).reset_index()
    return pivot


def analizza_mezzi(log: pd.DataFrame) -> pd.DataFrame:
    if "Asset_ID" not in log.columns:
        return pd.DataFrame()
    attesa_tot = max(float(log["Waiting_Time"].sum()), 1.0)
    mezzi = (
        log.groupby("Asset_ID", as_index=True, observed=False)
        .agg(
            eventi=("Waiting_Time", "count"),
            ritardi=("Logistics_Delay", "sum"),
            attesa_totale=("Waiting_Time", "sum"),
            attesa_media=("Waiting_Time", "mean"),
        )
        .reset_index()
    )
    mezzi = mezzi[mezzi["eventi"] > 0].sort_values("attesa_totale", ascending=False).reset_index(drop=True)
    mezzi["tasso_ritardo_pct"] = mezzi["ritardi"] / mezzi["eventi"] * 100.0
    mezzi["quota_attesa_pct"] = mezzi["attesa_totale"] / attesa_tot * 100.0
    mezzi["cumulata_pct"] = mezzi["quota_attesa_pct"].cumsum()
    return mezzi


def analizza_fasce_orarie(log: pd.DataFrame) -> pd.DataFrame:
    attesa_tot = max(float(log["Waiting_Time"].sum()), 1.0)
    fasce = (
        log.groupby("fascia_oraria", as_index=True, observed=False)
        .agg(
            eventi=("Waiting_Time", "count"),
            ritardi=("Logistics_Delay", "sum"),
            attesa_totale=("Waiting_Time", "sum"),
            attesa_media=("Waiting_Time", "mean"),
        )
        .reset_index()
    )
    fasce["tasso_ritardo_pct"] = np.where(fasce["eventi"] > 0, fasce["ritardi"] / fasce["eventi"] * 100.0, 0.0)
    fasce["quota_attesa_pct"] = fasce["attesa_totale"] / attesa_tot * 100.0
    return fasce


def analizza_efficienza_marginale(df_pol: pd.DataFrame, min_marg_netto: float = 8.0) -> pd.DataFrame:
    df = df_pol.sort_values("unita_extra").copy().reset_index(drop=True)
    esposizione_iniziale = max(float(df["esposizione_post_eur"].max()), 1.0)

    df["delta_unita"] = df["unita_extra"].diff().fillna(0.0)
    df["delta_costo_eur"] = df["costo_eur"].diff().fillna(0.0)
    df["delta_lordo_eur"] = df["valore_lordo_protetto_eur"].diff().fillna(0.0)
    df["delta_netto_eur"] = df["valore_netto_eur"].diff().fillna(0.0)

    df["roi_lordo_totale"] = np.where(
        df["costo_eur"] > 0, df["valore_lordo_protetto_eur"] / df["costo_eur"], np.nan
    )
    df["costo_su_valore_lordo_pct"] = np.where(
        df["valore_lordo_protetto_eur"] > 0, df["costo_eur"] / df["valore_lordo_protetto_eur"] * 100.0, 0.0
    )
    df["riduzione_esposizione_pct"] = df["valore_lordo_protetto_eur"] / esposizione_iniziale * 100.0
    df["marg_lordo_per_eur"] = np.where(
        df["delta_costo_eur"] > 0, df["delta_lordo_eur"] / df["delta_costo_eur"], np.nan
    )
    df["marg_netto_per_eur"] = np.where(
        df["delta_costo_eur"] > 0, df["delta_netto_eur"] / df["delta_costo_eur"], np.nan
    )
    df["costo_marg_per_1m_lordo_eur"] = np.where(
        df["delta_lordo_eur"] > 0, df["delta_costo_eur"] / (df["delta_lordo_eur"] / 1e6), np.nan
    )
    df["copertura_netta_su_esposizione_post"] = np.where(
        df["esposizione_post_eur"] > 0, df["valore_netto_eur"] / df["esposizione_post_eur"], np.nan
    )
    df["rispetta_soglia_marginale"] = np.where(
        df["costo_eur"] == 0, True, df["marg_netto_per_eur"] >= float(min_marg_netto)
    )
    return df


def ottimizza_policy_budget(df_pol: pd.DataFrame, budget_max_eur: float = 300000.0) -> dict:
    if df_pol is None or df_pol.empty:
        return {
            "budget_max_eur": float(budget_max_eur),
            "policy_scelta": None,
            "costo_eur": 0.0,
            "valore_netto_eur": 0.0,
            "budget_residuo_eur": float(budget_max_eur),
        }
    amm = df_pol[df_pol["costo_eur"] <= float(budget_max_eur)].sort_values("valore_netto_eur", ascending=False)
    if amm.empty:
        return {
            "budget_max_eur": float(budget_max_eur),
            "policy_scelta": None,
            "costo_eur": 0.0,
            "valore_netto_eur": 0.0,
            "budget_residuo_eur": float(budget_max_eur),
        }
    scelta = amm.iloc[0]
    return {
        "budget_max_eur": float(budget_max_eur),
        "policy_scelta": scelta["policy"],
        "costo_eur": round(float(scelta["costo_eur"]), 2),
        "valore_netto_eur": round(float(scelta["valore_netto_eur"]), 2),
        "budget_residuo_eur": round(float(budget_max_eur) - float(scelta["costo_eur"]), 2),
    }


def valuta_sensibilita_soglie(df_pol: pd.DataFrame, soglie=(5.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0)) -> pd.DataFrame:
    righe = []
    for s in soglie:
        amm = df_pol[(df_pol["costo_eur"] > 0) & (df_pol["marg_netto_per_eur"] >= float(s))]
        if not amm.empty:
            scelta = amm.sort_values("unita_extra", ascending=False).iloc[0]
            righe.append(
                {
                    "soglia_marg_netto_eur": float(s),
                    "policy_selezionata": scelta["policy"],
                    "unita_extra": float(scelta["unita_extra"]),
                    "costo_eur": float(scelta["costo_eur"]),
                    "valore_netto_eur": float(scelta["valore_netto_eur"]),
                    "marg_netto_effettivo": float(scelta["marg_netto_per_eur"]),
                }
            )
        else:
            base = df_pol.sort_values("unita_extra", ascending=True).iloc[0]
            righe.append(
                {
                    "soglia_marg_netto_eur": float(s),
                    "policy_selezionata": base["policy"],
                    "unita_extra": 0.0,
                    "costo_eur": 0.0,
                    "valore_netto_eur": 0.0,
                    "marg_netto_effettivo": np.nan,
                }
            )
    return pd.DataFrame(righe)


def stima_saturazione_marginale(df_pol: pd.DataFrame) -> dict:
    valid = df_pol[df_pol["delta_costo_eur"] > 0].copy()
    if len(valid) < 2:
        return {"stima_disponibile": False}
    x = valid["costo_eur"].to_numpy(dtype=float)
    y = valid["marg_netto_per_eur"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    costo_zero_margine = float(-intercept / slope) if slope < 0 else None
    return {
        "stima_disponibile": True,
        "pendenza_margine_per_eur": round(float(slope), 8),
        "intercetta_margine": round(float(intercept), 4),
        "costo_stimato_margine_zero_eur": round(costo_zero_margine, 2) if costo_zero_margine else None,
    }


def costruisci_hotspot_pareto(log: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    attesa_globale = max(float(log["Waiting_Time"].sum()), 1.0)
    has_asset = "Asset_ID" in log.columns
    group_cols = ["Asset_ID", "fascia_oraria", "Logistics_Delay_Reason"] if has_asset else [
        "fascia_oraria",
        "Logistics_Delay_Reason",
    ]

    hotspots = (
        log.groupby(group_cols, as_index=True, observed=False)
        .agg(
            eventi=("Waiting_Time", "count"),
            ritardi=("Logistics_Delay", "sum"),
            attesa_totale=("Waiting_Time", "sum"),
            attesa_media=("Waiting_Time", "mean"),
        )
        .reset_index()
    )
    hotspots = hotspots[hotspots["eventi"] > 0].copy()
    hotspots["tasso_ritardo_pct"] = np.where(
        hotspots["eventi"] > 0, hotspots["ritardi"] / hotspots["eventi"] * 100.0, 0.0
    )
    hotspots["quota_attesa_pct"] = hotspots["attesa_totale"] / attesa_globale * 100.0
    hotspots["killer_score"] = hotspots["quota_attesa_pct"] * (1.0 + hotspots["tasso_ritardo_pct"] / 100.0)

    hotspots = (
        hotspots.sort_values("attesa_totale", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    hotspots["cumulata_pct"] = hotspots["quota_attesa_pct"].cumsum()
    return hotspots


def simula_combinazione_coda_buffer(
    df_pol: pd.DataFrame,
    unita_coda_risparmiate: float,
    attesa_totale_rete: float,
    costo_unit_coda: float = 15.0,
) -> pd.DataFrame:
    sim = df_pol.copy()
    quota_riduzione_attesa = unita_coda_risparmiate / max(attesa_totale_rete, 1.0)
    valore_diretto_coda = unita_coda_risparmiate * costo_unit_coda

    sim["valore_diretto_coda_eur"] = valore_diretto_coda
    sim["sollievo_ritardo_esposizione_eur"] = (
        sim["esposizione_post_eur"] * quota_riduzione_attesa
    )
    sim["valore_netto_combinato_stimato_eur"] = (
        sim["valore_netto_eur"]
        + sim["valore_diretto_coda_eur"]
        + sim["sollievo_ritardo_esposizione_eur"]
    )
    return sim


def simula_sensibilita_taglio_coda(unita_coda: float, costo_unit: float = 15.0) -> pd.DataFrame:
    righe = []
    for pct in (0.10, 0.20, 0.30, 0.40, 0.50):
        unita_risp = unita_coda * pct
        righe.append(
            {
                "taglio_coda_pct": round(pct * 100.0, 1),
                "unita_risparmiate": round(unita_risp, 1),
                "costo_unit_eur": float(costo_unit),
                "valore_recuperato_eur": round(unita_risp * float(costo_unit), 2),
            }
        )
    return pd.DataFrame(righe)


def simula_scenari_operativi(unita_coda: float, attesa_p2: float) -> pd.DataFrame:
    scenari = [
        {"scenario": "Conservativo", "taglio_coda_pct": 0.15, "recupero_p2_pct": 0.25, "costo_unit_eur": 12.0},
        {"scenario": "Base", "taglio_coda_pct": 0.30, "recupero_p2_pct": 0.50, "costo_unit_eur": 15.0},
        {"scenario": "Aggressivo", "taglio_coda_pct": 0.45, "recupero_p2_pct": 0.75, "costo_unit_eur": 20.0},
    ]
    righe = []
    for sc in scenari:
        unita_coda_rec = unita_coda * sc["taglio_coda_pct"]
        unita_p2_visibili = attesa_p2 * sc["recupero_p2_pct"]
        valore_coda_eur = unita_coda_rec * sc["costo_unit_eur"]
        righe.append(
            {
                "scenario": sc["scenario"],
                "taglio_coda_pct": round(sc["taglio_coda_pct"] * 100.0, 1),
                "unita_coda_recuperate": round(unita_coda_rec, 1),
                "recupero_p2_pct": round(sc["recupero_p2_pct"] * 100.0, 1),
                "unita_p2_riportate_a_visibilita": round(unita_p2_visibili, 1),
                "costo_unit_eur": sc["costo_unit_eur"],
                "valore_diretto_coda_eur": round(valore_coda_eur, 2),
            }
        )
    return pd.DataFrame(righe)


def simula_montecarlo_coda(unita_coda: float, n_sim: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    riduzione_pct = rng.triangular(0.15, 0.30, 0.45, size=n_sim)
    costo_unit = rng.triangular(10.0, 15.0, 25.0, size=n_sim)
    valori = (unita_coda * riduzione_pct) * costo_unit
    p10, p50, p90 = np.percentile(valori, [10, 50, 90])
    return {
        "simulazioni": int(n_sim),
        "p10_eur": round(float(p10), 2),
        "p50_mediana_eur": round(float(p50), 2),
        "p90_eur": round(float(p90), 2),
        "media_eur": round(float(np.mean(valori)), 2),
    }


def estrai_piano_azione(log: pd.DataFrame, top_per_coda: int = 10) -> pd.DataFrame:
    cols = [
        c
        for c in ["Asset_ID", "fascia_oraria", "Logistics_Delay_Reason", "Shipment_Status"]
        if c in log.columns
    ]
    agg_df = (
        log.groupby(cols + ["Priorita"], as_index=True, observed=False)
        .agg(
            eventi=("Waiting_Time", "count"),
            ritardi=("Logistics_Delay", "sum"),
            attesa_totale=("Waiting_Time", "sum"),
            attesa_media=("Waiting_Time", "mean"),
        )
        .reset_index()
    )
    agg_df = agg_df[agg_df["eventi"] > 0].sort_values("attesa_totale", ascending=False)
    p1 = agg_df[agg_df["Priorita"].str.startswith("P1")].head(top_per_coda).copy()
    p2 = agg_df[agg_df["Priorita"].str.startswith("P2")].head(top_per_coda).copy()
    return pd.concat([p1, p2], ignore_index=True)


def _crea_dashboard_vincoli(
    hotspots: pd.DataFrame,
    df_pol: pd.DataFrame | None,
    df_soglie: pd.DataFrame | None,
    mezzi: pd.DataFrame,
    fasce: pd.DataFrame,
):
    fig, axes = plt.subplots(2, 3, figsize=(21.0, 9.8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    if not hotspots.empty:
        labels = [
            f"{r.get('Asset_ID', 'ALL')} | {str(r['fascia_oraria'])[:3]} | {str(r['Logistics_Delay_Reason'])[:10]}"
            for _, r in hotspots.iterrows()
        ]
        y_pos = np.arange(len(labels))
        ax1.barh(y_pos, hotspots["attesa_totale"], color="#1f4e78")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Attesa totale (unità)")
        ax1.set_title("Top Hotspot Operativi")
        ax1.grid(axis="x", linestyle="--", alpha=0.4)

    if df_pol is not None and not df_pol.empty:
        x_pos = np.arange(len(df_pol))
        netto_m = df_pol["valore_netto_eur"] / 1e6
        ax2.bar(x_pos, netto_m, width=0.5, color="#2e6f40", label="Valore Netto (M EUR)")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df_pol["policy"], rotation=15, fontsize=8.5)
        ax2.set_ylabel("Valore Netto (Milioni EUR)")
        ax2.set_title("Policy Buffer: Valore Netto vs Margine")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        ax2_twin = ax2.twinx()
        valid_mask = df_pol["marg_netto_per_eur"].notna()
        ax2_twin.plot(
            x_pos[valid_mask],
            df_pol.loc[valid_mask, "marg_netto_per_eur"],
            color="#c05621",
            marker="o",
            linewidth=2,
        )
        ax2_twin.set_ylabel("EUR Netti / EUR Extra")

    if df_soglie is not None and not df_soglie.empty:
        x_s = np.arange(len(df_soglie))
        ax3.bar(x_s, df_soglie["valore_netto_eur"] / 1e6, width=0.5, color="#2b6cb0")
        ax3.set_xticks(x_s)
        ax3.set_xticklabels(
            [f">={s:.0f}\n({p[:6]})" for s, p in zip(df_soglie["soglia_marg_netto_eur"], df_soglie["policy_selezionata"])],
            fontsize=8,
        )
        ax3.set_ylabel("Valore Netto Scelto (M EUR)")
        ax3.set_title("Sensibilità Soglia Marginale")
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

    if not mezzi.empty:
        top_m = mezzi.head(8)
        x_m = np.arange(len(top_m))
        ax4.bar(x_m, top_m["attesa_totale"], width=0.55, color="#6b46c1")
        ax4.set_xticks(x_m)
        ax4.set_xticklabels(top_m["Asset_ID"].astype(str), rotation=20, fontsize=8.5)
        ax4.set_ylabel("Attesa totale (unità)")
        ax4.set_title("Top Mezzi per Attesa Totale")
        ax4.grid(axis="y", linestyle="--", alpha=0.4)

    if not fasce.empty:
        x_f = np.arange(len(fasce))
        ax5.bar(x_f, fasce["quota_attesa_pct"], width=0.55, color="#b7791f")
        ax5.set_xticks(x_f)
        ax5.set_xticklabels(fasce["fascia_oraria"].astype(str), rotation=15, fontsize=8.5)
        ax5.set_ylabel("Quota attesa (%)")
        ax5.set_title("Quota Attesa per Fascia Oraria")
        ax5.grid(axis="y", linestyle="--", alpha=0.4)

    if df_pol is not None and not df_pol.empty:
        x_r = np.arange(len(df_pol))
        ax6.bar(x_r, df_pol["riduzione_esposizione_pct"], width=0.55, color="#2c7a7b")
        ax6.set_xticks(x_r)
        ax6.set_xticklabels(df_pol["policy"], rotation=15, fontsize=8.5)
        ax6.set_ylabel("Riduzione esposizione (%)")
        ax6.set_title("Abbattimento Esposizione Iniziale (%)")
        ax6.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return _salva_grafico(fig, "motore_vincoli_dashboard.png")


def _scrivi_brief_markdown(
    out_dir: str,
    riepilogo: dict,
    hotspots: pd.DataFrame,
    df_soglie: pd.DataFrame | None,
    df_scenari: pd.DataFrame,
):
    top_hs = hotspots.iloc[0].to_dict() if not hotspots.empty else {}
    mc = riepilogo.get("montecarlo_coda_eur", {})
    audit = riepilogo.get("audit_qualita_dati", {})
    sat = riepilogo.get("saturazione_marginale", {})
    bud = riepilogo.get("ottimizzazione_budget", {})

    lines = [
        "# Bottleneck Engine Brief Esecutivo Automatico",
        "",
        "## 1. Diagnosi Vincoli",
        f"Attesa totale: {riepilogo['attesa_totale']:,} unità su {riepilogo['eventi_totali']:,} eventi ({riepilogo['tasso_ritardo_pct']:.1f}% con ritardo attivo).".replace(",", "."),
        f"Picco pomeridiano (1217): tasso di ritardo al {riepilogo['picco_pomeridiano_pct']:.1f}%.",
        f"Concentrazione flotta: top 5 mezzi al {riepilogo['top5_mezzi_quota_pct']:.1f}% dell'attesa totale.",
        f"Buchi informativi P2: {audit.get('record_non_specificati', 0)} cause non specificate e {audit.get('record_delivered_in_ritardo', 0)} consegne Delivered in ritardo ({audit.get('attesa_p2_unita', 0):,} unità, {audit.get('quota_attesa_cieca_pct', 0):.1f}% del totale).".replace(",", "."),
    ]
    if top_hs:
        lines.append(
            f"Top Hotspot: {top_hs.get('Asset_ID', 'N/D')} | {top_hs.get('fascia_oraria', 'N/D')} | {top_hs.get('Logistics_Delay_Reason', 'N/D')} ({int(top_hs.get('attesa_totale', 0)):,} unità, killer score {top_hs.get('killer_score', 0):.2f}).".replace(",", ".")
        )
    lines.extend([
        "",
        "## 2. Coda Critica e Simulazione Monte Carlo",
        f"Soglia p80: {riepilogo['p80_waiting_time']:.1f} | Monte Carlo (5.000 run): p10 = {mc.get('p10_eur', 0):,.0f} EUR | mediana = {mc.get('p50_mediana_eur', 0):,.0f} EUR | p90 = {mc.get('p90_eur', 0):,.0f} EUR.".replace(",", "."),
        "",
        "## 3. Decisione Marginale Buffer e Vincolo di Budget",
        f"Soglia rendimento netto marginale attiva: >= {riepilogo['soglia_margine_netto_eur_per_eur']:.1f} EUR netti per EUR extra.",
        f"Policy selezionata sotto soglia marginale: {riepilogo.get('policy_scelta_sotto_soglia_marginale')}.",
    ])
    if bud.get("policy_scelta"):
        lines.append(
            f"Ottimizzazione budget fisso ({bud['budget_max_eur']:,.0f} EUR): {bud['policy_scelta']} (costo {bud['costo_eur']:,.0f} EUR, netto {bud['valore_netto_eur']:,.0f} EUR, residuo {bud['budget_residuo_eur']:,.0f} EUR).".replace(",", ".")
        )
    if sat.get("stima_disponibile") and sat.get("costo_stimato_margine_zero_eur"):
        lines.append(
            f"Spesa stimata a margine netto nullo: circa {sat['costo_stimato_margine_zero_eur']:,.0f} EUR.".replace(",", ".")
        )
    if df_soglie is not None and not df_soglie.empty:
        lines.extend(["", "## 4. Sensibilità Multi Soglia Marginale"])
        for _, r in df_soglie.iterrows():
            lines.append(
                f"Soglia >= {r['soglia_marg_netto_eur']:.1f} EUR per EUR -> {r['policy_selezionata']} (Netto: {r['valore_netto_eur']:,.0f} EUR).".replace(",", ".")
            )
    if df_scenari is not None and not df_scenari.empty:
        lines.extend(["", "## 5. Scenari Operativi"])
        for _, r in df_scenari.iterrows():
            lines.append(
                f"Scenario {r['scenario']}: taglio coda {r['taglio_coda_pct']:.0f}% ({r['unita_coda_recuperate']:,.0f} unità, valore diretto {r['valore_diretto_coda_eur']:,.0f} EUR) e recupero P2 {r['recupero_p2_pct']:.0f}% ({r['unita_p2_riportate_a_visibilita']:,.0f} unità).".replace(",", ".")
            )

    brief_path = os.path.join(out_dir, "brief_esecutivo.md")
    with open(brief_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return brief_path


def esegui_motore_completo(
    csv_path="data/logistics.csv",
    policy_xlsx="confronto_policy_buffer.xlsx",
    min_marg_netto: float = 8.0,
    budget_max_eur: float = 300000.0,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File non trovato: '{csv_path}'. Controlla che il CSV sia nella cartella indicata."
        )

    log = pd.read_csv(csv_path)
    log["Logistics_Delay_Reason"] = log["Logistics_Delay_Reason"].fillna("Non specificata")
    log["ts"] = pd.to_datetime(log["Timestamp"])
    log["ora"] = log["ts"].dt.hour
    log["fascia_oraria"] = pd.cut(
        log["ora"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Notte (0005)", "Mattina (0611)", "Pomeriggio (1217)", "Sera (1823)"],
    )

    attesa_totale = int(log["Waiting_Time"].sum())
    tasso_ritardo_pct = float(log["Logistics_Delay"].mean() * 100)
    picco_pom_pct = float(
        log[log["fascia_oraria"] == "Pomeriggio (1217)"]["Logistics_Delay"].mean() * 100
    )
    mezzi = analizza_mezzi(log)
    fasce = analizza_fasce_orarie(log)
    buchi_mezzi = analizza_buchi_per_mezzo(log)
    pom_mezzi = analizza_picco_pomeridiano_mezzi(log)
    matrice_mf = costruisci_matrice_mezzo_fascia(log)
    top5_mezzi_quota = float(mezzi.head(5)["quota_attesa_pct"].sum()) if not mezzi.empty else 0.0

    audit = audit_qualita_dati(log)

    print("=== 1. NOTA ESECUTIVA AUTOMATICA ===")
    print(
        f"Attesa totale analizzata: {attesa_totale:,} su {len(log):,} eventi ({tasso_ritardo_pct:.1f}% con ritardo attivo).".replace(
            ",", "."
        )
    )
    print(f"Picco critico pomeridiano (1217): tasso di ritardo al {picco_pom_pct:.1f}%.")
    print(f"Concentrazione flotta: i primi 5 mezzi pesano per il {top5_mezzi_quota:.1f}% dell'attesa totale.")
    print(
        f"Buchi informativi: {audit['record_non_specificati']} cause non specificate e "
        f"{audit['record_delivered_in_ritardo']} consegne Delivered con ritardo "
        f"({audit['attesa_p2_unita']:,} unità d'attesa coinvolte, pari al {audit['quota_attesa_cieca_pct']:.1f}% del totale).\n".replace(
            ",", "."
        )
    )

    log["Priorita"] = log.apply(assegna_