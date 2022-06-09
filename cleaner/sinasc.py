# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import json
import numpy as np
import pandas as pd

from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization


################################
# cleaner
################################


def div(x, y):
    if y == 0:
        return np.nan
    return x / y


def cleaner(df, out_filename):
    print(f"Starting {out_filename}...")

    valid_rows = pd.DataFrame(True, index=df.index, columns=df.columns)
    results = {
        "COMP": {"COMP_REG": {}},
        "ACC": {"ACC_SINT": {}, "RAN_ACC": {}, "ACC_SEMAN": {}},
        "CRED": {"CRED_VAL_DAT": {}},
        "CONS": {"CONS_SEMAN": {}},
        "CURR": {"CURR_UPD": {}},
        "UNI": {"UNI_REG": {}},
    }

    ################################
    # completeness (completude) COMP
    ################################

    # COMP_REG
    for c in df.columns:
        resp = ~df.loc[valid_rows[c], c].isna()
        results["COMP"]["COMP_REG"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # accuracy (acurácia) ACC
    ################################

    def to_numeric(col):
        return pd.to_numeric(col, errors="coerce")

    def to_date(col):
        values = col.astype(int, errors="ignore").astype(str).str.zfill(8)
        return pd.to_datetime(values, format="%d%m%Y", errors="coerce")

    # ACC_SINT #####################
    columns_date = [
        "dt_cadastro",
        "dt_declarac",
        "dt_nasc",
        "dt_nasc_mae",
        "dt_recebim",
        "dt_rec_orig",
        "dt_ult_menst",
    ]
    for c in columns_date:
        values = to_date(df.loc[valid_rows[c], c])
        resp = ~values.isna()
        results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    mun_ibge = pd.read_csv("utils/municipiosIBGE.csv")

    columns_mun_ibge = ["cod_mun_nasc", "cod_mun_natu", "cod_mun_res"]
    for c in columns_mun_ibge:
        values = df.loc[valid_rows[c], c].astype(int, errors="ignore").astype(str)
        resp = values.isin(mun_ibge["id"].astype(int, errors="ignore").astype(str))
        results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    # RAN_ACC ######################
    min_date = pd.to_datetime("01011970", format="%d%m%Y")
    max_date = pd.to_datetime("31122017", format="%d%m%Y")
    for c in ["dt_cadastro", "dt_declarac", "dt_nasc", "dt_recebim", "dt_rec_orig"]:
        values = to_date(df.loc[valid_rows[c], c])
        resp = (values >= min_date) & (values <= max_date)
        results["ACC"]["RAN_ACC"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    for c in ["dt_nasc_mae", "dt_ult_menst"]:
        values = to_date(df.loc[valid_rows[c], c])
        resp = values <= max_date
        results["ACC"]["RAN_ACC"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    # ACC_SEMAN ####################
    c = "idade_mae"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 120
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "idade_pai"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 120
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "sema_gestac"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 42
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "cons_prenat"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 42
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    max_date = pd.to_datetime("31122017", format="%d%m%Y")
    columns_date = [
        "dt_nasc",
        "dt_nasc_mae",
        "dt_ult_menst",
    ]
    for c in columns_date:
        values = to_date(df.loc[valid_rows[c], c])
        resp = ((max_date - values).dt.days / 365) <= 120
        results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "consultas"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 42
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # credibility (credibilidade) CRED
    ################################

    # CRED_VAL_DAT #################
    c = "cod_mun_nasc"
    resp = df.loc[valid_rows[c], c].astype(int, errors="ignore").astype(str).isin(
        mun_ibge.loc[mun_ibge["UF"] == "ES", "id"].astype(str)
    )
    results["CRED"]["CRED_VAL_DAT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # consistency (consistência) CONS
    ################################

    # CONS_SEMAN ###################
    mask = valid_rows.qtd_fil_mort & valid_rows.qtd_fil_vivo & valid_rows.qtd_gest_ant
    r1 = to_numeric(df.loc[mask, "qtd_fil_mort"]) + to_numeric(
        df.loc[mask, "qtd_fil_vivo"]
    ) <= to_numeric(df.loc[mask, "qtd_gest_ant"])
    results["CONS"]["CONS_SEMAN"]["qtd_fil_mort#qtd_fil_vivo#qtd_gest_ant"] = div(
        r1.sum(), mask.sum()
    )

    mask = valid_rows.qtd_part_ces & valid_rows.qtd_part_nor & valid_rows.qtd_gest_ant
    r2 = to_numeric(df.loc[mask, "qtd_part_ces"]) + to_numeric(
        df.loc[mask, "qtd_part_nor"]
    ) <= to_numeric(df.loc[mask, "qtd_gest_ant"])
    results["CONS"]["CONS_SEMAN"]["qtd_part_ces#qtd_part_nor#qtd_gest_ant"] = div(
        r2.sum(), mask.sum()
    )

    mask = valid_rows.id_anomal & valid_rows.cod_anomal
    r3 = to_numeric(df.loc[mask, "id_anomal"] != 2.0) | (
        to_numeric(df.loc[mask, "id_anomal"] == 2.0)
        & df.loc[mask, "cod_anomal"].isnull()
    )
    results["CONS"]["CONS_SEMAN"]["id_anomal#cod_anomal"] = div(r3.sum(), mask.sum())

    mask = valid_rows.dt_nasc & valid_rows.dt_cadastro
    dt_1 = to_date(df.loc[mask, "dt_nasc"])
    dt_2 = to_date(df.loc[mask, "dt_cadastro"])
    r4 = dt_1 <= dt_2
    results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_cadastro"] = div(r4.sum(), mask.sum())

    mask = valid_rows.dt_nasc & valid_rows.dt_ult_menst
    dt_1 = to_date(df.loc[mask, "dt_ult_menst"])
    dt_2 = to_date(df.loc[mask, "dt_nasc"])
    r5 = dt_1 <= dt_2
    results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_ult_menst"] = div(r5.sum(), mask.sum())

    mask = valid_rows.dt_nasc & valid_rows.dt_rec_orig
    dt_1 = to_date(df.loc[mask, "dt_nasc"])
    dt_2 = to_date(df.loc[mask, "dt_rec_orig"])
    r6 = dt_1 <= dt_2
    results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_rec_orig"] = div(r6.sum(), mask.sum())

    mask = valid_rows.dt_nasc & valid_rows.dt_recebim
    dt_1 = to_date(df.loc[mask, "dt_nasc"])
    dt_2 = to_date(df.loc[mask, "dt_recebim"])
    r7 = dt_1 <= dt_2
    results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_recebim"] = div(r7.sum(), mask.sum())

    mask = valid_rows.dt_nasc & valid_rows.dt_nasc_mae
    dt_1 = to_date(df.loc[mask, "dt_nasc_mae"])
    dt_2 = to_date(df.loc[mask, "dt_nasc"])
    r8 = dt_1 <= dt_2
    results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_nasc_mae"] = div(r8.sum(), mask.sum())

    mask = valid_rows.loc_nasc & valid_rows.cod_estab
    r9 = (
        to_numeric(df.loc[mask, "loc_nasc"] == 1.0)
        | to_numeric(df.loc[mask, "loc_nasc"] == 2.0)
    ) & (~df.loc[mask, "cod_estab"].isnull())
    results["CONS"]["CONS_SEMAN"]["loc_nasc#cod_estab"] = div(r9.sum(), mask.sum())

    valid_rows.loc[r1.loc[~r1].index, "qtd_fil_mort"] = False
    valid_rows.loc[r1.loc[~r1].index, "qtd_fil_vivo"] = False
    valid_rows.loc[(r1 & r2).loc[~(r1 & r2)].index, "qtd_gest_ant"] = False
    valid_rows.loc[r2.loc[~r2].index, "qtd_part_ces"] = False
    valid_rows.loc[r2.loc[~r2].index, "qtd_part_nor"] = False
    valid_rows.loc[r3.loc[~r3].index, "id_anomal"] = False
    valid_rows.loc[r3.loc[~r3].index, "cod_anomal"] = False
    valid_rows.loc[
        (r4 & r5 & r6 & r7 & r8).loc[~(r4 & r5 & r6 & r7 & r8)].index, "dt_nasc"
    ] = False
    valid_rows.loc[r4.loc[~r4].index, "dt_cadastro"] = False
    valid_rows.loc[r5.loc[~r5].index, "dt_ult_menst"] = False
    valid_rows.loc[r6.loc[~r6].index, "dt_rec_orig"] = False
    valid_rows.loc[r7.loc[~r7].index, "dt_recebim"] = False
    valid_rows.loc[r8.loc[~r8].index, "dt_nasc_mae"] = False
    valid_rows.loc[r9.loc[~r9].index, "loc_nasc"] = False
    valid_rows.loc[r9.loc[~r9].index, "cod_estab"] = False

    ################################
    # currentness (atualidade) CURR
    ################################

    # CURR_UPD #####################
    values_dt_recebim = to_date(df.loc[valid_rows["dt_recebim"], "dt_recebim"])
    for c in ["dt_cadastro", "dt_declarac", "dt_nasc"]:
        mask = valid_rows[c] & valid_rows.dt_recebim
        values = to_date(df.loc[mask, c])
        resp = (values_dt_recebim.loc[mask] - values).dt.days <= 365
        results["CURR"]["CURR_UPD"][c] = div(resp.sum(), mask.sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # uniqueness (unicidade) UNI
    ################################

    # UNI_REG ######################

    ################################
    # finaly
    ################################

    final = {
        "COMP": np.mean(list(results["COMP"]["COMP_REG"].values())),
        "ACC": np.prod(
            [
                np.mean(list(results["ACC"]["ACC_SINT"].values())),
                np.mean(list(results["ACC"]["RAN_ACC"].values())),
                np.mean(list(results["ACC"]["ACC_SEMAN"].values())),
            ]
        ),
        "CRED": np.mean(list(results["CRED"]["CRED_VAL_DAT"].values())),
        "CONS": np.mean(list(results["CONS"]["CONS_SEMAN"].values())),
        "CURR": np.mean(list(results["CURR"]["CURR_UPD"].values())),
        "UNI": -1,
    }

    with open(out_filename, "w") as f:
        json.dump(final, f, indent=4, sort_keys=False)


################################
# LGPD columns
################################

LGPD_COLUMNS = [
    "loc_nasc",
    "cod_mun_nasc",
    "idade_mae",
    "est_civ_mae",
    "esc_mae",
    "qtd_fil_vivo",
    "qtd_fil_mort",
    "cod_mun_res",
    "gestacao",
    "gravidez",
    "parto",
    "consultas",
    "dt_nasc",
    "sexo",
    "apgar_1",
    "apgar_5",
    "raca_cor",
    "peso",
    "cod_anomal",
    "cod_estab",
    "cod_ocup_mae",
    "id_anomal",
    "cod_bai_nasc",
    "cod_bai_res",
    "uf_inform",
    "hora_nasc",
    "dt_cadastro",
    "dt_recebim",
    "origem",
    "cod_cart",
    "num_reg_cart",
    "dt_reg_cart",
    "cod_pais_res",
    "numero_lote",
    "versao_sist",
    "dif_data",
    "dt_rec_orig",
    "natural_mae",
    "cod_mun_natu",
    "seri_esc_mae",
    "dt_nasc_mae",
    "raca_cor_mae",
    "qtd_gest_ant",
    "qtd_part_nor",
    "qtd_part_ces",
    "idade_pai",
    "dt_ult_menst",
    "sema_gestac",
    "tp_met_estim",
    "cons_prenat",
    "mes_prenat",
    "tp_apresent",
    "st_trab_part",
    "st_ces_parto",
    "tp_robson",
    "std_nepidem",
    "std_nova",
    "raca_cor_rn",
    "raca_cor_n",
    "esc_mae_2010",
    "cod_mun_cart",
    "cod_uf_natu",
    "tp_nasc_assi",
    "esc_mae_agr_1",
    "dt_rec_orig_a",
    "tp_func_resp",
    "td_doc_resp",
    "dt_declarac",
    "par_idade",
    "kotelchuck",
]

rules = {
    "cod_mun_nasc": {"type": "crop", "start": 0, "stop": 2},
    "idade_mae": {"type": "hist", "nbins": 5},
    "qtd_fil_vivo": {"type": "hist", "nbins": 3},
    "qtd_fil_mort": {"type": "hist", "nbins": 3},
    "cod_mun_res": {"type": "crop", "start": 0, "stop": 2},
    "cod_mun_natu": {"type": "crop", "start": 0, "stop": 2},
    "qtd_gest_ant": {"type": "hist", "nbins": 3},
    "qtd_part_nor": {"type": "hist", "nbins": 3},
    "qtd_part_ces": {"type": "hist", "nbins": 3},
    "idade_pai": {"type": "hist", "nbins": 5},
    "sema_gestac": {"type": "hist", "nbins": 5},
    "cons_prenat": {"type": "hist", "nbins": 3},
}


################################
# run
################################

df = pd.read_parquet("datasets/sinasc.parquet")
cleaner(df, "output/sinasc_raw.json")
cleaner(Supression.anonymize(df, LGPD_COLUMNS), "output/sinasc_supression.json")
cleaner(Randomization.anonymize(df, LGPD_COLUMNS), "output/sinasc_randomization.json")
cleaner(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    "output/sinasc_generalization.json",
)
cleaner(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    "output/sinasc_pseudoanonymization.json",
)
