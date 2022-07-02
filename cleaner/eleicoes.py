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


# VER RANDOMIZAÇÃO
def cleaner(df, out_filename=None):
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

    # ACC_SINT #####################

    DICT = {
        "declara_bens": ["S", "N", "NAO DIVULGAVEL"],
        "cargo": [
            "VEREADOR",
            "PREFEITO",
            "VICE-PREFEITO",
            "DEPUTADO ESTADUAL",
            "DEPUTADO FEDERAL",
            "DEPUTADO DISTRITAL",
            "2o SUPLENTE SENADOR",
            "1o SUPLENTE SENADOR",
            "SENADOR",
            "VICE-GOVERNADOR",
            "GOVERNADOR",
            "PRESIDENTE",
            "VICE-PRESIDENTE",
        ],
        "estado_civil": [
            "CASADO(A)",
            "SOLTEIRO(A)",
            "DIVORCIADO(A)",
            "SEPARADO(A) JUDICIALMENTE",
            "VIUVO(A)",
            "NAO INFORMADO",
            "NAO DIVULGAVEL",
        ],
        "genero": ["MASCULINO", "FEMININO", "NAO DIVULGAVEL", "NAO INFORMADO"],
        "grau_instrucao": [
            "ENSINO MEDIO COMPLETO",
            "SUPERIOR COMPLETO",
            "ENSINO FUNDAMENTAL INCOMPLETO",
            "ENSINO FUNDAMENTAL COMPLETO",
            "SUPERIOR INCOMPLETO",
            "MEDIO COMPLETO",
            "ENSINO MEDIO INCOMPLETO",
            "LE E ESCREVE",
            "1O GRAU INCOMPLETO",
            "FUNDAMENTAL INCOMPLETO",
            "2O GRAU COMPLETO",
            "1O GRAU COMPLETO",
            "FUNDAMENTAL COMPLETO",
            "2O GRAU INCOMPLETO",
            "MEDIO INCOMPLETO",
            "NAO INFORMADO",
            "NAO DIVULGAVEL",
            "ANALFABETO",
        ],
        "etnia": [
            "BRANCA",
            "PARDA",
            "PRETA",
            "NAO INFORMADO",
            "AMARELA",
            "INDIGENA",
            "NAO DIVULGAVEL",
        ],
    }
    for c in DICT.keys():
        resp = df.loc[valid_rows[c], c].isin(DICT[c])
        results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "ano"
    values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y", errors="coerce")
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "data_nascimento"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce"
    )
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    DICT_SIZES = {"cpf": 11, "titulo_eleitoral": 12}

    for c in DICT_SIZES.keys():
        resp = df.loc[valid_rows[c], c].astype(str).str.len() == DICT_SIZES[c]
        results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    mun_ibge = pd.read_csv("utils/municipiosIBGE.csv")

    columns_mun_ibge = [
        "sigla_unidade_federativa",
        "sigla_unidade_federativa_nascimento",
    ]
    for c in columns_mun_ibge:
        values = df.loc[valid_rows[c], c].astype(str)
        resp = values.isin(mun_ibge["UF"])
        results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "municipio_nascimento"
    values = df.loc[valid_rows[c], c].astype(str)
    resp = values.str.upper().isin(mun_ibge["nome"].str.upper())
    results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "unidade_eleitoral"
    values = df.loc[valid_rows[c], c].astype(str)
    resp = values.str.upper().isin(mun_ibge["nome"].str.upper())
    results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # RAN_ACC ######################

    min_date = pd.to_datetime("1996-01-01")
    max_date = pd.to_datetime("2020-12-31")

    c = "ano"
    values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y", errors="coerce")
    resp = (values >= min_date) & (values <= max_date)
    results["ACC"]["RAN_ACC"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "data_nascimento"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce"
    )
    resp = values <= max_date
    results["ACC"]["RAN_ACC"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # ACC_SEMAN ####################

    c = "data_nascimento"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce"
    ).dt.to_pydatetime()
    resp = (
        pd.Series(max_date - values, index=df.loc[valid_rows[c], c].index).apply(
            lambda x: x.days
        )
        / 365
    ) <= 120
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    pleitos = pd.read_csv("utils/eleicoes.csv")

    c = "ano"
    resp = (
        pd.to_numeric(df.loc[valid_rows[c], c], errors="coerce")
        .astype(int)
        .isin(pleitos["pleito_ano"].astype(int))
    )
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # credibility (credibilidade) CRED
    ################################

    # CRED_VAL_DAT #################
    c = "unidade_eleitoral"
    resp = (
        df.loc[valid_rows[c], c]
        .astype(str)
        .str.upper()
        .isin(mun_ibge.loc[mun_ibge["UF"] == "ES", "nome"].str.upper())
    )
    results["CRED"]["CRED_VAL_DAT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # consistency (consistência) CONS
    ################################

    # CONS_SEMAN ###################
    mask = (
        valid_rows.sigla_unidade_federativa_nascimento & valid_rows.municipio_nascimento
    )
    if any(mask):
        r1 = (
            df.loc[mask, :]
            .groupby("sigla_unidade_federativa_nascimento")
            .apply(
                lambda x: x["municipio_nascimento"]
                .astype(str)
                .str.upper()
                .isin(
                    mun_ibge.loc[
                        mun_ibge["UF"]
                        == x["sigla_unidade_federativa_nascimento"].iloc[0],
                        "nome",
                    ].str.upper()
                )
            )
        )
        results["CONS"]["CONS_SEMAN"][
            "sigla_unidade_federativa_nascimento#municipio_nascimento"
        ] = div(r1.sum(), mask.sum())
    else:
        results["CONS"]["CONS_SEMAN"][
            "sigla_unidade_federativa_nascimento#municipio_nascimento"
        ] = np.nan

    mask = valid_rows.despesa_maxima_campanha
    r2 = pd.to_numeric(df.loc[mask, "despesa_maxima_campanha"], errors="coerce") >= 0
    results["CONS"]["CONS_SEMAN"]["despesa_maxima_campanha"] = div(r2.sum(), mask.sum())

    ################################
    # currentness (atualidade) CURR
    ################################

    # CURR_UPD #####################

    ################################
    # uniqueness (unicidade) UNI
    ################################

    # UNI_REG ######################
    u = "cpf"
    columns_uni = [
        "data_nascimento",
        "etnia",
        "nacionalidade",
        "nome",
        "municipio_nascimento",
        "sigla_unidade_federativa_nascimento",
        "titulo_eleitoral",
    ]
    for c in columns_uni:
        resp = df.groupby(u)[c].apply(
            lambda x: len(x.index)
            if len(x.loc[valid_rows[c] & valid_rows[u]].unique()) <= 1
            else -len(x.index)
        )
        results["UNI"]["UNI_REG"][c] = div(resp[resp > 0].sum(), abs(resp).sum())
        for p in df[u].unique():
            valid_rows.loc[df.loc[(df[u] == p) & ((resp.get(p) or 1) < 0)].index, c] = False

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
        "CURR": -1,
        "UNI": np.mean(list(results["UNI"]["UNI_REG"].values())),
    }

    if out_filename:
        with open(out_filename, "w") as f:
            json.dump(final, f, indent=4, sort_keys=False)

    return valid_rows


################################
# LGPD columns
################################

LGPD_COLUMNS = [
    "cpf",
    "data_nascimento",
    "declara_bens",
    "cargo",
    "etnia",
    "estado_civil",
    "genero",
    "grau_instrucao",
    "nacionalidade",
    "ocupacao",
    "unidade_eleitoral",
    "despesa_maxima_campanha",
    "email",
    "nome",
    "municipio_nascimento",
    "partido",
    "nome_social",
    "nome_urna",
    "sigla_partido",
    "sigla_unidade_federativa",
    "sigla_unidade_federativa_nascimento",
    "titulo_eleitoral",
]

rules = {
    "cpf": {"type": "crop", "start": 0, "stop": 5},
    "despesa_maxima_campanha": {"type": "hist", "nbins": 20},
    "email": {"type": "crop", "start": 0, "stop": 5},
    "nome": {"type": "split", "char": " ", "keep": 0},
    "nome_social": {"type": "split", "char": " ", "keep": 0},
    "nome_urna": {"type": "split", "char": " ", "keep": 0},
    "titulo_eleitoral": {"type": "crop", "start": 0, "stop": 5},
}

################################
# run
################################

df = pd.read_parquet("datasets/eleicoes.parquet")
cleaner(df, "output/eleicoes_raw.json")
cleaner(Supression.anonymize(df, LGPD_COLUMNS), "output/eleicoes_supression.json")
cleaner(Randomization.anonymize(df, LGPD_COLUMNS), "output/eleicoes_randomization.json")
cleaner(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    "output/eleicoes_generalization.json",
)
cleaner(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    "output/eleicoes_pseudoanonymization.json",
)
