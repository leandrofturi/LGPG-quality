import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster.elbow import kelbow_visualizer

# from cleaner.eleicoes import cleaner
from unsupervised.model_featured import learn
from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization


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


np.random.seed(42)


################################
# encoding
################################

# numeric
"ano", "despesa_maxima_campanha"

# sorted categorical

# unsorted categorical
"ocupacao", "unidade_eleitoral", "municipio_nascimento", "partido", "sigla_partido",
"sigla_unidade_federativa", "sigla_unidade_federativa_nascimento"

# small domain categorical
"declara_bens", "cargo", "etnia", "eleicao", "estado_civil", "genero", "grau_instrucao",
"nacionalidade", "tipo_eleicao"

# dates
"data_nascimento",

# remove
"cpf", "email", "nome", "nome_social", "nome_urna", "titulo_eleitoral"


# df = pd.read_parquet("datasets/eleicoes.parquet")
# valid_rows = cleaner(df)
# v = valid_rows.sum() / len(df.index)
# v_cols = v.loc[v >= 0.85]

# ['ano', 'data_nascimento', 'cargo', 'eleicao', 'estado_civil', 'genero',
#  'grau_instrucao', 'nacionalidade', 'ocupacao',
#  'despesa_maxima_campanha', 'nome', 'partido', 'tipo_eleicao',
#  'nome_urna', 'sigla_partido', 'sigla_unidade_federativa',
#  'sigla_unidade_federativa_nascimento']


def categorize(df):
    X = df[["ano"]].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    X["data_nascimento"] = (
        pd.to_datetime(
            df["data_nascimento"], format="%Y-%m-%d", errors="coerce"
        ).values.astype(np.int64)
        // 10 ** 9
    )

    mun_ibge = pd.read_csv("utils/municipiosIBGE.csv")
    mun_ibge_dict = {
        k: mun_ibge.loc[mun_ibge.UF == k, "id"].iloc[0].astype(str)[:2]
        for k in mun_ibge.UF.unique()
    }
    X["sigla_unidade_federativa"] = df["sigla_unidade_federativa"].apply(
        lambda x: mun_ibge_dict.get(x)
    )
    X["sigla_unidade_federativa_nascimento"] = df[
        "sigla_unidade_federativa_nascimento"
    ].apply(lambda x: mun_ibge_dict.get(x))

    for c in [
        "cargo",
        "eleicao",
        "estado_civil",
        "genero",
        "grau_instrucao",
        "nacionalidade",
        "tipo_eleicao",
        "sigla_partido",
    ]:
        X = pd.concat(
            [
                X,
                pd.get_dummies(df[c].astype(str), prefix=c).apply(
                    lambda x: x.astype(int)
                ),
            ],
            axis=1,
        )

    X["despesa_maxima_campanha"] = pd.to_numeric(
        df["despesa_maxima_campanha"], errors="coerce"
    )

    t = pd.get_dummies(df["ocupacao"].astype(str), prefix="ocupacao")
    # pca = PCA()
    # tt = pca.fit_transform(t)
    # plt.plot(range(len(pca.explained_variance_ratio_.cumsum())), pca.explained_variance_ratio_.cumsum())
    # plt.show()
    # pca.explained_variance_ratio_.cumsum()

    n_components = min(len(t.columns), 22)
    pca = PCA(n_components=n_components)
    tt = pd.DataFrame(
        pca.fit_transform(t),
        columns=[f"ocupacao_PCA{i}" for i in range(n_components)],
        index=X.index,
    )
    X = pd.concat([X, tt], axis=1)

    X.fillna(-1, inplace=True)
    X_scl = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    return X_scl


################################
# elbow
################################

# df = pd.read_parquet("datasets/eleicoes.parquet")
# X = categorize(df)

# kelbow_visualizer(KMeans(), X, k=(2, 24), title="eleicoes")

K = 6

################################
# learn
################################

df = pd.read_parquet("datasets/eleicoes.parquet")
y_true = learn(categorize(df)[:30000], K, "output/km_eleicoes_raw.png")
y_supression = learn(
    categorize(Supression.anonymize(df, LGPD_COLUMNS))[:30000],
    K,
    "output/km_eleicoes_supression.png",
)
y_randomization = learn(
    categorize(Randomization.anonymize(df, LGPD_COLUMNS))[:30000],
    K,
    "output/km_eleicoes_randomization.png",
)
y_generalization = learn(
    categorize(Generalization.anonymize(df, LGPD_COLUMNS, rules))[:30000],
    K,
    "output/km_eleicoes_generalization.png",
)
y_pseudoanonymization = learn(
    categorize(PseudoAnonymization.anonymize(df, LGPD_COLUMNS))[:30000],
    K,
    "output/km_eleicoes_pseudoanonymization.png",
)

final = {
    "supression": rand_score(y_true, y_supression),
    "randomization": rand_score(y_true, y_randomization),
    "generalization": rand_score(y_true, y_generalization),
    "pseudoanonymization": rand_score(y_true, y_pseudoanonymization)
}

with open("rand_score_eleicoes.json", "w") as f:
    json.dump(final, f, indent=4, sort_keys=False)
