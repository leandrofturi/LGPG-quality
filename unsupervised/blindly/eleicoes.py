import pandas as pd

from unsupervised.model import learn
from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization

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

K = 11

df = pd.read_parquet("datasets/eleicoes.parquet")
learn(df, K, "output/eleicoes_raw.png")
learn(Supression.anonymize(df, LGPD_COLUMNS), K, "output/eleicoes_supression.png")
learn(Randomization.anonymize(df, LGPD_COLUMNS), K, "output/eleicoes_randomization.png")
learn(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    K,
    "output/eleicoes_generalization.png",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/eleicoes_pseudoanonymization.png",
)
