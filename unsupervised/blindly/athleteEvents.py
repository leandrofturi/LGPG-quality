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
    "name",
    "sex",
    "age",
    "height",
    "weight",
    "team",
    "games",
    "year",
    "city",
    "sport",
    "event",
    "medal",
    "region",
]

rules = {
    "name": {"type": "split", "char": " ", "keep": 0},
    "event": {"type": "split", "char": " ", "keep": 0},
    "age": {"type": "hist", "nbins": 10},
    "height": {"type": "hist", "nbins": 10},
    "weight": {"type": "hist", "nbins": 10},
}


################################
# run
################################

K = 8

df = pd.read_parquet("datasets/athleteEvents.parquet")
learn(df, K, "output/athleteEvents_raw.png")
learn(Supression.anonymize(df, LGPD_COLUMNS), K, "output/athleteEvents_supression.png")
learn(
    Randomization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/athleteEvents_randomization.png",
)
learn(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    K,
    "output/athleteEvents_generalization.png",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/athleteEvents_pseudoanonymization.png",
)
