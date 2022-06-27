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
    "date_of_birth",
    "place_of_birth",
    "date_of_death",
    "profession",
    "years_of_service",
    "cause_of_death",
]

rules = {
    "place_of_birth": {"type": "split", "char": " ", "keep": -1},
    "profession": {"type": "split", "char": " ", "keep": 0},
    "years_of_service": {"type": "split", "char": " ", "keep": 0},
}

rules_ = {
    "place_of_birth": {"type": "split", "char": " ", "keep": -1},
    "profession": {"type": "split", "char": " ", "keep": 0},
    "years_of_service": {"type": "hist", "nbins": 4},
}

################################
# run
################################

K = 7

df = pd.read_parquet("datasets/Canada.parquet")
learn(df, K, "output/Canada_raw.png")
learn(Supression.anonymize(df, LGPD_COLUMNS), K, "output/Canada_supression.png")
learn(
    Randomization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/Canada_randomization.png",
)
df_ = Generalization.anonymize(df, LGPD_COLUMNS, rules)
learn(
    Generalization.anonymize(df_, LGPD_COLUMNS, rules_),
    K,
    "output/Canada_generalization.png",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/Canada_pseudoanonymization.png",
)
