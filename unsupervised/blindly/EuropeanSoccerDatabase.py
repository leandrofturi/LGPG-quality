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
    "birthday",
    "height",
    "weight",
    "overall_rating",
    "potential",
    "preferred_foot",
    "attacking_work_rate",
    "defensive_work_rate",
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "free_kick_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes",
]


rules = {
    "height": {"type": "hist", "nbins": 10},
    "weight": {"type": "hist", "nbins": 10},
    "overall_rating": {"type": "hist", "nbins": 8},
    "potential": {"type": "hist", "nbins": 8},
    "crossing": {"type": "hist", "nbins": 8},
    "finishing": {"type": "hist", "nbins": 8},
    "heading_accuracy": {"type": "hist", "nbins": 8},
    "short_passing": {"type": "hist", "nbins": 8},
    "volleys": {"type": "hist", "nbins": 8},
    "dribbling": {"type": "hist", "nbins": 8},
    "curve": {"type": "hist", "nbins": 8},
    "free_kick_accuracy": {"type": "hist", "nbins": 8},
    "long_passing": {"type": "hist", "nbins": 8},
    "ball_control": {"type": "hist", "nbins": 8},
    "acceleration": {"type": "hist", "nbins": 8},
    "sprint_speed": {"type": "hist", "nbins": 8},
    "agility": {"type": "hist", "nbins": 8},
    "reactions": {"type": "hist", "nbins": 8},
    "balance": {"type": "hist", "nbins": 8},
    "shot_power": {"type": "hist", "nbins": 8},
    "jumping": {"type": "hist", "nbins": 8},
    "stamina": {"type": "hist", "nbins": 8},
    "strength": {"type": "hist", "nbins": 8},
    "long_shots": {"type": "hist", "nbins": 8},
    "aggression": {"type": "hist", "nbins": 8},
    "interceptions": {"type": "hist", "nbins": 8},
    "positioning": {"type": "hist", "nbins": 8},
    "vision": {"type": "hist", "nbins": 8},
    "penalties": {"type": "hist", "nbins": 8},
    "marking": {"type": "hist", "nbins": 8},
    "standing_tackle": {"type": "hist", "nbins": 8},
    "sliding_tackle": {"type": "hist", "nbins": 8},
    "gk_diving": {"type": "hist", "nbins": 8},
    "gk_handling": {"type": "hist", "nbins": 8},
    "gk_kicking": {"type": "hist", "nbins": 8},
    "gk_positioning": {"type": "hist", "nbins": 8},
    "gk_reflexes": {"type": "hist", "nbins": 8},
}

################################
# run
################################

K = 8

df = pd.read_parquet("datasets/EuropeanSoccer.parquet")
learn(df, K, "output/EuropeanSoccer_raw.png")
learn(
    Supression.anonymize(df, LGPD_COLUMNS),
    K,
    "output/EuropeanSoccer_supression.png",
)
learn(
    Randomization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/EuropeanSoccer_randomization.png",
)
learn(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    K,
    "output/EuropeanSoccer_generalization.png",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/EuropeanSoccer_pseudoanonymization.png",
)
