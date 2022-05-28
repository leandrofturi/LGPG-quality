# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


class Randomization:
    @staticmethod
    def anonymize(df, columns):
        for c in columns:
            df[c] = df[c].sample(frac=1, random_state=42).reset_index(drop=True)
        return df
