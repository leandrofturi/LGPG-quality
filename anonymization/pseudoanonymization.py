# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import hashlib


class PseudoAnonymization:
    @staticmethod
    def anonymize(df, columns):
        df_cpy = df.copy()
        for c in columns:
            df_cpy[c] = [
                hashlib.sha224(str(row).encode("UTF-8")).hexdigest() for row in df_cpy[c]
            ]
        return df_cpy
