# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import hashlib


class PseudoAnonymization:
    @staticmethod
    def anonymize(df, columns):
        for c in columns:
            df[c] = [
                hashlib.sha224(str(row).encode("UTF-8")).hexdigest() for row in df[c]
            ]
        return df
