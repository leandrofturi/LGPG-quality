import pandas as pd
from requests import get as GET

df = pd.read_parquet('datasets/sinasc.parquet')

# accuracy (acurácia) EXAC
# EXAC_SINT
columns_date = ['dt_cadastro', 'dt_declarac', 'dt_nasc', 'dt_nasc_mae',
                'dt_recebim', 'dt_rec_orig', 'dt_ult_menst']
for c in columns_date:
    values = pd.to_datetime(df[c], format='%d%m%Y', errors='coerce')
    count = ((max_date - values).values.astype(int) > 120).sum()
    print(c, count)

r = GET('https://servicodados.ibge.gov.br/api/v1/localidades/municipios')
mun_ibge = [{
    'id': m['id'],
    'nome': m['nome'],
    'UF': m['microrregiao']['mesorregiao']['UF']['sigla'],
} for m in r.json()]
mun_ibge = pd.DataFrame(mun_ibge)

columns_mun_ibge = ['cod_mun_nasc', 'cod_mun_natu', 'cod_mun_res']
for c in columns_mun_ibge:
    values = df[c].astype(int, errors='ignore').astype(str)
    count = (~values.isin(mun_ibge['id'].astype(str).str[:6])).sum()
    print(c, count)

# EXAC_SEMAN

# RAN_EXAC
# dates: 1970 - 2017
min_date =  pd.to_datetime('01011970', format='%d%m%Y')
max_date =  pd.to_datetime('31122017', format='%d%m%Y')
for c in ['dt_cadastro', 'dt_declarac', 'dt_nasc',
          'dt_recebim', 'dt_rec_orig']:
    values = pd.to_datetime(df[c], format='%d%m%Y', errors='coerce')
    count = ((values < min_date) | (values > max_date)).sum()
    print(c, count)

for c in ['dt_nasc_mae','dt_ult_menst']:
    values = pd.to_datetime(df[c], format='%d%m%Y', errors='coerce')
    count = (values > max_date).sum()
    print(c, count)

# completeness (completude) COMP
# COMP_REG
df.isna().sum()

# consistency (consistência) CONS
# CONS_SEMAN
# qtdfilmort + qtdfilvivo > qtdgestant
sum(df.qtd_fil_mort + df.qtd_fil_vivo > df.qtd_gest_ant)
# qtdpartces + qtdpartnor > qtdgestant
sum(df.qtd_part_ces + df.qtd_part_nor > df.qtd_gest_ant)
# idanomal != 1 & codanomal !=NULL
sum((df.id_anomal != 1.0) & (~df.cod_anomal.isnull()))
# dtnasc > dtcadastro
sum(df.dt_nasc > df.dt_cadastro)
# dtnasc < dtultmenst
sum(df.dt_nasc < df.dt_ult_menst)
# dtnasc > dtrecoriga
sum(df.dt_nasc > df.dt_rec_orig)
# dtnasc > dtrecebim
sum(df.dt_nasc > df.dt_recebim)
# dtnasc < dtnascmae
sum(df.dt_nasc < df.dt_nasc_mae)
# escmae > idademae
sum(df.esc_mae > df.idade_mae)
# (locnasc != 1 & 2) & codestab != NULL
sum((df.loc_nasc != 1.0) & (df.loc_nasc != 2.0) & (~df.cod_estab.isnull()))
# (stcesparto == 1 |2) & parto ==1
sum((df.st_ces_parto == 1.0) & (df.st_ces_parto == 2.0) & (df.parto == 1.0))

# credibility (credibilidade) CRED
# CRED_VAL_DAT
df.consultas.value_counts()
(~df.cod_mun_nasc.isin(mun_ibge.loc[mun_ibge['UF'] == 'ES', 'id'].astype(str).str[:6])).sum()


# currentness (atualidade) CURR
# CURR_UPD

# fazer grafico
values_dt_recebim = pd.to_datetime(df.dt_recebim, format='%d%m%Y', errors='coerce')
for c in ['dt_cadastro', 'dt_declarac', 'dt_nasc']:
    values = pd.to_datetime(df[c], format='%d%m%Y', errors='coerce')
    mean = (values_dt_recebim - values).mean().days
    print(c, mean)
