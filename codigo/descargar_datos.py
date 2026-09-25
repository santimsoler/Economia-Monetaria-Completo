"""
DESCARGAR_DATOS.PY
Baja los datos de los siete paises: EE.UU. (ya estaba en tp_datos del panel
original), zona euro, Canada, Reino Unido, Rusia, Turquia. Argentina usa
argentina_mensual.csv, que ya tenias.

Fuentes: FRED (series de la OCDE/BIS: IPC, dinero, produccion, tasas,
acciones, tipo de cambio, salario, vivienda real, credito/PIB), BCE (SPF,
encuesta de pronosticadores), Banco de Canada (encuesta de consumidores,
solo de control), Banco de Inglaterra (encuesta de actitudes hacia la
inflacion, trimestral, se intenta bajar sola y si falla se pide a mano) y
el banco central turco (encuesta de participantes del mercado, M2 y tasa
de depositos, via el paquete "evds", pide una clave gratuita).

Guarda en tp_datos/: pais_EA.csv, pais_CA.csv, pais_GB.csv, pais_RU.csv,
pais_TR.csv (mensuales); pais_EA_trim.csv, pais_CA_trim.csv, pais_GB_trim.csv,
pais_TR_trim.csv (trimestrales); y cobertura_paises.csv con un resumen de
que se bajo, desde donde y hasta cuando.

Para cada variable se prueban codigos en orden y se usa el primero que
funciona; el informe de cobertura deja registrado cual fue.
"""
import io, re, subprocess, sys, time, unicodedata, warnings
from getpass import getpass
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings('ignore')

DIR = 'tp_datos'
VENTANA_COBERTURA = (pd.Period('2006-01', 'M'), pd.Period('2024-12', 'M'))


# ============================================================ utilidades
def fred(sid, intentos=3):
    for k in range(intentos):
        try:
            t = pd.read_csv(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}')
            t.columns = ['fecha', 'v']
            s = pd.Series(pd.to_numeric(t['v'], errors='coerce').values,
                          index=pd.to_datetime(t['fecha'])).dropna()
            return s if len(s) else None
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def a_mensual(s):
    return s.groupby(s.index.to_period('M')).mean()


def a_trimestral(s):
    return s.groupby(s.index.to_period('Q')).mean()


def a_trimestre(txt):
    """'1999-Q1', '1999Q1', '1999-03', '1999-03-31' -> Period trimestral."""
    txt = str(txt)
    m = re.match(r'(\d{4})\D*Q(\d)', txt)
    if m:
        return pd.Period(year=int(m.group(1)), quarter=int(m.group(2)), freq='Q')
    m = re.match(r'(\d{4})-(\d{2})', txt)
    if m:
        return pd.Period(year=int(m.group(1)), month=int(m.group(2)), freq='M').asfreq('Q')
    return pd.NaT


def norm(s):
    """Minusculas sin tildes ni letras turcas (i, s, g, u, o, c sin puntos)."""
    s = str(s).replace('\u0130', 'I').replace('\u0131', 'i')
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


def primera(nombre, candidatos, informe, pais, bce_fn=None):
    """candidatos: lista de ('fred'|'bce', codigo)."""
    for fuente, cod in candidatos:
        s = fred(cod) if fuente == 'fred' else (bce_fn(cod) if bce_fn else None)
        if s is not None and len(s) > 24:
            informe.append({'pais': pais, 'variable': nombre, 'fuente': fuente, 'codigo': cod,
                            'desde': s.index.min().strftime('%Y-%m'),
                            'hasta': s.index.max().strftime('%Y-%m'), 'n': len(s)})
            return s
    informe.append({'pais': pais, 'variable': nombre, 'fuente': '-', 'codigo': 'NO DISPONIBLE',
                    'desde': '', 'hasta': '', 'n': 0})
    return None


def bce(clave):
    try:
        r = requests.get(f'https://data-api.ecb.europa.eu/service/data/{clave}',
                         params={'format': 'csvdata'}, timeout=90)
        r.raise_for_status()
        t = pd.read_csv(io.StringIO(r.text))
        p = t['TIME_PERIOD'].astype(str)
        idx = (pd.PeriodIndex(p.str.replace('-', ''), freq='Q').to_timestamp()
              if p.str.contains('Q').any() else pd.to_datetime(p))
        return pd.Series(pd.to_numeric(t['OBS_VALUE'], errors='coerce').values, index=idx).dropna()
    except Exception:
        return None


def catalogo(p, iso, bis):
    """p: codigo OCDE de 2 letras (EZ para zona euro, CA, GB, RU, TR).
    iso: codigo ISO3 usado en algunos codigos alternativos de FRED.
    bis: codigo de pais del BIS, para vivienda real y credito/PIB."""
    m = {
        'ipc': [('fred', f'CPALTT01{p}M661N'), ('fred', f'{iso}CPIALLMINMEI')],
        'dinero': [('fred', f'MABMM301{p}M189S')],
        'produccion': [('fred', f'{iso}PROINDMISMEI'), ('fred', f'PRINTO01{p}M661S')],
        'tasa_corta': [('fred', f'IR3TIB01{p}M156N'), ('fred', f'IRSTCI01{p}M156N')],
        'tasa_larga': [('fred', f'IRLTLT01{p}M156N')],
        'acciones': [('fred', f'SPASTT01{p}M661N')],
        'tipo_cambio': [('fred', f'CCUSMA02{p}M618N')],
        'salario': [('fred', f'LCEAMN01{p}M661S'), ('fred', f'LCEAMN01{p}M661N')],
    }
    q = {
        'vivienda_real': [('fred', f'Q{bis}R628BIS')],
        'credito_pib': [('fred', f'Q{bis}PAM770A')],
    }
    if p == 'EZ':      # respaldos del BCE para la zona euro
        m['ipc'].insert(0, ('fred', 'CP0000EZ19M086NEST'))
        m['ipc'].append(('bce', 'ICP/M.U2.N.000000.4.INX'))
        m['dinero'].append(('bce', 'BSI/M.U2.Y.V.M30.X.1.U2.2300.Z01.E'))
        m['produccion'].append(('bce', 'STS/M.I8.Y.PROD.NS0020.4.000'))
        m['tasa_corta'].insert(0, ('bce', 'FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA'))
    return m, q


# ============================================================ zona euro: SPF
def spf_zona_euro(informe):
    """Encuesta de pronosticadores del BCE (SPF), trimestral desde 1999.
    Hay dos claves publicadas para el mismo horizonte de 12 meses (una con
    fechas anuales, otra con fechas trimestrales): se usa la trimestral."""
    try:
        r = requests.get('https://data-api.ecb.europa.eu/service/data/SPF/.U2.HICP.POINT.P12M.Q.AVG',
                         params={'format': 'csvdata'}, timeout=90)
        r.raise_for_status()
        t = pd.read_csv(io.StringIO(r.text))
        claves = t['KEY'].unique().tolist()
        clave = (next((c for c in claves if c.startswith('SPF.Q.')), None)
                or next((c for c in claves if c.startswith('SPF.M.')), claves[0]))
        x = t[t['KEY'] == clave]
        s = pd.Series(pd.to_numeric(x['OBS_VALUE'], errors='coerce').values,
                      index=[a_trimestre(p) for p in x['TIME_PERIOD']]).dropna()
        s = s[[not pd.isna(i) for i in s.index]].groupby(level=0).mean().sort_index()
        informe.append({'pais': 'EA', 'variable': 'expectativa_12m (SPF, trim.)', 'fuente': 'bce',
                        'codigo': clave, 'desde': str(s.index.min()), 'hasta': str(s.index.max()),
                        'n': len(s)})
        return s
    except Exception as e:
        informe.append({'pais': 'EA', 'variable': 'expectativa_12m (SPF)', 'fuente': '-',
                        'codigo': f'NO DISPONIBLE ({type(e).__name__})', 'desde': '', 'hasta': '', 'n': 0})
        return None


# ============================================================ Canada: encuesta (solo control)
def encuesta_canada(informe):
    """Encuesta de expectativas de consumidores del Banco de Canada (desde
    2014); solo se usa como control, la medida principal de Canada es
    adaptativa."""
    try:
        lst = requests.get('https://www.bankofcanada.ca/valet/lists/series/json',
                           timeout=60).json()['series']
        cand = {k: v for k, v in lst.items()
                if '2023Q1' not in k and 'consistent' not in v.get('label', '')
                and 'inflation' in (v.get('label', '') + v.get('description', '')).lower()
                and 'consumer' in (v.get('label', '') + v.get('description', '')).lower()
                and ('one year' in (v.get('label', '') + v.get('description', '')).lower()
                     or '1-year' in (v.get('label', '') + v.get('description', '')).lower())}
        if not cand:
            raise ValueError('sin candidatas')
        cod = list(cand)[0]
        obs = requests.get(f'https://www.bankofcanada.ca/valet/observations/{cod}/json',
                           timeout=60).json()['observations']
        s = pd.Series({pd.Timestamp(o['d']): float(o[cod]['v']) for o in obs if o.get(cod, {}).get('v')})
        informe.append({'pais': 'CA', 'variable': 'expectativa_12m (encuesta, control)', 'fuente': 'valet',
                        'codigo': cod, 'desde': str(s.index.min())[:7], 'hasta': str(s.index.max())[:7],
                        'n': len(s)})
        return s
    except Exception as e:
        informe.append({'pais': 'CA', 'variable': 'expectativa_12m (encuesta, control)', 'fuente': '-',
                        'codigo': f'NO DISPONIBLE ({type(e).__name__})', 'desde': '', 'hasta': '', 'n': 0})
        return None


# ============================================================ Reino Unido: encuesta
def encuesta_reino_unido(informe):
    """Encuesta de actitudes hacia la inflacion del Banco de Inglaterra
    (con Ipsos), trimestral desde 1999, gratuita. La ruta del archivo en
    el sitio del banco cambia seguido; si falla, se busca un respaldo
    bajado a mano en tp_datos/reino_unido_expectativas.csv (dos columnas,
    trimestre y valor, mediana de "prices to rise over next 12 months")."""
    intentos = [
        'https://www.bankofengland.co.uk/-/media/boe/files/statistics/'
        'public-attitudes-to-inflation/iasdata.xlsx',
    ]
    for url in intentos:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            t = pd.read_excel(io.BytesIO(r.content))
            informe.append({'pais': 'GB', 'variable': 'expectativa_encuesta (trim.)', 'fuente': 'boe',
                            'codigo': url, 'desde': '?', 'hasta': '?', 'n': len(t)})
            print('  columnas del archivo del BoE:', list(t.columns)[:8])
            return t
        except Exception as e:
            print(f'  [!] {url}: {type(e).__name__}')
    manual = f'{DIR}/reino_unido_expectativas.csv'
    try:
        t = pd.read_csv(manual)
        s = pd.Series(pd.to_numeric(t.iloc[:, 1], errors='coerce').values,
                      index=pd.PeriodIndex(t.iloc[:, 0].astype(str), freq='Q')).dropna()
        informe.append({'pais': 'GB', 'variable': 'expectativa_encuesta (trim.)', 'fuente': 'archivo',
                        'codigo': manual, 'desde': str(s.index.min()), 'hasta': str(s.index.max()),
                        'n': len(s)})
        return s.to_frame('expectativa_spf')
    except FileNotFoundError:
        print(f'  [!] Bajala a mano de bankofengland.co.uk (buscar "Inflation Attitudes '
              f'Survey", pregunta "prices to rise over next 12 months", mediana) y guardala '
              f'en {manual} con columnas trimestre,valor.')
        informe.append({'pais': 'GB', 'variable': 'expectativa_encuesta (trim.)', 'fuente': '-',
                        'codigo': 'NO DISPONIBLE', 'desde': '', 'hasta': '', 'n': 0})
        return None


# ============================================================ Turquia (EVDS)
def turquia_todo(informe):
    """Encuesta de expectativas (participantes del mercado, empalmada con
    la archivada 2001-2012), M2 y tasa de depositos en liras (la de mejor
    cobertura 2006-2024) del banco central turco. Pide una clave gratuita
    de evds3.tcmb.gov.tr; sin clave, Turquia queda sin estas tres series
    (las demas, IPC/produccion/acciones/tipo de cambio, salen de FRED igual)."""
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'evds'], check=False)
    clave = getpass('  Pega tu clave de EVDS (evds3.tcmb.gov.tr) y apreta Enter '
                    '(Enter vacio para seguir sin estas tres series): ').strip()
    if not clave:
        for v in ['expectativa_encuesta', 'dinero_m2', 'tasa_interbancaria']:
            informe.append({'pais': 'TR', 'variable': v, 'fuente': '-', 'codigo': 'SIN CLAVE',
                            'desde': '', 'hasta': '', 'n': 0})
        return {}
    from evds import evdsAPI
    ev = evdsAPI(clave)
    try:
        ev.get_data(['TP.DK.USD.A.YTL'], startdate='01-01-2024', enddate='31-01-2024')
    except Exception as e:
        print(f'  [!] la clave no funciono ({type(e).__name__}): generá una nueva desde tu perfil.')
        for v in ['expectativa_encuesta', 'dinero_m2', 'tasa_interbancaria']:
            informe.append({'pais': 'TR', 'variable': v, 'fuente': '-',
                            'codigo': f'ERROR {type(e).__name__}', 'desde': '', 'hasta': '', 'n': 0})
        return {}

    def bajar_serie(cod):
        d = ev.get_data([cod], startdate='01-01-1995', enddate=pd.Timestamp.today().strftime('%d-%m-%Y'),
                        frequency=5, aggregation_types='avg')
        fcol = [c for c in d.columns if c.lower() in ('tarih', 'date')][0]
        vcol = [c for c in d.columns if c not in (fcol, 'UNIXTIME', 'YEARWEEK')][0]
        s = pd.Series(pd.to_numeric(d[vcol], errors='coerce').values,
                      index=pd.PeriodIndex(pd.to_datetime(d[fcol], errors='coerce'), freq='M')).dropna()
        return s.groupby(level=0).mean()

    def series_de(grupo):
        sl = ev.get_series(grupo)
        ncol = [c for c in sl.columns if 'SERIE_NAME' in c.upper()]
        ccol = [c for c in sl.columns if 'SERIE_CODE' in c.upper()][0]
        nom = sl[ncol].astype(str).agg(' | '.join, axis=1).map(norm)
        return pd.DataFrame({'codigo': sl[ccol], 'nombre': nom})

    grupos = []
    for cid in ev.main_categories.iloc[:, 0]:
        try:
            grupos.append(ev.get_sub_categories(cid))
        except Exception:
            pass
    grupos = pd.concat(grupos, ignore_index=True)
    gcol = [c for c in grupos.columns if 'DATAGROUP_CODE' in c.upper()][0]
    nom_g = grupos.astype(str).agg(' | '.join, axis=1).map(norm)
    out = {}

    # --- expectativas: vigente (participantes del mercado) + archivada
    es12 = lambda n: ('tufe' in n or 'cpi' in n) and '12 ay' in n and ('yillik' in n or 'annual' in n) \
        and not any(w in n for w in ['24 ay', 'olasilik', 'dagilim', 'yil sonu', 'standart',
                                     'gozlem', 'en buyuk', 'en kucuk', 'mod)', 'uygun'])
    try:
        v = series_de('bie_pkauo')
        v = v[v['nombre'].apply(es12)]
        pref = v[v['nombre'].str.contains('medyan|median')]
        cod_n = (pref if len(pref) else v)['codigo'].iloc[0]
        nueva = bajar_serie(cod_n)
        vieja = bajar_serie('TP.BEKA.S01.E.M')      # archivada, mediana
        exp = pd.concat([vieja[vieja.index < nueva.index.min()], nueva])
        out['expectativa_encuesta'] = exp
        informe.append({'pais': 'TR', 'variable': 'expectativa_encuesta', 'fuente': 'evds',
                        'codigo': f'{cod_n} + TP.BEKA.S01.E.M', 'desde': str(exp.index.min()),
                        'hasta': str(exp.index.max()), 'n': len(exp)})
    except Exception as e:
        informe.append({'pais': 'TR', 'variable': 'expectativa_encuesta', 'fuente': '-',
                        'codigo': f'ERROR {type(e).__name__}', 'desde': '', 'hasta': '', 'n': 0})

    # --- M2: id conocido (TP.PBD.H09); si cambiara, buscar por texto
    try:
        m2 = bajar_serie('TP.PBD.H09')
        out['dinero_m2'] = m2
        informe.append({'pais': 'TR', 'variable': 'dinero_m2', 'fuente': 'evds', 'codigo': 'TP.PBD.H09',
                        'desde': str(m2.index.min()), 'hasta': str(m2.index.max()), 'n': len(m2)})
    except Exception as e:
        informe.append({'pais': 'TR', 'variable': 'dinero_m2', 'fuente': '-',
                        'codigo': f'ERROR {type(e).__name__}', 'desde': '', 'hasta': '', 'n': 0})

    # --- tasa: la de mejor cobertura 2006-2024 entre depositos en liras y repo
    es_tasa = lambda n: (('mevduat' in n or 'deposit' in n) and ('tl' in n or 'turk lira' in n)
                         and not any(w in n for w in ['usd', 'eur', 'doviz', 'fx', 'altin'])) \
        or ('repo' in n and ('gecelik' in n or 'overnight' in n))
    gr = grupos.loc[nom_g.str.contains('faiz|interest|repo|mevduat|deposit'), gcol].tolist()
    cands = []
    for g in gr:
        try:
            x = series_de(g)
            cands.append(x[x['nombre'].apply(es_tasa)].assign(grupo=g))
        except Exception:
            pass
    cands = pd.concat(cands).drop_duplicates('codigo').head(30) if cands else pd.DataFrame()
    filas = []
    for _, c in cands.iterrows():
        try:
            s = bajar_serie(c['codigo'])
            cob = int(((s.index >= VENTANA_COBERTURA[0]) & (s.index <= VENTANA_COBERTURA[1])).sum())
            filas.append({'codigo': c['codigo'], 's': s, 'cobertura': cob})
        except Exception:
            pass
    if filas:
        mejor = max(filas, key=lambda f: f['cobertura'])
        out['tasa_interbancaria'] = mejor['s']
        informe.append({'pais': 'TR', 'variable': 'tasa_interbancaria', 'fuente': 'evds',
                        'codigo': mejor['codigo'], 'desde': str(mejor['s'].index.min()),
                        'hasta': str(mejor['s'].index.max()), 'n': len(mejor['s'])})
    else:
        informe.append({'pais': 'TR', 'variable': 'tasa_interbancaria', 'fuente': '-',
                        'codigo': 'NO DISPONIBLE', 'desde': '', 'hasta': '', 'n': 0})
    return out


# ============================================================ EE.UU.
def bajar_eeuu(informe):
    """m2, cpi, ip, ff, credito, acciones, vivienda, salario (en
    tp_datos/eeuu_exceso.csv) y la expectativa de Michigan (en
    tp_datos/eeuu_expectativas.csv). fx y riqueza del 10% se bajan aparte,
    ya adentro de pais_eeuu() en estimar_paises.py."""
    pedidos = {'m2': ['M2SL'], 'cpi': ['CPIAUCSL'], 'ip': ['INDPRO'], 'ff': ['FEDFUNDS'],
              'credito': ['LOANS', 'TOTLL', 'TOTBKCR'], 'acciones': ['NASDAQCOM'],
              'vivienda': ['CSUSHPISA'], 'salario': ['CES0500000008', 'AHETPI']}
    D = {}
    for k, cods in pedidos.items():
        s = primera(k, [('fred', c) for c in cods], informe, 'US')
        if s is not None:
            D[k] = a_mensual(s)
    if D:
        x = pd.DataFrame(D); x.index = x.index.astype(str)
        x.to_csv(f'{DIR}/eeuu_exceso.csv')
    mich = primera('expectativa_michigan', [('fred', 'MICH')], informe, 'US')
    if mich is not None:
        x = a_mensual(mich).rename('expectativa_michigan').to_frame()
        x.index = x.index.astype(str)
        x.to_csv(f'{DIR}/eeuu_expectativas.csv')


# ============================================================ Argentina: credito privado (API BCRA)
def bcra_credito_privado(informe):
    """Prestamos de las entidades financieras al sector privado, en pesos
    (idVariable 26 de la API del BCRA). Se baja en vivo; no depende de que
    el archivo ya exista de una sesion anterior."""
    base = 'https://api.bcra.gob.ar/estadisticas/v4.0/monetarias'
    try:
        regs, offset = [], 0
        while True:
            r = requests.get(f'{base}/26', params={'desde': '2003-01-01',
                             'hasta': pd.Timestamp.today().strftime('%Y-%m-%d'),
                             'limit': 3000, 'offset': offset}, verify=False, timeout=60)
            r.raise_for_status()
            det = []
            for x in r.json().get('results', []):
                det += x.get('detalle', [x] if 'fecha' in x else [])
            if not det:
                break
            regs += det
            if len(det) < 3000:
                break
            offset += 3000
        s = pd.Series({pd.Timestamp(d['fecha']): d['valor'] for d in regs}).sort_index()
        s = pd.to_numeric(s, errors='coerce').dropna()
        c = a_mensual(s)
        x = c.copy(); x.index = x.index.astype(str)
        x.to_frame('valor').to_csv(f'{DIR}/bcra_prestamos_privados.csv')
        informe.append({'pais': 'AR', 'variable': 'credito_privado', 'fuente': 'bcra',
                        'codigo': 'idVariable 26', 'desde': str(c.index.min()),
                        'hasta': str(c.index.max()), 'n': len(c)})
    except Exception as e:
        informe.append({'pais': 'AR', 'variable': 'credito_privado', 'fuente': '-',
                        'codigo': f'ERROR {type(e).__name__}', 'desde': '', 'hasta': '', 'n': 0})
        print(f'  [!] credito privado de Argentina: {type(e).__name__}: {e}')


# ============================================================ principal
def bajar():
    informe = []
    print('=== EE.UU. ===')
    bajar_eeuu(informe)
    print('=== Argentina: credito privado ===')
    import os
    if not os.path.exists(f'{DIR}/bcra_prestamos_privados.csv'):
        bcra_credito_privado(informe)
    else:
        print('  ya existe, no se vuelve a bajar')
    paises = [('EZ', 'EA19', 'XM', 'EA'), ('CA', 'CAN', 'CA', 'CA'),
             ('GB', 'GBR', 'GB', 'GB'), ('RU', 'RUS', 'RU', 'RU'),
             ('TR', 'TUR', 'TR', 'TR')]
    for p, iso, bisc, nombre in paises:
        print(f'\n=== {nombre} ===')
        m, q = catalogo(p, iso, bisc)
        M = pd.DataFrame({k: a_mensual(s) for k, c in m.items()
                          if (s := primera(k, c, informe, nombre, bce)) is not None})
        Q = pd.DataFrame({k: a_trimestral(s) for k, c in q.items()
                          if (s := primera(k, c, informe, nombre, bce)) is not None})
        if nombre == 'EA':
            spf = spf_zona_euro(informe)
            if spf is not None:
                Q['expectativa_spf'] = spf
        if nombre == 'CA':
            enc = encuesta_canada(informe)
            if enc is not None:
                Q['expectativa_encuesta'] = a_trimestral(enc)
        if nombre == 'GB':
            enc = encuesta_reino_unido(informe)
            if enc is not None and 'expectativa_spf' in getattr(enc, 'columns', []):
                Q['expectativa_spf'] = enc['expectativa_spf']
        if nombre == 'TR':
            for k, v in turquia_todo(informe).items():
                M[k] = v
        for df, suf in [(M, ''), (Q, '_trim')]:
            if len(df):
                x = df.copy(); x.index = x.index.astype(str)
                x.to_csv(f'{DIR}/pais_{nombre}{suf}.csv')

    inf = pd.DataFrame(informe)
    inf.to_csv(f'{DIR}/cobertura_paises.csv', index=False)
    print('\n' + '=' * 100 + '\nCOBERTURA\n' + '=' * 100)
    print(inf.to_string(index=False))
    print('\n  Rusia: la OCDE dejo de reportar sus series despues de feb-2022 (no es un')
    print('  quiebre dentro de la serie); la muestra util llega hasta 2022 aprox.')
    return inf


cob = bajar()

# ============================================================ empaquetar
def empaquetar_zip():
    """Comprime TODO lo que haya en tp_datos/ (lo que ya tenias embebido en
    Colab mas lo que este script acaba de bajar) en un unico zip y dispara
    la descarga al navegador."""
    import shutil
    nombre = shutil.make_archive('tp_datos_completo', 'zip', DIR)
    print(f'\n  Armado: {nombre}')
    try:
        from google.colab import files
        files.download(nombre)
    except ImportError:
        print('  (no estas en Colab: el zip quedo guardado en el disco, bajalo a mano)')


empaquetar_zip()
