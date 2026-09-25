"""
=============================================================================
REPLICACION COMPLETA: TODOS LOS ESTADISTICOS DEL TRABAJO
=============================================================================
Lee la carpeta tp_datos/ generada por exportar_datos.py y reproduce cada
numero que figura en el texto, comparandolo con el valor escrito.

    OK            coincide dentro del redondeo
    DIFERENCIA    se estimo y da otro valor
    NO REPRODUC.  falta el insumo

BLOQUES
    1  Panel: resultado principal, efectos marginales, regimenes
    2  Panel: explicaciones alternativas y largo plazo
    3  Panel: explicaciones fiscales (dominancia y composicion por moneda)
    4  Argentina: demanda de dinero
    5  Argentina: deuda consolidada con el BCRA
    6  Wicksell: tasa de politica respecto de la natural, tres economias
    7  Hayek: composicion de la produccion con shocks exogenos

USO
    exec(open('replicar_tp.py').read())
    rep = replicar()
=============================================================================
"""

import io
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

warnings.filterwarnings("ignore")
pd.set_option('display.width', 220)

DIR = 'tp_datos'
CTRL_F = ['m_exceso', 'r_fisher', 'depreciacion']
REGS = ['m_exceso', '_m_x_pie', 'infl_prev_c', 'r_fisher', 'deficit',
        'depreciacion']
FILAS = []


# =======================================================================
def titulo(t):
    print("\n" + "=" * 86 + "\n" + t + "\n" + "=" * 86)


def anotar(seccion, item, texto, codigo, tol=0.0015):
    if codigo is None or (isinstance(codigo, float) and np.isnan(codigo)):
        est = 'NO REPRODUC.'
    else:
        est = 'OK' if abs(texto - codigo) <= tol else 'DIFERENCIA'
    FILAS.append({'sección': seccion, 'ítem': item, 'texto': texto,
                  'código': None if codigo is None else round(float(codigo), 4),
                  'estado': est})
    m = {'OK': '  ', 'DIFERENCIA': '!!', 'NO REPRODUC.': '??'}[est]
    c = '—' if codigo is None else f"{codigo:.4f}"
    print(f"  {m} {item:52s} texto {texto:>9.4f}   código {c:>9s}")


def leer(nombre):
    f = f'{DIR}/{nombre}.csv'
    return pd.read_csv(f) if os.path.exists(f) else None


def fe_dk(panel, dep, regs, bw=3):
    from linearmodels.panel import PanelOLS
    regs = [r for r in regs if r in panel.columns]
    m = panel[[dep] + regs].apply(pd.to_numeric, errors='coerce').dropna()
    if len(m) < 30:
        return None
    try:
        return PanelOLS(m[dep], m[regs], entity_effects=True,
                        time_effects=True, drop_absorbed=True).fit(
            cov_type='kernel', kernel='bartlett', bandwidth=bw)
    except Exception as e:
        print(f"     error: {e}")
        return None


def ols_hac(y, X, lags=12):
    m = pd.concat([y, X], axis=1).dropna()
    if len(m) < 30:
        return None
    return sm.OLS(m.iloc[:, 0], sm.add_constant(m.iloc[:, 1:])).fit(
        cov_type='HAC', cov_kwds={'maxlags': int(lags)})


def suma_lineal(res, vs):
    b = res.params[vs]
    V = res.cov.loc[vs, vs].values
    s = b.sum()
    se = float(np.sqrt(np.ones(len(vs)) @ V @ np.ones(len(vs))))
    return s, se, 2 * (1 - norm.cdf(abs(s / se)))


# =======================================================================
def preparar_panel(base, moneda=None):
    p = base.copy()
    for c in ['m_exceso', 'infl_prev_c', 'infl_ipc', 'r_fisher', 'deficit',
              'depreciacion', 'deuda_bruta_weo', 'resultado_primario_weo',
              'PIB_crec_real']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors='coerce')
    p = p.sort_values(['iso3', 'anio'])
    p['_m_x_pie'] = p['m_exceso'] * p['infl_prev_c']
    # infl_L1 y m_L1-L3 vienen ya calculados en panel_internacional.csv: NO
    # recalcular. infl_L1, pese al nombre, es el rezago de infl_deflactor
    # (no de infl_ipc); recalcularlo a partir de infl_ipc cambia el
    # multiplicador de largo plazo (5.1) en un 5% aprox. m_L1-L3 sí coinciden
    # con el recalculo, pero se dejan igual por consistencia con el resto.
    p['infl_L1'] = pd.to_numeric(p['infl_L1'], errors='coerce')
    for k in (1, 2, 3):
        p[f'm_L{k}'] = pd.to_numeric(p[f'm_L{k}'], errors='coerce')
    # fiscal
    p['_deuda'] = p.get('deuda_bruta_weo')
    p['_deuda_c'] = p['_deuda'] - p['_deuda'].mean()
    p['_deuda_alta'] = (p['_deuda'] > 60).astype(float)
    p['_def_x_alta'] = p['deficit'] * p['_deuda_alta']
    p['_r_x_alta'] = p['r_fisher'] * p['_deuda_alta']
    # composicion por moneda
    if moneda is not None and 'fx_share' in moneda.columns:
        mm = moneda[['iso3', 'anio', 'fx_share']].dropna()
        p = p.merge(mm, on=['iso3', 'anio'], how='left')
        p['_fx'] = pd.to_numeric(p['fx_share'], errors='coerce')
        p['_loc'] = 1 - p['_fx']
        p['_loc_c'] = p['_loc'] - p['_loc'].mean()
        p['_def'] = -pd.to_numeric(p.get('resultado_primario_weo'),
                                   errors='coerce')
        p['_b_lag'] = p.groupby('iso3')['_deuda'].shift(1)
        p['_d_b'] = p.groupby('iso3')['_deuda'].diff()
        p['_def_x_loc'] = p['_def'] * p['_loc_c']
        p['_dep_x_fx'] = p['depreciacion'] * p['_fx']
        p['_b_local'] = p['_b_lag'] * p['_loc']
        p['_b_fx'] = p['_b_lag'] * p['_fx']
    return p.set_index(['iso3', 'anio'])


# =======================================================================
# 1 - RESULTADO PRINCIPAL
# =======================================================================
def bloque_1(p):
    titulo("1. PANEL — RESULTADO PRINCIPAL (secciones 5.1 y 5.3)")
    r = fe_dk(p, 'infl_ipc', REGS)
    if r is None:
        anotar('5.1', 'especificación principal', 0.0109, None)
        return None
    anotar('5.1', 'β impulso monetario (en la media)', 0.0822,
           r.params['m_exceso'], 0.0006)
    anotar('5.1', 'β interacción', 0.0109, r.params['_m_x_pie'], 0.0006)
    anotar('5.1', 'N', 3054, r.nobs, 0.5)

    media = (pd.to_numeric(p['w_infl_media_movil'], errors='coerce') -
             p['infl_prev_c']).mean() if 'w_infl_media_movil' in p else 8.0
    b1, b2, V = r.params['m_exceso'], r.params['_m_x_pie'], r.cov
    print(f"\n  Efectos marginales (inflación previa centrada en {media:.2f}%)")
    for niv, esp, lo_t, hi_t in [(0, -0.005, None, None), (5, 0.050, None, None),
                                 (20, 0.213, None, None), (50, 0.538, None, None),
                                 (100, 1.081, 0.641, 1.521)]:
        x = niv - media
        e = b1 + b2 * x
        se = float(np.sqrt(V.loc['m_exceso', 'm_exceso'] +
                           x ** 2 * V.loc['_m_x_pie', '_m_x_pie'] +
                           2 * x * V.loc['m_exceso', '_m_x_pie']))
        anotar('5.1', f'efecto marginal en {niv}%', esp, e)
        if lo_t is not None:
            anotar('5.1', f'IC inferior en {niv}%', lo_t, e - 1.96 * se)
            anotar('5.1', f'IC superior en {niv}%', hi_t, e + 1.96 * se)

    print("\n  Por régimen de inflación")
    inf = pd.to_numeric(p['infl_ipc'], errors='coerce')
    con = ['m_exceso'] + CTRL_F[1:] + ['deficit']
    for nom, msk, bt, nt in [('baja', inf < 5, 0.0084, 1749),
                             ('moderada', (inf >= 5) & (inf < 20), 0.0144, 1177),
                             ('alta', inf >= 20, 0.8976, 128)]:
        rr = fe_dk(p[msk], 'infl_ipc', con)
        anotar('5.1 régimen', f'β impulso, {nom}',
               bt, None if rr is None else rr.params['m_exceso'], 0.0006)
        if rr is not None:
            anotar('5.1 régimen', f'N, {nom}', nt, rr.nobs, 0.5)
    return r


# =======================================================================
# 2 - ALTERNATIVAS Y LARGO PLAZO
# =======================================================================
def bloque_2(p):
    titulo("2. PANEL — EXPLICACIONES ALTERNATIVAS Y LARGO PLAZO (sección 5.2)")

    r = fe_dk(p, 'infl_ipc', REGS + ['infl_L1'])
    anotar('5.2', 'interacción con inflación rezagada', 0.0095,
           None if r is None else r.params['_m_x_pie'], 0.0006)

    iso = p.index.get_level_values(0)
    r = fe_dk(p[~iso.isin(['ARG', 'VEN', 'TUR', 'ZWE'])], 'infl_ipc', REGS)
    if r is not None:
        anotar('5.2', 'sin casos extremos: interacción', 0.0087,
               r.params['_m_x_pie'], 0.0006)
        anotar('5.2', 'sin casos extremos: estadístico t', 5.81,
               r.tstats['_m_x_pie'], 0.02)

    q = p.copy()
    for col, txt, nom in [('m_exceso', 0.0316, 'definición principal'),
                          ('M2_crec', 0.0315, 'M2 nominal'),
                          ('credito_crec_nom', 0.0182, 'crédito privado')]:
        if col not in q.columns:
            anotar('5.2', f'winsorizado, {nom}', txt, None)
            continue
        x = pd.to_numeric(q[col], errors='coerce')
        q['_x'] = x.clip(x.quantile(.01), x.quantile(.99))
        q['_xi'] = q['_x'] * q['infl_prev_c']
        rr = fe_dk(q, 'infl_ipc', ['_x', '_xi', 'infl_prev_c', 'r_fisher',
                                   'deficit', 'depreciacion'])
        anotar('5.2', f'winsorizado, {nom}', txt,
               None if rr is None else rr.params['_xi'], 0.0006)

    print("\n  Multiplicador de largo plazo: Σβ(impulso) / (1 − ρ)")
    M = ['m_exceso', 'm_L1', 'm_L2', 'm_L3']
    def lp(sub, etq, esperado):
        rr = fe_dk(sub, 'infl_ipc', M + ['infl_L1'] + ['r_fisher', 'deficit',
                                                       'depreciacion'])
        if rr is None:
            anotar('5.1 largo plazo', f'multiplicador, {etq}', esperado, None)
            return
        s, rho = rr.params[M].sum(), rr.params['infl_L1']
        g = np.r_[np.ones(len(M)) / (1 - rho), s / (1 - rho) ** 2]
        se = float(np.sqrt(g @ rr.cov.loc[M + ['infl_L1'],
                                          M + ['infl_L1']].values @ g))
        L = s / (1 - rho)
        print(f"    {etq:10s} Σβ {s:.3f}  ρ {rho:.3f}  LP {L:.3f}  "
              f"IC95 [{L - 1.96 * se:.3f}; {L + 1.96 * se:.3f}]  "
              f"p(LP=1) {2 * (1 - norm.cdf(abs((L - 1) / se))):.4f}  "
              f"N {int(rr.nobs)}")
        anotar('5.1 largo plazo', f'multiplicador, {etq}', esperado, L, 0.006)
    inf = pd.to_numeric(p['infl_ipc'], errors='coerce')
    lp(p, 'completa', 0.50)
    lp(p[inf < 5], 'baja', 0.03)
    lp(p[(inf >= 5) & (inf < 20)], 'moderada', 0.05)
    lp(p[inf >= 20], 'alta', 1.00)


# =======================================================================
# 3 - FISCAL
# =======================================================================
def bloque_3(p):
    titulo("3. PANEL — EXPLICACIONES FISCALES (sección 8.1)")

    r = fe_dk(p, 'infl_ipc', ['deficit', '_deuda_alta', '_def_x_alta'] + CTRL_F)
    anotar('8.1', 'déficit × deuda alta (umbral 60%)', -0.2060,
           None if r is None else r.params['_def_x_alta'], 0.002)

    r = fe_dk(p, 'infl_ipc', ['r_fisher', '_r_x_alta', '_deuda_alta',
                              'm_exceso', 'deficit', 'depreciacion'])
    if r is not None:
        anotar('8.1', 'tasa real × deuda alta', -0.7284,
               r.params['_r_x_alta'], 0.002)
        s, se, pv = suma_lineal(r, ['r_fisher', '_r_x_alta'])
        anotar('8.1', 'efecto total bajo deuda alta', -0.9252, s, 0.002)
        print(f"       (EE {se:.4f}, p {pv:.4f})")

    if '_fx' not in p.columns:
        print("\n  Sin composición por moneda: falta fx_share.")
        return
    print("\n  Composición por moneda (Arslanalp y Tsuda)")
    r = fe_dk(p, '_d_b', ['depreciacion', '_dep_x_fx', '_fx', '_def',
                          'PIB_crec_real'])
    anotar('8.1 moneda', 'validación: depreciación × fracción en dólares',
           0.179, None if r is None else r.params['_dep_x_fx'], 0.002)

    r = fe_dk(p, 'infl_ipc', ['_def', '_def_x_loc', '_loc_c'] + CTRL_F)
    anotar('8.1 moneda', 'déficit × fracción en pesos (flujo)', -0.084,
           None if r is None else r.params['_def_x_loc'], 0.005)

    r = fe_dk(p, 'infl_ipc', ['_b_local', '_b_fx', '_def'] + CTRL_F)
    if r is not None:
        anotar('8.1 moneda', 'deuda en moneda local', 0.0818,
               r.params['_b_local'], 0.0006)
        anotar('8.1 moneda', 'deuda en moneda extranjera', 0.0293,
               r.params['_b_fx'], 0.0006)
        V = r.cov
        dif = r.params['_b_local'] - r.params['_b_fx']
        se = float(np.sqrt(V.loc['_b_local', '_b_local'] +
                           V.loc['_b_fx', '_b_fx'] -
                           2 * V.loc['_b_local', '_b_fx']))
        anotar('8.1 moneda', 'diferencia', 0.0525, dif, 0.0006)
        print(f"       (EE {se:.4f}, p {2 * (1 - norm.cdf(abs(dif / se))):.4f})")


# =======================================================================
# 4 - ARGENTINA: DEMANDA DE DINERO
# =======================================================================
def bloque_4():
    titulo("4. ARGENTINA — DEMANDA DE DINERO (sección 5.3)")
    arg = leer('argentina_mensual')
    if arg is None:
        print("  Falta argentina_mensual.csv")
        return
    f = [c for c in arg.columns if 'fecha' in c.lower() or c.lower() == 'index']
    if f:
        arg.index = pd.PeriodIndex(pd.to_datetime(arg[f[0]]), freq='M')
    esperado = {'base_monetaria': -0.370, 'circulante': -0.339, 'M1': -0.213,
                'M2_privado': -0.122, 'M2': -0.119}
    if 'r_exante' not in arg.columns and 'badlar_efectiva' in arg.columns:
        arg['r_exante'] = ((1 + arg['badlar_efectiva'] / 100) /
                           (1 + arg['rem_infl_esperada'] / 100) - 1) * 100
    for agg, esp in esperado.items():
        if agg not in arg.columns:
            anotar('5.3', f'{agg}: respuesta a la expectativa', esp, None)
            continue
        m = arg.dropna(subset=[agg, 'ipc_empalmado', 'actividad',
                               'rem_infl_esperada']).copy()
        m['m_real'] = np.log(m[agg] / m['ipc_empalmado'])
        d = pd.DataFrame({
            'd_m': m['m_real'].diff(12) * 100,
            'd_act': np.log(m['actividad']).diff(12) * 100,
            'd_exp': m['rem_infl_esperada'].diff(12)})
        if 'r_exante' in m:
            d['d_r'] = m['r_exante'].diff(12)
        if 'brecha' in m:
            d['d_brecha'] = m['brecha'].diff(12)
        r = ols_hac(d['d_m'], d.drop(columns='d_m'))
        if r is None:
            anotar('5.3', f'{agg}: respuesta a la expectativa', esp, None)
            continue
        anotar('5.3', f'{agg}: respuesta a la expectativa', esp,
               r.params['d_exp'], 0.002)
        print(f"       n = {int(r.nobs)}   R² = {r.rsquared:.3f}   "
              f"brecha {r.params.get('d_brecha', float('nan')):.3f}")


# =======================================================================
# 5 - ARGENTINA: DEUDA CONSOLIDADA
# =======================================================================
def bloque_5(p_raw, moneda):
    titulo("5. ARGENTINA — DEUDA CONSOLIDADA CON EL BCRA (sección 8.2)")
    pesos = ['pases_pasivos', 'leliq_notaliq', 'lebac_nobac', 'nocom']
    ss = []
    for n in pesos:
        t = leer(f'bcra_{n}')
        if t is None:
            continue
        s = pd.Series(pd.to_numeric(t['valor'], errors='coerce').values,
                      index=pd.to_datetime(t['fecha'])).sort_index()
        ss.append(s)
    if not ss or moneda is None:
        print("  Faltan series del BCRA o la composición por moneda.")
        return
    pas = pd.concat(ss, axis=1).sum(axis=1, min_count=1)
    dic = pas[pas.index.month == 12]
    anual = dic.groupby(dic.index.year).mean()

    arg = p_raw[p_raw['iso3'] == 'ARG'].set_index('anio')
    fx = moneda[moneda['iso3'] == 'ARG'].set_index('anio')['fx_share']
    t = pd.DataFrame({'deuda': pd.to_numeric(arg['deuda_bruta_weo'],
                                             errors='coerce'),
                      'pib': pd.to_numeric(arg['PIB_nom_lcu'], errors='coerce'),
                      'fx': fx})
    t['bcra'] = 100 * anual * 1e6 / t['pib']
    t['pesos_tesoro'] = 1 - t['fx']
    t['pesos_consol'] = ((t['deuda'] * (1 - t['fx']) + t['bcra']) /
                         (t['deuda'] + t['bcra']))
    for a, esp_t, esp_c in [(2011, 0.40, 0.45), (2017, 0.31, 0.43),
                            (2019, 0.22, 0.27), (2022, 0.33, 0.42),
                            (2023, 0.28, 0.33)]:
        if a in t.index:
            anotar('8.2', f'{a}: pesos/total Tesoro', esp_t,
                   t.loc[a, 'pesos_tesoro'], 0.006)
            anotar('8.2', f'{a}: pesos/total consolidado', esp_c,
                   t.loc[a, 'pesos_consol'], 0.006)
    print()
    print(t[['deuda', 'bcra', 'pesos_tesoro', 'pesos_consol']]
          .dropna(how='all').round(3).to_string())


# =======================================================================
# 6 - WICKSELL
# =======================================================================
def _hlw():
    f = f'{DIR}/hlw_holston_laubach_williams.xlsx'
    if not os.path.exists(f):
        return None
    libro = pd.read_excel(f, sheet_name=None, header=None, engine='openpyxl')
    est = [h for h in libro if 'estimate' in h.lower()]
    t = libro[est[0]].reset_index(drop=True)
    fr = cr = None
    for i in range(min(15, len(t))):
        for j in range(t.shape[1]):
            v = str(t.iat[i, j]).lower().replace(' ', '')
            if len(v) <= 25 and ('r*' in v or v.startswith('naturalrate')):
                fr, cr = i, j
                break
        if fr is not None:
            break
    paises = t.iloc[fr + 1].astype(str).str.strip()
    cuerpo = t.iloc[fr + 2:]
    fechas = pd.to_datetime(cuerpo.iloc[:, 0], errors='coerce')
    ok = fechas.notna()
    idx = pd.PeriodIndex(fechas[ok], freq='Q')
    out = {}
    for cod, etq, hoja in [('US', 'US', 'US input data'),
                           ('CA', 'Canada', 'CA input data'),
                           ('EA', 'Euro Area', 'EA input data')]:
        col = [j for j in range(cr, t.shape[1]) if paises.iat[j] == etq]
        if not col or hoja not in libro:
            continue
        d = pd.DataFrame({'rstar': pd.to_numeric(
            cuerpo.iloc[:, col[0]][ok], errors='coerce').values}, index=idx)
        ins = libro[hoja].reset_index(drop=True)
        enc = ins.iloc[0].astype(str).str.strip().str.lower()
        ins = ins.iloc[1:].copy()
        ins.columns = enc
        fi = pd.to_datetime(ins['date'], errors='coerce')
        ins = ins[fi.notna()]
        ins.index = pd.PeriodIndex(fi[fi.notna()], freq='Q')
        for c in ['interest', 'inflation', 'inflation.expectations', 'covid.ind']:
            if c in ins.columns:
                d[c] = pd.to_numeric(ins[c], errors='coerce')
        d['brecha'] = d['interest'] - d['inflation.expectations'] - d['rstar']
        d['infl'] = d['inflation'].rolling(4).mean()
        d['covid'] = d.get('covid.ind', 0)
        out[cod] = d.dropna(subset=['brecha'])
    return out


def bloque_6():
    titulo("6. WICKSELL — TRES ECONOMÍAS (sección 6)")
    h = _hlw()
    if not h:
        print("  Falta hlw_holston_laubach_williams.xlsx")
        return
    esp = {'US': {8: -0.151, 12: -0.320, 16: -0.379, 20: -0.392},
           'CA': {8: -0.179, 12: -0.272, 16: -0.325, 20: -0.372},
           'EA': {8: -0.022, 12: -0.110, 16: -0.265, 20: -0.351}}
    nom = {'US': 'Estados Unidos', 'CA': 'Canadá', 'EA': 'Zona euro'}
    for cod, d in h.items():
        for hz, e in esp.get(cod, {}).items():
            dep = d['infl'].shift(-hz) - d['infl'].shift(1)
            X = pd.DataFrame({'brecha': d['brecha'],
                              'd_lag': d['infl'].diff().shift(1),
                              'covid': d['covid']})
            r = ols_hac(dep, X, lags=max(4, hz + 1))
            anotar('6', f'{nom[cod]}, {hz // 4} años', e,
                   None if r is None else r.params['brecha'], 0.002)


# =======================================================================
# 7 - HAYEK
# =======================================================================
def _jk(banco):
    t = leer(f'jk_shocks_{banco}')
    if t is None:
        return None
    cols = {c.lower(): c for c in t.columns}
    if 'year' in cols and 'month' in cols:
        idx = pd.PeriodIndex(year=t[cols['year']].astype(int),
                             month=t[cols['month']].astype(int), freq='M')
    else:
        cf = [c for c in t.columns if 'date' in c.lower()]
        idx = pd.PeriodIndex(pd.to_datetime(t[cf[0]]), freq='M')
    col = 'MP_median' if 'MP_median' in t.columns else None
    if col is None:
        return None
    return pd.Series(pd.to_numeric(t[col], errors='coerce').values,
                     index=idx).groupby(level=0).sum()


def _eurostat(nace):
    t = leer(f'eurostat_{nace}')
    if t is None:
        return None
    if 'unit' in t.columns:
        for u in ['I21', 'I15', 'I10']:
            if (t['unit'] == u).any():
                t = t[t['unit'] == u]
                break
    ct = [c for c in t.columns if c.lower() in ('time', 'time_period')][0]
    s = pd.Series(pd.to_numeric(t['valor'], errors='coerce').values,
                  index=pd.PeriodIndex(t[ct].astype(str), freq='M')).dropna()
    return s[~s.index.duplicated()].sort_index()


def lp(y, shock, etq, H=(0, 6, 12, 18, 24, 36), p=6):
    idx = y.index.intersection(shock.index)
    y, s = y.reindex(idx), shock.reindex(idx).fillna(0)
    s = s / s.std()
    dy = y.diff()
    ctrl = pd.concat({f'dy{k}': dy.shift(k) for k in range(1, p + 1)} |
                     {f's{k}': s.shift(k) for k in range(1, p + 1)}, axis=1)
    out = {}
    for h in H:
        r = ols_hac(y.shift(-h) - y.shift(1),
                    pd.concat([s.rename('shock'), ctrl], axis=1), lags=h + 1)
        out[h] = None if r is None else r.params['shock']
    return out


def bloque_7():
    titulo("7. HAYEK — COMPOSICIÓN DE LA PRODUCCIÓN (sección 7)")
    jf, je = _jk('fed'), _jk('ecb')

    cag, ndc, c28 = _eurostat('MIG_CAG'), _eurostat('MIG_NDCOG'), _eurostat('C28')
    if je is not None and cag is not None and ndc is not None:
        y = 100 * np.log(cag / ndc)
        for h, e in [(0, -0.70), (6, -1.11), (12, -1.18), (18, -1.18),
                     (24, -0.90), (36, -0.19)]:
            anotar('7', f'zona euro, bienes de capital, mes {h}', e,
                   lp(y, je, 'ea').get(h), 0.02)
        if c28 is not None:
            y2 = (100 * np.log(c28 / ndc)).dropna()
            for h, e in [(0, -0.48), (12, -1.57), (24, -0.90)]:
                anotar('7', f'zona euro, maquinaria, mes {h}', e,
                       lp(y2, je, 'c28').get(h), 0.02)

    us = leer('eeuu_mensual')
    if us is not None and jf is not None:
        f = [c for c in us.columns
            if c.lower() in ('fecha', 'index', 'mes') or 'fecha' in c.lower()]
        if f:
            us.index = pd.PeriodIndex(pd.to_datetime(us[f[0]]), freq='M')
        y = 100 * np.log(pd.to_numeric(us['bienes_capital'], errors='coerce') /
                         pd.to_numeric(us['consumo_no_durable'], errors='coerce'))
        for h, e in [(6, 0.42), (12, 0.45), (24, 0.24), (36, 0.39)]:
            anotar('7', f'Estados Unidos, mes {h}', e, lp(y, jf, 'us').get(h), 0.02)


# =======================================================================
def replicar(carpeta=DIR):
    global DIR
    DIR = carpeta
    FILAS.clear()
    print("=" * 86)
    print("REPLICACIÓN COMPLETA DEL TRABAJO")
    print("=" * 86)
    base = leer('panel_internacional')
    moneda = leer('moneda_deuda')
    if base is None:
        print("  Falta panel_internacional.csv: correr antes exportar_datos.py")
        return None
    p = preparar_panel(base, moneda)

    bloque_1(p)
    bloque_2(p)
    bloque_3(p)
    bloque_4()
    bloque_5(base, moneda)
    bloque_6()
    bloque_7()

    t = pd.DataFrame(FILAS)
    titulo("RESUMEN")
    print(t['estado'].value_counts().to_string())
    mal = t[t['estado'] != 'OK']
    if len(mal):
        print("\nA revisar:")
        print(mal.to_string(index=False))
    t.to_csv('replicacion_tp.csv', index=False)
    print("\n  Guardado: replicacion_tp.csv")
    return t
