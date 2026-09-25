"""
ESTIMAR_PAISES.PY
Exceso de oferta de dinero y transmision monetaria: EE.UU., zona euro,
Canada, Reino Unido (regimen bajo), Argentina, Rusia, Turquia (regimen alto).

IMPULSO PRINCIPAL: exceso nominal de oferta de dinero
    exceso_t = D12 log M_t - inflacion esperada para t (formada en t-12) - demanda real estimada_t
  La demanda real depende de la actividad, la inflacion esperada y el costo
  de oportunidad (Argentina: ademas tasa real ex ante y brecha cambiaria).
ROBUSTEZ: crecimiento del dinero menos crecimiento de la actividad ("M-actividad").
CONTROL: "Exceso+act" agrega rezagos del crecimiento de la actividad, para
  descartar que una recesion previa explique a la vez el exceso y la
  respuesta de los activos.
IV: exceso instrumentado con shocks de Jarocinski-Karadi (EE.UU., zona euro);
  el instrumento resulto debil (F < 10) en ambos, asi que no se reporta.
ESTRES (Argentina, Rusia, Turquia): 10% de meses con mayor indice de estres
  cambiario/financiero, estimado aparte por pais.

Notas por pais (para el texto, no solo para el codigo):
  - EE.UU.: costo de oportunidad = tasa de fondos federales sola (su
    rendimiento propio de M2 dejo de publicarse). Horizonte a 48 meses:
    ni asi aparece IPC; el exceso se acomoda en credito y vivienda.
  - Reino Unido: demanda de dinero con R2 bajo incluso con la mejor
    especificacion (nivel de la tasa corta, no su variacion): la demanda de
    dinero amplio britanica es conocida por su inestabilidad desde los 80.
    Entra como evidencia mas debil que el resto.
  - Argentina: IPC con San Luis incluido en 2010-2016 SOLO para esta
    comparacion (no para el cuerpo del TP). "Pasivos remunerados" es
    pases+LELIQ+LEBAC+NOCOM sobre la base monetaria (nominal/nominal),
    cortado en dic-2023 (despues empieza el desarme deliberado de jul-2024,
    reemplazo por la LEFI). La deuda en dolares (BOPREAL) queda aparte, sin
    prediccion fijada, solo de contexto.
  - Rusia: la OCDE dejo de reportar sus series despues de feb-2022; la
    muestra util es 1992-2022, treinta anios de inflacion alta sin la guerra
    actual adentro.
  - Turquia: recorte al regimen de inflacion alta, detectado en los datos
    (no a mano) y acotado en las dos puntas al periodo con produccion, M2 y
    tasa disponibles; se omite si deja menos de 36 meses (no alcanza para
    la demanda de dinero). Con la muestra actual, se omite.

Regla de lectura: RESPALDA / CONTRADICE si algun horizonte de la ventana es
significativo al 10% con el signo predicho / opuesto; MIXTO si ambos; NO
CONCLUYE si ninguno. Prediccion "sin efecto" (tipo de cambio en EE.UU.):
COMPATIBLE si no hay efecto significativo, CONTRADICE si lo hay.
"""
import time, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

DIR = 'tp_datos'
PERCENTIL_ESTRES = 0.90



# ============================================================ lectura
def leer_m(arch):
    d = pd.read_csv(f'{DIR}/{arch}', index_col=0)
    d.index = pd.PeriodIndex(d.index, freq='M')
    return d


def leer_q(arch):
    try:
        d = pd.read_csv(f'{DIR}/{arch}', index_col=0)
        d.index = pd.PeriodIndex(d.index, freq='Q')
        return d
    except FileNotFoundError:
        return pd.DataFrame()


def fred(sid, intentos=3):
    for k in range(intentos):
        try:
            t = pd.read_csv(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}')
            t.columns = ['fecha', 'v']
            return pd.Series(pd.to_numeric(t['v'], errors='coerce').values,
                             index=pd.to_datetime(t['fecha'])).dropna()
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def mensual(s):
    return s.groupby(s.index.to_period('M')).mean()


def jk(banco):
    t = pd.read_csv(f'{DIR}/jk_shocks_{banco}.csv')
    idx = pd.PeriodIndex([pd.Period(f'{a}-{m:02d}', 'M') for a, m in zip(t['year'], t['month'])])
    s = pd.Series(pd.to_numeric(t['MP_median'], errors='coerce').values, index=idx).groupby(level=0).sum()
    return -s / s.std()          # expansivo


def zscore(s):
    return (s - s.mean()) / s.std()


# ============================================================ estimación
def regresores(y, imp, controles, lags, dummies=None):
    X = pd.DataFrame({'imp': imp})
    if dummies is not None:
        X = X.join(dummies)
    dy = y.diff()
    for k in range(1, lags + 1):
        X[f'dy{k}'] = dy.shift(k)
        X[f'imp{k}'] = imp.shift(k)
        for n, c in controles.items():
            X[f'{n}{k}'] = c.shift(k)
    return X


def dep(y, h, rango):
    return y.shift(-h) - y.shift(1) if rango is None else y.shift(-rango[1]) - y.shift(-rango[0])


def lp(y, imp, controles, H, lags, rango=None, dummies=None, excluir=None, estres=None):
    X0 = regresores(y, imp, controles, lags, dummies)
    if estres is not None:
        X0['s'] = estres
        X0['imp_s'] = imp * estres
    filas = []
    for h in ([0] if rango else range(H + 1)):
        d = pd.concat([dep(y, h, rango).rename('y'), X0], axis=1).dropna()
        if excluir is not None:
            d = d[~d.index.isin(excluir)]
        d = d.loc[:, d.std() > 0] if len(d) else d
        if len(d) < 30 or 'imp' not in d:
            continue
        r = sm.OLS(d['y'], sm.add_constant(d.drop(columns='y'), has_constant='add')).fit(
            cov_type='HAC', cov_kwds={'maxlags': (rango[1] if rango else h) + 1})
        f = {'h': h, 'b': r.params['imp'], 'ee': r.bse['imp'], 'n': int(r.nobs)}
        if estres is not None and 'imp_s' in r.params:
            V = r.cov_params()
            f['b_s'] = r.params['imp'] + r.params['imp_s']
            f['ee_s'] = np.sqrt(V.loc['imp', 'imp'] + V.loc['imp_s', 'imp_s'] + 2 * V.loc['imp', 'imp_s'])
        filas.append(f)
    t = pd.DataFrame(filas).set_index('h') if filas else pd.DataFrame(columns=['b', 'ee', 'n'])
    if len(t):
        t['t'] = t['b'] / t['ee']
        if 'b_s' in t:
            t['t_s'] = t['b_s'] / t['ee_s']
    return t


def primera_etapa(x, zins, controles, lags):
    X0 = regresores(x, x, controles, lags).drop(columns=['imp'])
    X0['z'] = zins
    for k in range(1, lags + 1):
        X0[f'z{k}'] = zins.shift(k)
    d = pd.concat([x.rename('x'), X0], axis=1).dropna()
    r = sm.OLS(d['x'], sm.add_constant(d.drop(columns='x'))).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    return (r.params['z'] / r.bse['z']) ** 2


def lp_iv(y, x, zins, controles, H, lags, rango=None):
    X0 = regresores(y, x, controles, lags)
    X0['z'] = zins
    for k in range(1, lags + 1):
        X0[f'z{k}'] = zins.shift(k)
    filas = []
    for h in ([0] if rango else range(H + 1)):
        d = pd.concat([dep(y, h, rango).rename('y'), X0], axis=1).dropna()
        if len(d) < 30:
            continue
        ctr = [c for c in d.columns if c not in ('y', 'imp', 'z')]
        W = sm.add_constant(d[ctr], has_constant='add').values
        Xm = np.column_stack([d['imp'].values, W])
        Zm = np.column_stack([d['z'].values, W])
        A = np.linalg.pinv(Zm.T @ Xm)
        b = A @ Zm.T @ d['y'].values
        u = d['y'].values - Xm @ b
        g = Zm * u[:, None]
        L = (rango[1] if rango else h) + 1
        S = g.T @ g
        for l in range(1, L + 1):
            G = g[l:].T @ g[:-l]
            S += (1 - l / (L + 1)) * (G + G.T)
        V = A @ S @ A.T
        filas.append({'h': h, 'b': b[0], 'ee': np.sqrt(V[0, 0]), 'n': len(d)})
    t = pd.DataFrame(filas).set_index('h')
    t['t'] = t['b'] / t['ee']
    return t


def veredicto(res, signo, h0, h1, col='b', colt='t', zc=1.645):
    if res is None or len(res) == 0 or col not in res:
        return 'SIN DATOS', np.nan
    v = res.loc[(res.index >= h0) & (res.index <= h1)].dropna(subset=[col])
    if v.empty:
        return 'SIN DATOS', np.nan
    tmax = v.loc[v[colt].abs().idxmax(), colt]
    sig = v[colt].abs() > zc
    if signo == 0:
        return ('CONTRADICE' if (sig & (v[col] > 0)).any() else 'COMPATIBLE'), tmax
    fav = (sig & (np.sign(v[col]) == signo)).any()
    con = (sig & (np.sign(v[col]) == -signo)).any()
    return ('MIXTO' if fav and con else 'RESPALDA' if fav else 'CONTRADICE' if con else 'NO CONCLUYE'), tmax


ABREV = {'RESPALDA': 'R', 'CONTRADICE': 'C', 'NO CONCLUYE': '·', 'MIXTO': 'M',
         'SIN DATOS': '-', 'COMPATIBLE': 'ok'}


def est(t):
    t = abs(t)
    return '***' if t > 2.576 else '**' if t > 1.96 else '*' if t > 1.645 else ''


def mostrar(nombre, res, hs, col='b', colt='t'):
    if res is None or len(res) == 0:
        return
    print(f'  {nombre:26s} ' + '  '.join(f'h{h}: {res.loc[h, col]:+.2f}{est(res.loc[h, colt])}'
                                        for h in hs if h in res.index))


# ============================================================ exceso de oferta
def exceso(D, pais):
    """D: DataFrame mensual con M, P, Y, pe (inflación esperada para t, %) y regresores de demanda."""
    m_real = 100 * np.log(D['M'] / D['P'])
    dem = pd.DataFrame({'dm': m_real.diff(12)})
    for c in D.columns:
        if c.startswith('dem_'):
            dem[c] = D[c]
    dem = dem.dropna()
    rd = sm.OLS(dem['dm'], sm.add_constant(dem.drop(columns='dm'))).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    coef = ', '.join(f'{k[4:]} {v:+.3f}' for k, v in rd.params.items() if k != 'const')
    print(f'  Demanda real de dinero: {coef}; R² {rd.rsquared:.2f}; n {int(rd.nobs)} '
          f'({dem.index.min()} a {dem.index.max()})')
    fitted = sm.add_constant(dem.drop(columns='dm')) @ rd.params
    pe = 100 * np.log(1 + D['pe'] / 100)
    exc = (100 * np.log(D['M']).diff(12) - pe - fitted).dropna()
    # 'alt' no depende del IPC ni del REM: se deja con todo su rango propio,
    # no acotado al de 'exc', para que pueda usarse en pruebas que necesiten
    # más historia (ej. los pasivos remunerados, hacia atrás en el tiempo).
    alt = (100 * np.log(D['M']).diff(12) - 100 * np.log(D['Y']).diff(12)).dropna()
    pi = 100 * np.log(D['P']).diff()
    print(f'  Correlación del exceso con la inflación del mes: {exc.corr(pi.reindex(exc.index)):+.2f}')
    return exc / exc.std(), alt / alt.std(), rd





def pais_eeuu():
    D0 = leer_m('eeuu_exceso.csv')
    mich = leer_m('eeuu_expectativas.csv')['expectativa_michigan']
    D = pd.DataFrame({'M': D0['m2'], 'P': D0['cpi'], 'Y': D0['ip']})
    D['pe'] = mich.shift(12)
    D['dem_actividad'] = (100 * np.log(D0['ip'])).diff(12)
    D['dem_expectativa'] = mich.diff(12)
    D['dem_costo'] = D0['ff'].diff(12)
    D = D[(D.index >= pd.Period('1990-01', 'M')) & (D.index <= pd.Period('2024-12', 'M'))]
    # dólar: índice amplio (suba = apreciación) -> se invierte para que suba = depreciación
    a, b = fred('TWEXBMTH'), fred('DTWEXBGS')
    fx = None
    if a is not None and b is not None:
        a, b = mensual(a), mensual(b)
        sol = a.index.intersection(b.index)
        fx = pd.concat([a[a.index < b.index.min()] * (b[sol].mean() / a[sol].mean()), b])
    elif b is not None:
        fx = mensual(b)
    mens = {'tasa': ('pp', D0['ff']), 'credito': ('real', D0['credito']),
            'acciones': ('real', D0['acciones']), 'vivienda': ('real', D0['vivienda']),
            'salario': ('real', D0['salario']), 'ipc': ('log', D0['cpi'])}
    if fx is not None:
        mens['fx'] = ('log', 1 / fx)
    trim = {}
    t1, t9 = fred('WFRBST01134'), fred('WFRBSN09161')
    if t1 is not None and t9 is not None:
        top = (t1.groupby(t1.index.to_period('Q')).mean()
               + t9.groupby(t9.index.to_period('Q')).mean())
        trim['riqueza'] = ('pp', top)
    return dict(nombre='EE.UU.', regimen='bajo', D=D, mens=mens, trim=trim, P=D0['cpi'],
                iv=jk('fed'), fx_signo=0, act=100 * np.log(D0['ip']).diff())


def pais_ea():
    D0 = leer_m('pais_EA.csv')
    Q = leer_q('pais_EA_trim.csv')
    spf = Q['expectativa_spf'].dropna()
    spf_m = pd.Series({p: v for q, v in spf.items()
                       for p in pd.period_range(q.asfreq('M', 'start'), q.asfreq('M', 'end'),
                                                freq='M')})
    D = pd.DataFrame({'M': D0['dinero'], 'P': D0['ipc'], 'Y': D0['produccion']})
    D['pe'] = spf_m                                   # ya fechada por período objetivo
    D['dem_actividad'] = (100 * np.log(D0['produccion'])).diff(12)
    D['dem_expectativa'] = spf_m.shift(-12).diff(12)  # formada en t (objetivo t+12)
    D['dem_costo'] = D0['tasa_corta'].diff(12)
    D = D[(D.index >= pd.Period('1999-01', 'M')) & (D.index <= pd.Period('2024-12', 'M'))]
    mens = {'tasa': ('pp', D0['tasa_corta']), 'acciones': ('real', D0['acciones']),
            'ipc': ('log', D0['ipc']), 'fx': ('log', D0['tipo_cambio'])}
    trim = {'vivienda': ('real_ya', Q.get('vivienda_real')),
            'credito': ('pp', Q.get('credito_pib'))}
    return dict(nombre='Zona euro', regimen='bajo', D=D, mens=mens, trim=trim, P=D0['ipc'],
                iv=jk('ecb'), fx_signo=+1, act=100 * np.log(D0['produccion']).diff())


def pais_ca():
    D0 = leer_m('pais_CA.csv')
    Q = leer_q('pais_CA_trim.csv')
    infl = (100 * np.log(D0['ipc'])).diff(12)
    D = pd.DataFrame({'M': D0['dinero'], 'P': D0['ipc'], 'Y': D0['produccion']})
    D['pe'] = infl.shift(12)                          # adaptativa
    D['dem_actividad'] = (100 * np.log(D0['produccion'])).diff(12)
    D['dem_expectativa'] = infl.shift(1).diff(12)     # información disponible en t-1
    D['dem_costo'] = D0['tasa_corta'].diff(12)
    D = D[(D.index >= pd.Period('1990-01', 'M')) & (D.index <= pd.Period('2024-12', 'M'))]
    mens = {'tasa': ('pp', D0['tasa_corta']), 'acciones': ('real', D0['acciones']),
            'salario': ('real', D0['salario']), 'ipc': ('log', D0['ipc']),
            'fx': ('log', D0['tipo_cambio'])}
    trim = {'vivienda': ('real_ya', Q.get('vivienda_real')),
            'credito': ('pp', Q.get('credito_pib'))}
    return dict(nombre='Canadá', regimen='bajo', D=D, mens=mens, trim=trim, P=D0['ipc'],
                iv=None, fx_signo=+1, act=100 * np.log(D0['produccion']).diff())



def transformar(tipo, x, P):
    if tipo == 'real':
        return 100 * np.log(x) - 100 * np.log(P.reindex(x.index))
    if tipo == 'log':
        return 100 * np.log(x)
    return x                    # 'pp': en niveles


def pais_ar():
    a = pd.read_csv(f'{DIR}/argentina_mensual.csv')
    a.index = pd.PeriodIndex(pd.to_datetime(a['mes']), freq='M')
    # IPC con San Luis incluido en 2010-2016 (sin nulear por fuente_ipc), solo
    # para esta comparación de cinco países: la demanda de dinero (5.3, con
    # el REM) sigue limitada a 2016 en adelante, pero 'M-actividad' y los
    # controles/outcomes que dependen del IPC (pasivos, salario, crédito,
    # IPC como resultado) ganan la ventana 2011-2016. Distinto de la decisión
    # tomada para el cuerpo del TP, donde San Luis queda afuera: si algo de
    # esto pasa al texto, aclarar la fuente en esos años.
    if 'r_exante' not in a and 'badlar_efectiva' in a:
        a['r_exante'] = ((1 + a['badlar_efectiva'] / 100)
                         / (1 + a['rem_infl_esperada'] / 100) - 1) * 100
    D = pd.DataFrame({'M': a['M2'], 'P': a['ipc_empalmado'], 'Y': a['actividad']})
    D['pe'] = a['rem_infl_esperada'].shift(12)
    D['dem_actividad'] = (100 * np.log(a['actividad'])).diff(12)
    D['dem_expectativa'] = a['rem_infl_esperada'].diff(12)
    D['dem_tasa_real'] = a['r_exante'].diff(12)
    D['dem_brecha'] = a['brecha'].diff(12)
    pas_pesos = None
    for f in ['bcra_pases_pasivos', 'bcra_leliq_notaliq', 'bcra_lebac_nobac', 'bcra_nocom']:
        try:
            t = pd.read_csv(f'{DIR}/{f}.csv', parse_dates=['fecha']).set_index('fecha')
            x = mensual(t['valor'])
            pas_pesos = x if pas_pesos is None else pas_pesos.add(x, fill_value=0)
        except FileNotFoundError:
            print(f'  [!] pasivos: falta {f}.csv, no se suma')
    try:
        letras_me = mensual(pd.read_csv(f'{DIR}/bcra_letras_bcra_me.csv',
                                        parse_dates=['fecha']).set_index('fecha')['valor'])
    except FileNotFoundError:
        letras_me = None
    mens = {'ipc': ('log', a['ipc_empalmado']), 'expect': ('pp', a['rem_infl_esperada']),
            'tasa': ('pp', a['badlar']), 'salario': ('real', a['ripte']),
            'fx': ('log', a['dolar_contadoconliqui'])}
    if pas_pesos is not None:
        p = pas_pesos.reindex(a.index).replace(0, np.nan)
        ratio = 100 * np.log(p / a['base_monetaria'])
        # se corta en dic-2023: después empieza el desarme deliberado de los
        # pases pasivos (discontinuados en jul-2024, reemplazados por la LEFI),
        # una decisión de una vez y para siempre, no una reacción al exceso de
        # dinero mes a mes. Incluirla contaminaría cualquier horizonte que
        # toque ese período.
        ratio[ratio.index >= pd.Period('2024-01', 'M')] = np.nan
        mens['pasivos'] = ('pp', ratio)
    if letras_me is not None:
        me_usd = letras_me.reindex(a.index).replace(0, np.nan) / a['dolar_oficial']
        mens['deuda_me_usd'] = ('log', me_usd)          # contexto, sin predicción
    try:
        c = pd.read_csv(f'{DIR}/bcra_prestamos_privados.csv', index_col=0).iloc[:, 0]
        c.index = pd.PeriodIndex(c.index, freq='M')
        mens['credito'] = ('real', c)
    except FileNotFoundError:
        pass
    ie = (zscore(np.log(a['dolar_contadoconliqui']).diff())
          + zscore(np.log(a['riesgo_pais']).diff()))
    return dict(nombre='Argentina', regimen='alto', D=D, mens=mens, trim={},
                P=a['ipc_empalmado'], iv=None, fx_signo=+1, indice_estres=ie,
                dep=100 * np.log(a['dolar_oficial']).diff(),
                act=100 * np.log(a['actividad']).diff())


# ============================================================ parte 3: Turquía
# Se agrega un recorte de muestra al régimen de inflación alta, detectado en
# los datos (no a mano): el primer mes desde el cual la inflación interanual
# supera el 15% y se mantiene arriba de ese nivel al menos 12 meses seguidos,
# hasta 2024-12. Se corre además de la muestra completa, no en su lugar. La
# demanda de dinero se reestima en la submuestra.



def turquia_recorte(D0, umbral=15.0, consecutivos=12):
    """Último tramo (el más reciente) en que la inflación interanual supera
    'umbral' y se sostiene así al menos 'consecutivos' meses seguidos,
    buscado solo dentro del período donde también hay dato de producción, M2
    y tasa interbancaria, con inicio Y FIN acotados a ese período. La
    producción industrial de Turquía termina antes que el IPC; sin acotar
    el final, la búsqueda podía caer en meses sin producción disponible y
    la demanda de dinero se quedaba sin datos."""
    cols = ['ipc', 'produccion', 'dinero_m2', 'tasa_interbancaria']
    usable = D0[cols].dropna()
    if usable.empty:
        return None
    ini, fin = usable.index.min(), usable.index.max()
    infl = (100 * np.log(D0['ipc'])).diff(12)
    infl = infl[(infl.index >= ini) & (infl.index <= fin)]
    sobre = infl > umbral
    for i in range(len(sobre) - consecutivos, -1, -1):
        if sobre.iloc[i:i + consecutivos].all():
            return sobre.index[i]
    return None


def pais_tr(desde=None, sufijo=''):
    D0 = leer_m('pais_TR.csv')
    Q = leer_q('pais_TR_trim.csv')
    e = D0['expectativa_encuesta']
    D = pd.DataFrame({'M': D0['dinero_m2'], 'P': D0['ipc'], 'Y': D0['produccion']})
    D['pe'] = e.shift(12)
    D['dem_actividad'] = (100 * np.log(D0['produccion'])).diff(12)
    D['dem_expectativa'] = e.diff(12)
    D['dem_costo'] = D0['tasa_interbancaria'].diff(12)
    ini = pd.Period('2003-01', 'M') if desde is None else desde
    D = D[(D.index >= ini) & (D.index <= pd.Period('2024-12', 'M'))]
    mens = {'ipc': ('log', D0['ipc']), 'expect': ('pp', e),
            'tasa': ('pp', D0['tasa_interbancaria']), 'fx': ('log', D0['tipo_cambio'])}
    trim = {'credito': ('pp', Q.get('credito_pib'))}
    ie = (zscore(np.log(D0['tipo_cambio']).diff())
          + zscore(D0['tasa_interbancaria'].diff()))
    return dict(nombre='Turquía' + sufijo, regimen='alto', D=D, mens=mens, trim=trim,
                P=D0['ipc'], iv=None, fx_signo=+1, indice_estres=ie,
                dep=100 * np.log(D0['tipo_cambio']).diff(),
                act=100 * np.log(D0['produccion']).diff())



def predicciones(regimen, fx_signo):
    if regimen == 'bajo':
        return [('Tasa corta', 'tasa', -1, 0, 3), ('Crédito real', 'credito', +1, 0, 12),
                ('Acciones reales', 'acciones', +1, 2, 6),
                ('Vivienda real', 'vivienda', +1, 3, 12),
                ('IPC (tarde)', 'ipc', +1, 12, 48), ('Salario real', 'salario', -1, 12, 36),
                ('Riqueza top 10%', 'riqueza', +1, 3, 24),
                ('Tipo de cambio', 'fx', fx_signo, 0, 12),
                ('Acciones 12→36', 'acciones', -1, None, None),
                ('IPC 12→36', 'ipc', +1, None, None)]
    return [('IPC', 'ipc', +1, 0, 6), ('Expectativas', 'expect', +1, 0, 6),
            ('Tasa', 'tasa', +1, 0, 12), ('Pasivos remunerados', 'pasivos', +1, 0, 12),
            ('Crédito real', 'credito', -1, 3, 18), ('Salario real', 'salario', -1, 6, 18),
            ('Tipo de cambio', 'fx', +1, 0, 6)]


DETALLE_HORIZONTE = []  # cada fila: pais, variante, variable, horizonte, frecuencia, beta, t


def estimar(p):
    print('\n' + '=' * 96 + f'\n{p["nombre"].upper()} — régimen {p["regimen"]}\n' + '=' * 96)
    exc, alt, _ = exceso(p['D'], p['nombre'])
    bajo = p['regimen'] == 'bajo'
    H, L = (48, 12) if bajo else (18, 4)
    Hq, Lq = (12, 4) if bajo else (6, 2)
    P = p['P']
    pi = 100 * np.log(P).diff()
    act = p['act']
    ctrl = {'pi': pi}
    dum = None
    if not bajo:
        ctrl['dep'] = p['dep']
        dum = pd.get_dummies(exc.index.month, prefix='m', drop_first=True).astype(float)
        dum.index = exc.index
    estres = meses_estres = None
    if 'indice_estres' in p:
        ie = p['indice_estres'].reindex(exc.index)
        estres = (ie > ie.quantile(PERCENTIL_ESTRES)).astype(float).where(ie.notna())
        meses_estres = estres[estres == 1].index
        print(f'  Meses de estrés ({len(meses_estres)}): '
              + ', '.join(str(m) for m in meses_estres))
    F = None
    if p.get('iv') is not None:
        F = primera_etapa(exc, p['iv'].reindex(exc.index).fillna(0), ctrl, L)
        print(f'  Primera etapa (exceso sobre shock de Jarocinski-Karadi): F = {F:.2f}'
              + ('' if F >= 10 else '  -> débil, no se reporta la columna IV'))

    variantes = {'Exceso': exc, 'M-actividad': alt}
    res = {v: {} for v in variantes}
    res['Exceso+act'] = {}
    if F is not None and F >= 10:
        res['IV'] = {}
    if estres is not None:
        res.update({'Normal': {}, 'Estrés': {}, 'Sin estrés': {}})

    series = {}
    for k, (tipo, x) in p['mens'].items():
        if x is None:
            continue
        series[k] = ('M', transformar(tipo, x.dropna(), P))
    for k, (tipo, x) in p['trim'].items():
        if x is None or k in series:
            continue
        x = x.dropna()
        series[k] = ('Q', 100 * np.log(x) if tipo == 'real_ya' else x)

    for k, (fr, y) in series.items():
        if fr == 'M':
            c = {} if k == 'ipc' else ctrl
            ca = dict(c, act=act)
            for v, imp in variantes.items():
                res[v][k] = (lp(y, imp, c, H, L, dummies=dum), 'b', 't')
                if k in ('acciones', 'ipc') and bajo:
                    res[v][k + '_rango'] = (lp(y, imp, c, 0, L, rango=(12, 36)), 'b', 't')
            res['Exceso+act'][k] = (lp(y, exc, ca, H, L, dummies=dum), 'b', 't')
            if k in ('acciones', 'ipc') and bajo:
                r_a = lp(y, exc, ca, 0, L, rango=(12, 36))
                res['Exceso+act'][k + '_rango'] = (r_a, 'b', 't')
            if 'IV' in res:
                ins = p['iv'].reindex(exc.index).fillna(0)
                res['IV'][k] = (lp_iv(y, exc, ins, c, H, L), 'b', 't')
                if k in ('acciones', 'ipc'):
                    r_iv = lp_iv(y, exc, ins, c, 0, L, rango=(12, 36))
                    res['IV'][k + '_rango'] = (r_iv, 'b', 't')
            if estres is not None:
                inter = lp(y, exc, c, H, L, dummies=dum, estres=estres)
                res['Normal'][k] = (inter, 'b', 't')
                res['Estrés'][k] = (inter, 'b_s', 't_s')
                sin = lp(y, exc, c, H, L, dummies=dum, excluir=meses_estres)
                res['Sin estrés'][k] = (sin, 'b', 't')
        else:
            piq = pi.groupby(pi.index.asfreq('Q')).mean()
            aq = act.groupby(act.index.asfreq('Q')).sum()
            for v, imp in variantes.items():
                iq = imp.groupby(imp.index.asfreq('Q')).mean()
                res[v][k] = (lp(y, iq, {'pi': piq}, Hq, Lq), 'b', 't')
            iq = exc.groupby(exc.index.asfreq('Q')).mean()
            res['Exceso+act'][k] = (lp(y, iq, {'pi': piq, 'act': aq}, Hq, Lq), 'b', 't')
            if estres is not None:
                eq = estres.groupby(estres.index.asfreq('Q')).max()
                inter = lp(y, iq, {'pi': piq}, Hq, Lq, estres=eq)
                res['Normal'][k] = (inter, 'b', 't')
                res['Estrés'][k] = (inter, 'b_s', 't_s')

    # guardar el detalle horizonte-por-horizonte de cada variable/variante, para
    # poder graficar sin tener que transcribir a mano lo que se imprime en pantalla
    for v, variables in res.items():
        for k, (fr, y) in series.items():
            ent = variables.get(k)
            if ent is None:
                continue
            r, col_b, col_t = ent
            if col_b not in r.columns:
                continue
            for h, fila in r.iterrows():
                DETALLE_HORIZONTE.append({
                    'pais': p['nombre'], 'regimen': p['regimen'], 'variante': v,
                    'variable': k, 'frecuencia': fr, 'horizonte': h,
                    'beta': fila.get(col_b, np.nan), 't': fila.get(col_t, np.nan),
                })

    hs = (0, 3, 6, 12, 18, 24, 36, 42, 48) if bajo else (0, 3, 6, 9, 12, 18)
    hq = range(0, Hq + 1, 2 if bajo else 1)
    for var in ['Exceso', 'Exceso+act']:
        print(f'\n  Respuestas a "{var}" (1 desvío), acumuladas (%; tasas en pp)')
        for k, (fr, _) in series.items():
            mostrar(k + (' (trim.)' if fr == 'Q' else ''), res[var][k][0],
                    hs if fr == 'M' else hq)

    preds = predicciones(p['regimen'], p['fx_signo'])
    print(f'\n  VEREDICTOS — {p["nombre"]}   (R respalda, C contradice, · no concluye, '
          f'M mixto, ok compatible; entre paréntesis el mayor |t|)')
    print('  ' + ' ' * 22 + ''.join(f'{v:>15s}' for v in res))
    filas = []
    for nombre, k, sg, h0, h1 in preds:
        celdas = ''
        for v in res:
            clave = k + '_rango' if h0 is None else k
            ent = res[v].get(clave)
            if ent is None:
                txt, tt = 'SIN DATOS', np.nan
            else:
                r, col, colt = ent
                if h0 is None:
                    txt, tt = veredicto(r, sg, 0, 0, col, colt)
                elif series.get(k, ('M',))[0] == 'Q':
                    txt, tt = veredicto(r, sg, h0 // 3, int(np.ceil(h1 / 3)), col, colt)
                else:
                    txt, tt = veredicto(r, sg, h0, h1, col, colt)
            if k == 'expect' and v != 'M-actividad' and txt != 'SIN DATOS':
                txt = txt + '*'
            marca = ABREV.get(txt.rstrip('*'), '-') + ('*' if txt.endswith('*') else '')
            if not np.isnan(tt):
                marca += f' ({tt:+.1f})'
            celdas += f'{marca:>15s}'
            filas.append({'pais': p['nombre'], 'regimen': p['regimen'], 'prediccion': nombre,
                          'variante': v, 'veredicto': txt.rstrip('*'), 't': tt,
                          'cuenta': not (k == 'expect' and v != 'M-actividad')})
        print(f'  {nombre:22s}{celdas}')
    if not bajo:
        print('  (*) expectativas con el exceso: sesgada a cero por construcción, no cuenta')
    fx = res['Exceso'].get('fx')
    if fx is not None and len(fx[0]):
        r = fx[0]
        print(f'  Tipo de cambio, respuesta a 6 y 12 meses: '
              f'{r["b"].get(6, np.nan):+.2f}% / {r["b"].get(12, np.nan):+.2f}%')
    return pd.DataFrame(filas)


# ============================================================ Reino Unido y Rusia
# GB: el dinero (M3 amplio) corta en 2023-11, la OCDE dejó de publicarlo; el
#     análisis de GB queda acotado a esa fecha. Sin encuesta pública mensual
#     ni trimestral bajada todavía, se usan expectativas adaptativas, como
#     en Canadá. Si conseguís la del BoE, se puede sumar como control aparte.
# RU: el IPC, el dinero, la producción y la tasa cortan justo antes de la
#     invasión de feb-2022 (la OCDE dejó de reportar Rusia, no es un quiebre
#     dentro de la serie). No sirve para ver la economía de guerra actual;
#     sí da un régimen de inflación alta de treinta años (1992-2022) con
#     varias crisis adentro. No hay salario disponible.



def pais_gb():
    D0 = leer_m('pais_GB.csv')
    Q = leer_q('pais_GB_trim.csv')
    infl = (100 * np.log(D0['ipc'])).diff(12)
    D = pd.DataFrame({'M': D0['dinero'], 'P': D0['ipc'], 'Y': D0['produccion']})
    D['pe'] = infl.shift(12)                          # adaptativa
    D['dem_actividad'] = (100 * np.log(D0['produccion'])).diff(12)
    D['dem_expectativa'] = infl.shift(1).diff(12)
    D['dem_costo'] = D0['tasa_corta']   # nivel, no variación (espec. C:
                                          # la que mejor explica la demanda en GB)
    D = D[(D.index >= pd.Period('1987-01', 'M')) & (D.index <= pd.Period('2024-12', 'M'))]
    mens = {'tasa': ('pp', D0['tasa_corta']), 'acciones': ('real', D0['acciones']),
            'salario': ('real', D0['salario']), 'ipc': ('log', D0['ipc']),
            'fx': ('log', D0['tipo_cambio'])}
    trim = {'vivienda': ('real_ya', Q.get('vivienda_real')),
            'credito': ('pp', Q.get('credito_pib'))}
    if 'expectativa_spf' in Q:
        trim['expectativa_boe'] = ('pp', Q['expectativa_spf'])   # solo si se bajó a mano
    return dict(nombre='Reino Unido', regimen='bajo', D=D, mens=mens, trim=trim,
                P=D0['ipc'], iv=None, fx_signo=+1, act=100 * np.log(D0['produccion']).diff())



def pais_ru():
    D0 = leer_m('pais_RU.csv')
    Q = leer_q('pais_RU_trim.csv')
    infl = (100 * np.log(D0['ipc'])).diff(12)
    D = pd.DataFrame({'M': D0['dinero'], 'P': D0['ipc'], 'Y': D0['produccion']})
    D['pe'] = infl.shift(12)                          # adaptativa
    D['dem_actividad'] = (100 * np.log(D0['produccion'])).diff(12)
    D['dem_expectativa'] = infl.shift(1).diff(12)
    D['dem_costo'] = D0['tasa_corta'].diff(12)
    D = D[(D.index >= pd.Period('1994-01', 'M')) & (D.index <= pd.Period('2022-01', 'M'))]
    mens = {'ipc': ('log', D0['ipc']), 'tasa': ('pp', D0['tasa_corta']),
            'fx': ('log', D0['tipo_cambio'])}
    trim = {'credito': ('pp', Q.get('credito_pib'))}
    ie = (zscore(np.log(D0['tipo_cambio']).diff()) + zscore(D0['tasa_corta'].diff()))
    return dict(nombre='Rusia', regimen='alto', D=D, mens=mens, trim=trim, P=D0['ipc'],
                iv=None, fx_signo=+1, indice_estres=ie,
                dep=100 * np.log(D0['tipo_cambio']).diff(),
                act=100 * np.log(D0['produccion']).diff())



def correr():
    todos = []
    for f in [pais_eeuu, pais_ea, pais_ca, pais_gb, pais_ar]:
        try:
            todos.append(estimar(f()))
        except Exception as e:
            print(f'\n  [!] {f.__name__}: {type(e).__name__}: {e}')
    try:
        todos.append(estimar(pais_ru()))
    except Exception as e:
        print(f'\n  [!] pais_ru: {type(e).__name__}: {e}')
    try:
        D0tr = leer_m('pais_TR.csv')
        corte = turquia_recorte(D0tr)
        todos.append(estimar(pais_tr()))
        MIN_MESES_ALTA = 36
        if corte is not None:
            meses = (pd.Period('2024-12', 'M') - corte).n + 1
            if meses >= MIN_MESES_ALTA:
                todos.append(estimar(pais_tr(desde=corte, sufijo=' (alta)')))
            else:
                print(f'\n  [!] Turquía (alta): el régimen detectado desde {corte} '
                      f'deja solo {meses} meses hasta 2024-12, no alcanza; se omite.')
        else:
            print('\n  [!] Turquía: no se detectó un tramo de inflación alta sostenida')
    except Exception as e:
        print(f'\n  [!] pais_tr: {type(e).__name__}: {e}')

    v = pd.concat(todos)
    v.to_csv('cinco_paises_veredictos.csv', index=False)
    pd.DataFrame(DETALLE_HORIZONTE).to_csv('respuestas_horizonte.csv', index=False)

    print('\n' + '=' * 96 + '\nRESUMEN\n' + '=' * 96)
    paises = ['EE.UU.', 'Zona euro', 'Canadá', 'Reino Unido', 'Argentina',
              'Rusia', 'Turquía', 'Turquía (alta)']
    orden = ([n for n, *_ in predicciones('bajo', 0)]
             + [n for n, *_ in predicciones('alto', 1)])
    for var in ['Exceso', 'Exceso+act', 'M-actividad']:
        x = v[v['variante'] == var]
        tab = x.pivot_table(index='prediccion', columns='pais', values='veredicto',
                            aggfunc='first')
        tab = tab.reindex(columns=[c for c in paises if c in tab])
        tab = tab.reindex([o for o in dict.fromkeys(orden) if o in tab.index]).fillna('')
        print(f'\n  {var}:')
        print(tab.replace(ABREV).to_string())
    print('\n  (En "Exceso" y "Exceso+act", la fila Expectativas no cuenta.)')
    print('\n  Guardado: cinco_paises_veredictos.csv y respuestas_horizonte.csv')
    return v


res = correr()
