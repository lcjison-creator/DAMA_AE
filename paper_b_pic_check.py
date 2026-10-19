# %% [markdown]
# # Paper B: external check in the PIC database
#
# **Purpose.** Post hoc check for the paper on deaths after discharge against medical advice (DAMA). Does a public pediatric ICU
# database, the Paediatric Intensive Care (PIC) database v1.1.0 (Children's Hospital, Zhejiang University School of Medicine; ICU
# admissions 2010-2018), record anything that would reveal deaths after DAMA? And how many children recorded as discharged alive had
# a vasoactive drug ordered on the day of discharge or the day before, the definition whose accuracy is measured in the Wuhan cohort
# (paper_b_dama_analysis, "PIC-compatible" definition)?
#
# **Data access.** PIC is credentialed-access data on PhysioNet (PhysioNet Credentialed Health Data License 1.5.0). This notebook is
# run only on the credentialed author's computer. It prints and saves aggregate counts only; no row of the database is shown.
#
# **How to run.** Put this notebook in the folder that holds the PIC files (subfolders are searched; .csv or .csv.gz) and run all
# cells. Results go to `output_pic_paper_b/`: `pic_summary.md` (counts of admissions from 1 to 4 shown as "<5"; the file to share),
# `etable_pic.csv` (same table, same masking) and `pic_numbers.json`.
#
# **Definitions** (as in the Wuhan notebook):
# - *Vasoactive order near discharge*: an order for dopamine, dobutamine, norepinephrine, milrinone, or vasopressin placed on the
#   discharge date or the day before, or an earlier order still running on one of those days (end date on or after the day before
#   discharge). Epinephrine is reported separately because PIC records no route, so nebulized or bolus epinephrine cannot be told
#   apart from an infusion. Desmopressin, terlipressin, posterior pituitary injection, isoproterenol, phenylephrine, and levosimendan
#   are not counted.
# - *Straight from an ICU*: the last ICU exit within 1 hour of hospital discharge.
# - *Beyond the neonatal period*: aged 28 days or older at admission, the age range of the Wuhan cohort.
# - *First-listed diagnosis*: ICD-10 chapter of the coded diagnosis (PIC holds 1 coded diagnosis per admission).

# %%
import json
import platform
import re
import time
from pathlib import Path

import pandas as pd

VERSION = 'paper_b_pic_check v1.1 (2026-10-19)'
HERE = Path.cwd()
OUT = HERE / 'output_pic_paper_b'
OUT.mkdir(exist_ok=True)
T0 = time.time()
NUMS = {'version': VERSION}
LINES = []


def say(text=''):
    print(text)
    LINES.append(str(text))


def small(n):
    """Counts of admissions or patients from 1 to 4 are shown as '<5'."""
    n = int(n)
    return '<5' if 0 < n < 5 else str(n)


def md_table(df):
    cols = [str(c) for c in df.columns]
    out = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join('---' for _ in cols) + ' |']
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(str(v) for v in r.values) + ' |')
    return '\n'.join(out)


TABLES = ['ADMISSIONS', 'PATIENTS', 'ICUSTAYS', 'PRESCRIPTIONS', 'DIAGNOSES_ICD', 'D_ICD_DIAGNOSES', 'D_ITEMS', 'CHARTEVENTS']
SOURCES = {}
for p in sorted(HERE.rglob('*')):
    if OUT in p.parents or not p.is_file():
        continue
    base = p.name.upper()
    for ext in ('.CSV.GZ', '.CSV'):
        if base.endswith(ext) and base[: -len(ext)] in TABLES:
            SOURCES.setdefault(base[: -len(ext)], p)
missing = [t for t in TABLES if t not in SOURCES]
if missing:
    raise FileNotFoundError('PIC tables not found in this folder or its subfolders: ' + ', '.join(missing))


def read(table, usecols=None):
    want = None if usecols is None else {c.upper() for c in usecols}
    for enc in ('utf-8-sig', 'gb18030'):
        try:
            df = pd.read_csv(SOURCES[table], encoding=enc, dtype=str, low_memory=False,
                             usecols=None if want is None else (lambda c: c.strip().upper() in want))
            df.columns = [c.strip().upper() for c in df.columns]
            for c in ('SUBJECT_ID', 'HADM_ID', 'ITEMID'):
                if c in df.columns:
                    df[c] = df[c].str.strip()
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f'{table}: neither UTF-8 nor GB18030')


def dt(x):
    x = x.astype('string').str.strip()
    out = pd.to_datetime(x, errors='coerce', format='%Y-%m-%d %H:%M:%S')
    bad = out.isna() & x.notna() & x.ne('')
    if bad.mean() > 0.01:
        try:
            alt = pd.to_datetime(x[bad], errors='coerce', format='mixed')
        except (ValueError, TypeError):
            alt = pd.to_datetime(x[bad], errors='coerce')
        out = out.where(~bad, alt)
    return out


def s(df, c):
    return df[c].fillna('').astype(str) if c in df.columns else pd.Series('', index=df.index, dtype=object)


# %% [markdown]
# ## 1. What PIC records about deaths and organ support

# %%
ADM = read('ADMISSIONS')
for c in ('ADMITTIME', 'DISCHTIME', 'DEATHTIME'):
    ADM[c] = dt(ADM[c])
ADM['DIED'] = pd.to_numeric(ADM.HOSPITAL_EXPIRE_FLAG, errors='coerce').eq(1)
PAT = read('PATIENTS', usecols=['SUBJECT_ID', 'DOB', 'DOD', 'EXPIRE_FLAG'])
PAT['DOB'], PAT['DOD'] = dt(PAT.DOB), dt(PAT.DOD)
ADM = ADM.merge(PAT[['SUBJECT_ID', 'DOB']], on='SUBJECT_ID', how='left')
ADM['AGE_DAYS'] = (ADM.ADMITTIME - ADM.DOB).dt.total_seconds() / 86400

disp_cols = [c for c in ADM.columns if re.search(r'LOCATION|DISPOSITION|OUTCOME|DISCHARGE_TYPE|LEAVE', c)]
any_death = ADM.groupby('SUBJECT_ID').DIED.max()
dod_outside = PAT.set_index('SUBJECT_ID').DOD.notna() & ~any_death.reindex(PAT.SUBJECT_ID).fillna(False).astype(bool).values
D_ITEMS = read('D_ITEMS')
charted = set(read('CHARTEVENTS', usecols=['ITEMID']).ITEMID.dropna().unique())
labels = (s(D_ITEMS, 'LABEL_CN') + ' | ' + s(D_ITEMS, 'LABEL'))[D_ITEMS.ITEMID.isin(charted)]
vent_terms = r'呼吸机|机械通气|有创|潮气量|呼气末正压|气道|插管|气管|SIMV|PEEP|tidal|ventilat|intubat|endotrache|airway'
NUMS['database'] = {
    'admissions': int(len(ADM)), 'patients': int(ADM.SUBJECT_ID.nunique()), 'in_hospital_deaths': int(ADM.DIED.sum()),
    'discharged_alive': int((~ADM.DIED).sum()), 'disposition_columns': disp_cols,
    'patients_with_date_of_death': int(PAT.DOD.notna().sum()),
    'dates_of_death_without_in_hospital_death': int(dod_outside.sum()),
    'items_charted_in_chartevents': int(len(charted)),
    'charted_items_with_ventilation_terms': int(labels.str.contains(vent_terms, case=False, regex=True).sum()),
    'admissions_with_negative_age': int((ADM.AGE_DAYS < 0).sum())}
d = NUMS['database']
say(f'# PIC check for paper B: summary\n\n{VERSION}. Run on {time.strftime("%Y-%m-%d %H:%M")}; Python {platform.python_version()}, '
    f'pandas {pd.__version__}. Aggregate counts only; counts of admissions from 1 to 4 are shown as "<5".\n')
say('## What PIC records\n')
say(f"- Admissions {d['admissions']}; patients {d['patients']}; died in hospital {d['in_hospital_deaths']} "
    f"({100 * d['in_hospital_deaths'] / d['admissions']:.1f}%); discharged alive {d['discharged_alive']}.")
say(f"- Columns that could hold a discharge disposition: {', '.join(disp_cols) if disp_cols else 'none'}.")
say(f"- Patients with a date of death: {d['patients_with_date_of_death']}; dates of death without an in-hospital death: "
    f"{small(d['dates_of_death_without_in_hospital_death']) if d['dates_of_death_without_in_hospital_death'] else 0}.")
say(f"- Items charted in CHARTEVENTS: {d['items_charted_in_chartevents']}; with a ventilation or airway term in the label: "
    f"{d['charted_items_with_ventilation_terms']}.")
say(f"- Admissions with a negative age (date error), left out of the age groups: {small(d['admissions_with_negative_age'])}.")

# %% [markdown]
# ## 2. Vasoactive orders near discharge

# %%
AGENTS = [  # order matters: names that contain another agent's name come first
    ('phenylephrine', r'去氧肾上腺素|苯肾上腺素|phenylephrine'), ('isoproterenol', r'异丙肾上腺素|isoprenaline|isoproterenol'),
    ('norepinephrine', r'去甲肾上腺素|norepinephrine|noradrenaline'), ('epinephrine', r'肾上腺素|epinephrine|adrenaline'),
    ('dobutamine', r'多巴酚丁胺|dobutamine'), ('dopamine', r'多巴胺|dopamine'), ('milrinone', r'米力农|milrinone'),
    ('desmopressin', r'去氨加压素|desmopressin'), ('terlipressin', r'特利加压素|terlipressin'), ('pituitrin', r'垂体后叶|pituitrin'),
    ('vasopressin', r'血管加压素|精氨酸加压素|加压素|vasopressin'), ('levosimendan', r'左西孟旦|levosimendan')]
COUNTED = ['dopamine', 'dobutamine', 'norepinephrine', 'milrinone', 'vasopressin']
RX = read('PRESCRIPTIONS', usecols=['HADM_ID', 'STARTDATE', 'ENDDATE', 'STARTTIME', 'ENDTIME', 'DRUG_NAME', 'DRUG_NAME_CN', 'DRUG_NAME_EN',
                                    'DRUG_NAME_GENERIC'])
name = s(RX, 'DRUG_NAME') + ' | ' + s(RX, 'DRUG_NAME_CN') + ' | ' + s(RX, 'DRUG_NAME_EN') + ' | ' + s(RX, 'DRUG_NAME_GENERIC')
agent = pd.Series(None, index=RX.index, dtype=object)
for a, pat in AGENTS:
    agent[agent.isna() & name.str.contains(pat, case=False, regex=True)] = a
RX = RX.assign(AGENT=agent)[agent.notna()].copy()
start_col = 'STARTDATE' if 'STARTDATE' in RX.columns else 'STARTTIME'
end_col = 'ENDDATE' if 'ENDDATE' in RX.columns else 'ENDTIME'
RX['START'], RX['END'] = dt(RX[start_col]), (dt(RX[end_col]) if end_col in RX.columns else pd.NaT)
dis_day = ADM.set_index('HADM_ID').DISCHTIME.dt.normalize().rename('DIS_DAY')
r = RX.join(dis_day, on='HADM_ID')
st, en, d0 = r.START.dt.normalize(), r.END.dt.normalize(), r.DIS_DAY - pd.Timedelta(days=1)
near = (st <= r.DIS_DAY) & (((st >= d0) & (st <= r.DIS_DAY)) | (en >= d0))
counted = r.AGENT.isin(COUNTED)
NUMS['orders'] = {'counted_agent_orders': int(counted.sum()), 'counted_agent_admissions': int(r.loc[counted, 'HADM_ID'].nunique()),
                  'counted_orders_with_end_date_pct': round(100 * float(r.loc[counted, 'END'].notna().mean()), 1) if counted.any() else None,
                  'start_column': start_col, 'end_column': end_col,
                  'orders_by_agent': {k: int(v) for k, v in r.AGENT.value_counts().items()}}
ADM['VASO'] = ADM.HADM_ID.isin(set(r.loc[near & counted, 'HADM_ID']))
ADM['EPI'] = ADM.HADM_ID.isin(set(r.loc[near & r.AGENT.eq('epinephrine'), 'HADM_ID']))

ICU = read('ICUSTAYS', usecols=['HADM_ID', 'OUTTIME', 'FIRST_CAREUNIT', 'LAST_CAREUNIT'])
ICU['OUTTIME'] = dt(ICU.OUTTIME)
last_out = ICU.groupby('HADM_ID').OUTTIME.max()
gap_h = (ADM.DISCHTIME - ADM.HADM_ID.map(last_out)).dt.total_seconds() / 3600
ADM['ICU_DIRECT'] = gap_h.between(-1, 1)

DX = read('DIAGNOSES_ICD')
DDX = read('D_ICD_DIAGNOSES').drop_duplicates('ICD10_CODE_CN')
DX = DX.merge(DDX[['ICD10_CODE_CN', 'ICD10_CODE', 'TITLE_CN']], on='ICD10_CODE_CN', how='left')
DX['SEQ'] = pd.to_numeric(DX.SEQ_NUM, errors='coerce')
first = DX.sort_values('SEQ').drop_duplicates('HADM_ID').set_index('HADM_ID')
CHAPTERS = [('A', 'B', 'A-B Infectious'), ('C', 'C', 'C-D48 Neoplasms'), ('E', 'E', 'E Endocrine, metabolic'), ('G', 'G', 'G Nervous system'),
            ('I', 'I', 'I Circulatory'), ('J', 'J', 'J Respiratory'), ('K', 'K', 'K Digestive'), ('N', 'N', 'N Genitourinary'),
            ('P', 'P', 'P Perinatal'), ('Q', 'Q', 'Q Congenital'), ('R', 'R', 'R Symptoms, signs'), ('S', 'T', 'S-T Injury, poisoning')]


def chapter(code):
    code = code.strip().upper() if isinstance(code, str) else ''
    if not code:
        return 'No coded diagnosis'
    c = code[0]
    if c == 'D':
        digits = re.sub(r'[^0-9]', '', code[1:3])
        return 'C-D48 Neoplasms' if digits and int(digits) < 50 else 'D50-D89 Blood, immune'
    for lo, hi, nm in CHAPTERS:
        if lo <= c <= hi:
            return nm
    return 'Other'


code = first.ICD10_CODE.where(first.ICD10_CODE.notna(), first.ICD10_CODE_CN)
ADM['CHAPTER'] = ADM.HADM_ID.map(code).map(chapter).fillna('No coded diagnosis')

# Acute encephalopathy as in the Wuhan cohort (Chinese title patterns on the coded diagnosis), beyond the neonatal period
ACUTE = [r'急性脑病综合征', r'中毒性脑病(?!后遗症)', r'^脑病$', r'缺氧缺血性脑病(?!后遗症)|缺血缺氧性脑病(?!后遗症)', r'代谢性脑病', r'高血压脑病',
         r'可逆性后部', r'低血糖性脑病', r'狼疮性脑病', r'百日咳脑病', r'自身免疫性脑炎|自体免疫性脑炎|抗NMDA|自身免疫相关性脑病|急性播散性脑脊髓炎',
         r'病毒性脑炎|脑炎(?!恢复期)', r'心脏停搏|心搏骤停|心跳骤停|心脏骤停', r'溺水', r'癫痫持续状态', r'脓毒|败血', r'坏死性脑病']
title = ADM.HADM_ID.map(first.TITLE_CN).fillna('').str.strip()
index_dx = title.str.contains(r'脑病|脑炎|脑脊髓炎', regex=True)
acute = pd.Series(False, index=ADM.index)
for p in ACUTE:
    acute |= title.str.contains(p, regex=True)
sequelae = title.str.contains('后遗症') | title.str.contains('康复')
ADM['AGE_OK'] = ADM.AGE_DAYS.ge(0)
ADM['BEYOND_NEONATAL'] = ADM.AGE_DAYS.ge(28) & ADM.AGE_DAYS.lt(18 * 365.25)
dept = s(ADM, 'DISCHARGE_DEPARTMENT')
nicu_stay = set(ICU.loc[(s(ICU, 'FIRST_CAREUNIT') + ' ' + s(ICU, 'LAST_CAREUNIT')).str.contains(r'NICU', case=False, regex=True), 'HADM_ID'])
neonatal = (ADM.HADM_ID.isin(nicu_stay) | s(ADM, 'ADMISSION_DEPARTMENT').str.contains(r'NICU|新生儿|Neonat', case=False, regex=True))
ADM['AE'] = index_dx & acute & ~sequelae & ADM.BEYOND_NEONATAL & ~neonatal   # Wuhan rule: neonatal = <28 days or a neonatal unit


def row(nm, g):
    al = g[~g.DIED]
    n_died, n_vaso = int(g.DIED.sum()), int(al.VASO.sum())
    return {'Group': nm, 'Admissions': len(g), 'Died in hospital': n_died, 'Discharged alive': len(al),
            'Alive, vasoactive order near discharge': n_vaso, 'Of whom straight from an ICU': int((al.VASO & al.ICU_DIRECT).sum()),
            'Alive, epinephrine order near discharge': int(al.EPI.sum()),
            # no ratio when either count is below 5, so a masked count cannot be recovered from it
            'Vasoactive near discharge per 100 in-hospital deaths': '' if (n_died < 5 or 0 < n_vaso < 5) else f'{100 * n_vaso / n_died:.0f}'}


rows = [row('All admissions', ADM),
        row('Age under 28 days', ADM[ADM.AGE_OK & ADM.AGE_DAYS.lt(28)]),
        row('Beyond the neonatal period (28 days to <18 y)', ADM[ADM.BEYOND_NEONATAL])]
icu_dept = dept.str.contains(r'ICU', case=False, regex=True)
for k in sorted(dept[icu_dept].unique()):
    rows.append(row(f'Discharged from {k}', ADM[dept.eq(k)]))
rows.append(row('Discharged from a non-ICU department', ADM[~icu_dept]))
B = ADM[ADM.BEYOND_NEONATAL]
for k, g in B.groupby('CHAPTER'):
    rows.append(row(f'Beyond the neonatal period, first-listed diagnosis {k}', g))
rows.append(row('Acute encephalopathy (Wuhan definition; beyond the neonatal period)', ADM[ADM.AE]))
TAB = pd.DataFrame(rows)
NUMS['table'] = TAB.to_dict('records')
masked = TAB.drop(columns=['Discharged alive'])          # admissions minus deaths would reveal a masked death count
for c in ['Admissions', 'Died in hospital', 'Alive, vasoactive order near discharge', 'Of whom straight from an ICU',
          'Alive, epinephrine order near discharge']:
    masked[c] = masked[c].map(small)
masked.to_csv(OUT / 'etable_pic.csv', index=False)
o = NUMS['orders']
say('\n## Vasoactive orders near discharge\n')
say(f"Orders of the counted agents: {o['counted_agent_orders']} in {o['counted_agent_admissions']} admissions; order times from "
    f"{o['start_column']} and {o['end_column']}; end date present for {o['counted_orders_with_end_date_pct']}% of counted orders.")
say('Orders by agent (all matched names): ' + '; '.join(f'{k} {v}' for k, v in o['orders_by_agent'].items()) + '.\n')
say(md_table(masked))

# %% [markdown]
# ## 3. Save

# %%
NUMS['run_time_s'] = round(time.time() - T0)
(OUT / 'pic_numbers.json').write_text(json.dumps(NUMS, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
(OUT / 'pic_summary.md').write_text('\n'.join(LINES) + '\n', encoding='utf-8')
print(f'\nWritten to {OUT.name}/: pic_summary.md (share this file), etable_pic.csv, pic_numbers.json.')
