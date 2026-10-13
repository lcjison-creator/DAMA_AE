# %% [markdown]
# # Paper B: discharge against medical advice and hidden mortality in pediatric acute encephalopathy
#
# Reproducible analysis on the frozen dataset **AE_freeze_v1.2**. Put this notebook in a folder that contains `AE_freeze_v1.2/`
# and run all cells. Every number in the manuscript draft comes from this notebook.
#
# Optional inputs, used automatically once they are filled in (same folder, or `output_dama_followup/`):
# * `dama_phone_followup_INTERNAL.xlsx` (completed call list): post-discharge vital status, linked by patient_id and used only where the
#   guardian gave verbal consent; gives the accuracy of the critical label against death within 30 days and the follow-up-corrected
#   mortality.
# * `dama_second_review_INTERNAL.xlsx` (optional blinded second review): inter-rater agreement. Not used in the current analysis.
#
# Study period: admissions from April 2020 (records from July 2019 to March 2020 were deleted from the hospital's data platform during
# the COVID-19 pandemic; the six earlier months of 2019 enter a sensitivity analysis only).
#
# Outputs go to `output_paper_b/`: tables (CSV), figures (`figures/`, PNG 600 dpi and PDF), `numbers.json` and `summary.md`.
# No row-level identifiers are written to the output folder.

# %%
import sys
import json
import re
import math
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path.cwd()
ROOT = HERE / 'AE_freeze_v1.2'
if not (ROOT / 'manifest.json').exists():
    raise FileNotFoundError('Put this notebook in the folder that contains AE_freeze_v1.2/.')
sys.path.insert(0, str(ROOT / 'code'))
from load_frozen import load, verify  # noqa: E402

OUT = HERE / 'output_paper_b'
OUT.mkdir(exist_ok=True)
CACHE = HERE / 'cache_frozen_tables'
FOLLOWUP_DAYS = 30
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)
NUM = {}            # key numbers for the manuscript
SUMMARY = []


def note(text):
    print(text)
    SUMMARY.append(text)


problems = verify(verbose=False)
note('Frozen dataset: ' + ('intact (every file matches manifest.json)' if not problems else 'PROBLEMS: ' + '; '.join(problems)))
L = lambda name: load(name, cache_dir=str(CACHE))  # noqa: E731
adm = L('admission_summary').set_index('visit_id')
admx = L('admissions').set_index('visit_id')
pat = L('patients').set_index('patient_id')
dxc = L('diagnoses_coded')

# %% [markdown]
# ## 1. Cohorts
# Supplementary-export admissions are the export's index admissions (encephalopathy-related diagnoses) and the only ones with a
# recorded disposition. They are split into mutually exclusive groups from age, admission ward and coded discharge diagnoses:
# neonatal (<28 days or admitted to a neonatal ward), rehabilitation, sequelae without an acute code, **acute encephalopathy beyond the
# neonatal period** (primary cohort: any acute encephalopathy-related code), and other non-acute admissions (mostly epileptic
# encephalopathy).

# %%
sup = admx[admx.in_supplement == True].copy()
d = dxc[dxc.visit_id.isin(sup.index)]


def has(pattern):
    return sup.index.isin(d[d.diagnosis_cn.str.contains(pattern, na=False, regex=True)].visit_id.unique())


ETIOLOGY = {   # non-exclusive labels from coded discharge diagnoses (Chinese patterns, English labels)
    'Acute encephalopathy syndrome or encephalopathy, unspecified': r'急性脑病综合征|^脑病$',
    'Toxic (infection-associated) encephalopathy': r'中毒性脑病(?!后遗症)',
    'Status epilepticus': r'癫痫持续状态',
    'Sepsis': r'脓毒|败血',
    'Hypoxic-ischemic encephalopathy': r'缺氧缺血性脑病(?!后遗症)|缺血缺氧性脑病(?!后遗症)',
    'Cardiac arrest': r'心脏停搏|心搏骤停|心跳骤停|心脏骤停',
    'Drowning': r'溺水',
    'Influenza': r'流感|流行性感冒',
    'Mycoplasma infection': r'支原体',
    'Viral or autoimmune encephalitis': r'病毒性脑炎|脑炎(?!恢复期)|自身免疫性脑炎|自体免疫性脑炎|抗NMDA|急性播散性脑脊髓炎',
    'Metabolic, hypoglycemic or hypertensive encephalopathy, PRES': r'代谢性脑病|低血糖性脑病|高血压脑病|可逆性后部',
    'Lupus encephalopathy': r'狼疮性脑病',
}
ACUTE = [r'急性脑病综合征', r'中毒性脑病(?!后遗症)', r'^脑病$', r'缺氧缺血性脑病(?!后遗症)|缺血缺氧性脑病(?!后遗症)', r'代谢性脑病', r'高血压脑病',
         r'可逆性后部', r'低血糖性脑病', r'狼疮性脑病', r'百日咳脑病', r'自身免疫性脑炎|自体免疫性脑炎|抗NMDA|自身免疫相关性脑病|急性播散性脑脊髓炎',
         r'病毒性脑炎|脑炎(?!恢复期)', r'心脏停搏|心搏骤停|心跳骤停|心脏骤停', r'溺水', r'癫痫持续状态', r'脓毒|败血', r'坏死性脑病']
neonatal = (sup.age_days < 28).values | (sup.admit_unit == 'neonatal_ward').values
rehab = has(r'康复')
sequelae_only = has(r'后遗症') & ~(has(r'急性脑病综合征') | has(r'中毒性脑病(?!后遗症)') | has(r'^脑病$') | has(r'癫痫持续状态'))
acute_any = np.zeros(len(sup), dtype=bool)
for p in ACUTE:
    acute_any |= has(p)
cohort = np.select([neonatal, rehab, sequelae_only, acute_any], ['Neonatal encephalopathy', 'Rehabilitation', 'Sequelae', 'Acute encephalopathy'],
                   default='Other non-acute')
A = adm.loc[sup.index].copy()
A['cohort'] = cohort
for k, p in ETIOLOGY.items():
    A[k] = has(p)
A['epileptic_encephalopathy_code'] = has(r'癫痫性脑病')   # describes the 'Other non-acute' group in Figure 1
A = A.join(admx[['age_years', 'admission_source', 'discharge_unit', 'admit_unit']], rsuffix='_adm')
A['residence'] = A.patient_id.map(pat.residence_wuhan).map({True: 'Wuhan', False: 'Outside Wuhan'}).fillna('Not recorded')
A['ethnicity'] = A.patient_id.map(pat.ethnicity_cn).map(lambda v: 'Han' if v == '汉族' else ('Other' if isinstance(v, str) and v.strip() else 'Not recorded'))
A['outcome_group'] = np.select(
    [A.death.eq(True).fillna(False).astype(bool), A.dama_critical_final.eq(True).fillna(False).astype(bool), A.dama.eq(True).fillna(False).astype(bool), A.disposition.notna()],
    ['In-hospital death', 'Critical DAMA', 'Not-critical DAMA', 'Discharged alive'], default='Disposition not recorded')
A['composite'] = A.death.eq(True).fillna(False).astype(bool) | A.dama_critical_final.eq(True).fillna(False).astype(bool)
A['picu'] = A.any_picu_stay.eq(True).fillna(False).astype(bool)
A['period'] = pd.cut(A.admit_year, [2019, 2021, 2023, 2026], labels=['2020-2021', '2022-2023', '2024-2026'])   # study period starts April 2020
A['age_group'] = pd.cut(A.age_years, [-0.01, 1, 3, 6, 12, 19], right=False, labels=['<1 y', '1 to <3 y', '3 to <6 y', '6 to <12 y', '12 to 18 y'])
A['los_days'] = A.los_hours / 24

# Patients aged 18 years or older are adult women treated by the hospital's maternal service: not part of a pediatric study.
NUM['n_export_admissions'] = int(len(A)); NUM['n_export_patients'] = int(A.patient_id.nunique())
adult_mask = A.age_years >= 18
NUM['n_adult_excluded'] = int(adult_mask.sum()); NUM['n_adult_excluded_primary'] = int((adult_mask & (A.cohort == 'Acute encephalopathy')).sum())
note(f"Export: {len(A):,} index admissions ({A.patient_id.nunique():,} patients); excluded {int(adult_mask.sum())} admissions of patients aged "
     f"18 years or older (adult women of the maternal service; {NUM['n_adult_excluded_primary']} of them had acute-encephalopathy codes).")
A = A[~adult_mask].copy()

# Study period. Records of admissions from July 2019 to March 2020 were deleted from the hospital's data platform during the COVID-19
# pandemic and cannot be re-exported. The analysis uses the continuous period from April 2020; the six earlier months (January to
# June 2019) are kept only for a sensitivity analysis.
STUDY_START = pd.Timestamp('2020-04-01')
A_ALL = A.copy()
early = A.admit_time < STUDY_START
NUM['study_start'] = f'{STUDY_START:%B} {STUDY_START.year}'
NUM['n_before_study_start'] = int(early.sum())
NUM['n_before_study_start_primary'] = int((early & (A.cohort == 'Acute encephalopathy')).sum())
note(f"Study period from {NUM['study_start']}: excluded {NUM['n_before_study_start']} index admissions admitted earlier "
     f"({NUM['n_before_study_start_primary']} in the primary cohort; kept for a sensitivity analysis).")
A = A[~early].copy()

# Coverage of the export: calendar months without any admission (main export, all admissions of the included children)
def month_ranges(months):
    out, run = [], []
    for m in months:
        if run and m != run[-1] + 1:
            out.append(run); run = []
        run.append(m)
    if run:
        out.append(run)
    return [f'{r[0]}' if len(r) == 1 else f'{r[0]} to {r[-1]}' for r in out]


study_months = pd.period_range(A.admit_time.min().to_period('M'), A.admit_time.max().to_period('M'), freq='M')
present = set(admx.admit_time.dt.to_period('M').dropna().unique())
GAP = [m for m in study_months if m not in present]
NUM['months_in_study_period'] = len(study_months); NUM['months_without_admissions'] = month_ranges(GAP)
NUM['months_covered'] = len(study_months) - len(GAP)
note(f"Study period {study_months[0]} to {study_months[-1]} ({len(study_months)} months); months without any admission in the export: "
     + (', '.join(month_ranges(GAP)) + f' ({len(GAP)} months)' if GAP else 'none') + f"; months covered {NUM['months_covered']}.")
GAP_YEARS = sorted({m.year for m in GAP})

flow = A.groupby('cohort').agg(admissions=('patient_id', 'size'), children=('patient_id', 'nunique'), picu=('picu', 'sum'),
                               deaths=('death', 'sum'), dama=('dama', 'sum'), critical_dama=('dama_critical_final', 'sum'))
flow.loc['All index admissions'] = [len(A), A.patient_id.nunique(), int(A.picu.sum()), int(A.death.sum()), int(A.dama.sum()), int(A.dama_critical_final.sum())]
flow = flow.astype(int)
flow.to_csv(OUT / 'cohort_flow.csv')
print(flow)
P = A[A.cohort == 'Acute encephalopathy'].copy()          # primary cohort
NEO = A[A.cohort == 'Neonatal encephalopathy'].copy()
P_ALL = A_ALL[A_ALL.cohort == 'Acute encephalopathy'].copy()   # including January to June 2019 (sensitivity analysis)
NUM['n_index_admissions'] = int(len(A)); NUM['n_index_children'] = int(A.patient_id.nunique())
NUM['n_primary_admissions'] = int(len(P)); NUM['n_primary_children'] = int(P.patient_id.nunique())
NUM['years'] = f'{int(A.admit_year.min())}-{int(A.admit_year.max())}'
note(f"Index admissions {len(A):,} ({A.patient_id.nunique():,} children), {NUM['years']}; primary cohort (acute encephalopathy beyond the "
     f"neonatal period) {len(P):,} admissions ({P.patient_id.nunique():,} children); neonatal {len(NEO):,}.")

# %% [markdown]
# ## 2. Recorded versus DAMA-adjusted mortality

# %%
def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


from decimal import Decimal, ROUND_HALF_UP  # noqa: E402


def half_up(x, nd=0):
    return str(Decimal(repr(float(x))).quantize(Decimal('1') if nd == 0 else Decimal('0.1'), rounding=ROUND_HALF_UP))


def pci(k, n):
    """Percentage with Wilson 95% CI. AMA style: one decimal, or whole numbers (rounded half up) when the denominator is under 100."""
    if not n:
        return ''
    lo, hi = wilson(k, n)
    if n >= 100:
        return f'{100 * k / n:.1f} ({lo:.1f}-{hi:.1f})'
    return f'{half_up(100 * k / n)} ({half_up(lo)}-{half_up(hi)})'


def gap_row(name, df):
    n = len(df)
    known = df.disposition.notna() | df.death.notna()
    n_k = int(known.sum())
    deaths = int(df.death.eq(True).fillna(False).astype(bool).sum())
    crit = int(df.dama_critical_final.eq(True).fillna(False).astype(bool).sum())
    comp = deaths + crit
    lo1, hi1 = wilson(deaths, n_k)
    lo2, hi2 = wilson(comp, n_k)
    lo3, hi3 = wilson(crit, comp) if comp else (np.nan, np.nan)
    return {'Group': name, 'Admissions': n, 'With disposition': n_k, 'In-hospital deaths': deaths,
            'Recorded mortality, % (95% CI)': pci(deaths, n_k),
            'Critical DAMA': crit,
            'Death or critical DAMA, % (95% CI)': pci(comp, n_k),
            'Absolute difference, percentage points': round(100 * crit / n_k, 1) if n_k else np.nan,
            'Critical DAMA per death': round(crit / deaths, 1) if deaths else np.nan,
            'Terminal outcomes outside the hospital record, % (95% CI)': f'{100 * crit / comp:.0f} ({lo3:.0f}-{hi3:.0f})' if comp else '',
            '_rec': 100 * deaths / n_k if n_k else np.nan, '_comp': 100 * comp / n_k if n_k else np.nan}


first_adm = P.sort_values('admit_time').groupby('patient_id').head(1)
SUBGROUPS = ([('Acute encephalopathy, all', P), ('  PICU stay', P[P.picu]), ('  No PICU stay', P[~P.picu]),
              ('  Invasive ventilation', P[P.imv_ever.eq(True).fillna(False).astype(bool)]),
              ('  Vasoactive infusion', P[P.vasoactive_ever.eq(True).fillna(False).astype(bool)])]
             + [(f'  Age {g}', P[P.age_group == g]) for g in P.age_group.cat.categories if (P.age_group == g).any()]
             + [(f'  {g}', P[P.period == g]) for g in P.period.cat.categories]
             + [(f'  Residence: {g}', P[P.residence == g]) for g in ['Wuhan', 'Outside Wuhan', 'Not recorded']]
             + [(f'  {k}', P[P[k]]) for k in ETIOLOGY if P[k].sum() >= 20]
             + [('Neonatal encephalopathy', NEO), ('All index admissions (sensitivity)', A),
                ('Acute encephalopathy, first admission per child (sensitivity)', first_adm),
                ('Acute encephalopathy, including January to June 2019 (sensitivity)', P_ALL)])
T2 = pd.DataFrame([gap_row(name, df) for name, df in SUBGROUPS])
T2.drop(columns=['_rec', '_comp']).to_csv(OUT / 'table2_recorded_vs_adjusted_mortality.csv', index=False)
print(T2.drop(columns=['_rec', '_comp']).to_string(index=False))

pr, pp = T2.iloc[0], T2[T2.Group == '  PICU stay'].iloc[0]
NUM.update({'primary_deaths': int(pr['In-hospital deaths']), 'primary_critical_dama': int(pr['Critical DAMA']),
            'primary_recorded_mortality': pr['Recorded mortality, % (95% CI)'], 'primary_adjusted_mortality': pr['Death or critical DAMA, % (95% CI)'],
            'primary_ratio': float(pr['Critical DAMA per death']), 'primary_share_outside_record': pr['Terminal outcomes outside the hospital record, % (95% CI)'],
            'picu_n': int(pp['Admissions']), 'picu_deaths': int(pp['In-hospital deaths']), 'picu_critical_dama': int(pp['Critical DAMA']),
            'picu_recorded_mortality': pp['Recorded mortality, % (95% CI)'], 'picu_adjusted_mortality': pp['Death or critical DAMA, % (95% CI)']})
allr = T2[T2.Group == 'All index admissions (sensitivity)'].iloc[0]
NUM.update({'all_deaths': int(allr['In-hospital deaths']), 'all_critical_dama': int(allr['Critical DAMA']), 'all_ratio': float(allr['Critical DAMA per death']),
            'all_recorded_mortality': allr['Recorded mortality, % (95% CI)'], 'all_adjusted_mortality': allr['Death or critical DAMA, % (95% CI)']})
note(f"Primary cohort: {NUM['primary_deaths']} in-hospital deaths and {NUM['primary_critical_dama']} critical DAMA; recorded mortality "
     f"{NUM['primary_recorded_mortality']}% vs death or critical DAMA {NUM['primary_adjusted_mortality']}%; {NUM['primary_ratio']} critical DAMA per death. "
     f"PICU: {NUM['picu_recorded_mortality']}% vs {NUM['picu_adjusted_mortality']}%.")

# %% [markdown]
# ## 3. Table 1: characteristics by discharge outcome (primary cohort)

# %%
GROUPS = ['In-hospital death', 'Critical DAMA', 'Not-critical DAMA', 'Discharged alive']


def n_pct(mask, base):
    k, n = int(mask.sum()), int(len(base))
    return f'{k} ({100 * k / n:.1f})' if n else ''


def med_iqr(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return f'{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})' if len(s) else ''


def t1_rows(df_by, dama_groups=('Critical DAMA', 'Not-critical DAMA')):
    rows = [('Admissions, n', {g: f'{len(x)}' for g, x in df_by.items()})]
    rows.append(('Age, years, median (IQR)', {g: med_iqr(x.age_years) for g, x in df_by.items()}))
    for lab in P.age_group.cat.categories:
        rows.append((f'  {lab}', {g: n_pct(x.age_group == lab, x) for g, x in df_by.items()}))
    rows.append(('Male sex', {g: n_pct(x.sex == 'male', x) for g, x in df_by.items()}))
    for lab_e, txt_e in [('Han', 'Han ethnicity'), ('Other', 'Other ethnicity'), ('Not recorded', 'Ethnicity not recorded')]:
        rows.append((txt_e, {g: n_pct(x.ethnicity == lab_e, x) for g, x in df_by.items()}))
    for lab in P.period.cat.categories:
        rows.append((f'Admitted {lab}', {g: n_pct(x.period == lab, x) for g, x in df_by.items()}))
    for src, lab in [('emergency', 'Admitted through the emergency department'), ('transfer_other_hospital', 'Transferred from another hospital')]:
        rows.append((lab, {g: n_pct(x.admission_source == src, x) for g, x in df_by.items()}))
    rows.append(('Residence outside Wuhan (of recorded)', {g: (lambda y: f'{int((y == "Outside Wuhan").sum())}/{int((y != "Not recorded").sum())}')(x.residence) for g, x in df_by.items()}))
    rows.append(('PICU stay', {g: n_pct(x.picu, x) for g, x in df_by.items()}))
    rows.append(('Discharged from the PICU', {g: n_pct(x.discharge_unit.isin(['picu_main', 'picu_west']), x) for g, x in df_by.items()}))
    for c, lab in [('imv_ever', 'Invasive mechanical ventilation'), ('vasoactive_ever', 'Vasoactive infusion'), ('rrt_ever', 'Renal replacement therapy'),
                   ('plasma_exchange_ever', 'Plasma exchange'), ('cpr_ever', 'Cardiopulmonary resuscitation'), ('ecmo_ever', 'ECMO')]:
        rows.append((lab, {g: n_pct(x[c].eq(True).fillna(False).astype(bool), x) for g, x in df_by.items()}))
    rows.append(('Organ support in the 24 h before discharge', {g: n_pct(x.dama_support_last_24h.eq(True).fillna(False).astype(bool), x) if g in dama_groups else '' for g, x in df_by.items()}))
    rows.append(('Length of stay, days, median (IQR)', {g: med_iqr(x.los_days) for g, x in df_by.items()}))
    for k in ETIOLOGY:
        if P[k].sum() >= 20:
            rows.append((k, {g: n_pct(x[k], x) for g, x in df_by.items()}))
    return rows


by = {'All admissions': P, **{g: P[P.outcome_group == g] for g in GROUPS}}
T1 = pd.DataFrame([{'Characteristic': lab, **vals} for lab, vals in t1_rows(by)])
by = {g: P[P.outcome_group == g] for g in GROUPS}
# supporting counts quoted in the text
P.groupby(['outcome_group', P.disposition.fillna('not recorded')]).size().rename('admissions').to_csv(OUT / 'dispositions_primary.csv')
pd.Series({k: int(P[k].sum()) for k in ETIOLOGY}, name='admissions').to_csv(OUT / 'etiology_counts_primary.csv')
NUM['primary_age_median_iqr'] = med_iqr(P.age_years)
fa_ = P.sort_values('admit_time').groupby('patient_id').head(1)      # children: first admission in the study period
NUM['primary_children_male'] = int((fa_.sex == 'male').sum())
NUM['primary_children_age_first_admission_median_iqr'] = med_iqr(fa_.age_years)
NUM['primary_children_han'] = int((fa_.ethnicity == 'Han').sum())
NUM['primary_male_admissions'] = int((P.sex == 'male').sum())
NUM['primary_transfers_out'] = int(P.disposition.isin(['transfer_hospital', 'transfer_community']).sum())
NUM['children_critical_dama_primary'] = int(P[P.outcome_group == 'Critical DAMA'].patient_id.nunique())
NUM['children_deaths_primary'] = int(P[P.outcome_group == 'In-hospital death'].patient_id.nunique())
T1.to_csv(OUT / 'appendix_table1_characteristics_by_discharge_outcome.csv', index=False)   # appendix Table 1 (by discharge outcome and label)
print(T1.to_string(index=False))
note('Outcome groups in the primary cohort: ' + ', '.join(f'{g} {len(by[g])}' for g in GROUPS)
     + f"; disposition not recorded {int((P.outcome_group == 'Disposition not recorded').sum())}.")

# %% [markdown]
# ## 4. Where dying children die: in-hospital deaths versus critical DAMA

# %%
term = P[P.outcome_group.isin(['In-hospital death', 'Critical DAMA'])].copy()
term['home'] = term.outcome_group.eq('Critical DAMA')
cmp_rows = []
for lab, s in [('Age, years', 'age_years'), ('Length of stay, days', 'los_days')]:
    a, b = term.loc[~term.home, s].dropna(), term.loc[term.home, s].dropna()
    cmp_rows.append({'Characteristic': lab + ', median (IQR)', 'In-hospital death': med_iqr(a), 'Critical DAMA': med_iqr(b),
                     'P (Mann-Whitney)': f'{stats.mannwhitneyu(a, b).pvalue:.3f}' if len(a) and len(b) else ''})
for lab, m in [('PICU stay', term.picu), ('Discharged from the PICU', term.discharge_unit.isin(['picu_main', 'picu_west'])),
               ('Invasive mechanical ventilation', term.imv_ever.eq(True).fillna(False).astype(bool)), ('Vasoactive infusion', term.vasoactive_ever.eq(True).fillna(False).astype(bool)),
               ('Cardiopulmonary resuscitation', term.cpr_ever.eq(True).fillna(False).astype(bool)),
               ('Residence outside Wuhan (of recorded)', term.residence.eq('Outside Wuhan')),
               ('Admitted through the emergency department', term.admission_source.eq('emergency'))]:
    base = term[term.residence != 'Not recorded'] if 'Residence' in lab else term   # residence: only admissions with a recorded residence
    m = m.loc[base.index]
    t = pd.crosstab(base.home, m)
    t = t.reindex(index=[False, True], columns=[False, True], fill_value=0)
    pv = stats.fisher_exact(t.values).pvalue
    cmp_rows.append({'Characteristic': lab, 'In-hospital death': n_pct(m[~base.home], base[~base.home]),
                     'Critical DAMA': n_pct(m[base.home], base[base.home]), 'P (Mann-Whitney)': f'Fisher {pv:.3f}'})
T3 = pd.DataFrame(cmp_rows)
T3.to_csv(OUT / 'table3_death_vs_critical_dama.csv', index=False)
print(T3.to_string(index=False))
h = term[term.home].los_hours / 24
NUM['crit_los_median_days'] = round(float(h.median()), 1)
NUM['crit_los_iqr_days'] = f'{h.quantile(.25):.1f}-{h.quantile(.75):.1f}'
NUM['crit_support_24h'] = int(term[term.home].dama_support_last_24h.eq(True).fillna(False).astype(bool).sum())
NUM['crit_from_picu'] = int(term[term.home].discharge_unit.isin(['picu_main', 'picu_west']).sum())

# %% [markdown]
# ## 5. The critical label: rule versus first clinician, and the blinded second review

# %%
dama = P[P.dama.eq(True).fillna(False).astype(bool)].copy()
dall = A[A.dama.eq(True).fillna(False).astype(bool)].copy()


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if pe < 1 else np.nan
    return k, (k - 1.96 * se, k + 1.96 * se)


rows = []
for name, df in [('All DAMA admissions', dall), ('Primary cohort', dama)]:
    dec = df[df.dama_review_decision.isin(['critical', 'not_critical'])]
    k, ci = kappa(dec.dama_critical.astype(bool), dec.dama_review_decision.eq('critical'))
    rows.append({'Set': name, 'DAMA': len(df), 'Clinician decided': len(dec), 'Unsure': int(df.dama_review_decision.eq('unsure').sum()),
                 'Rule critical': int(df.dama_critical.eq(True).fillna(False).astype(bool).sum()), 'Clinician critical': int(df.dama_review_decision.eq('critical').sum()),
                 'Agreement, %': round(100 * np.mean(dec.dama_critical.astype(bool) == dec.dama_review_decision.eq('critical')), 1),
                 'Kappa (95% CI)': f'{k:.2f} ({ci[0]:.2f}-{ci[1]:.2f})', 'Final critical': int(df.dama_critical_final.eq(True).fillna(False).astype(bool).sum())})
AG = pd.DataFrame(rows)
AG.to_csv(OUT / 'agreement_rule_vs_clinician.csv', index=False)
print(AG.to_string(index=False))
print(pd.crosstab(dall.dama_critical_category, dall.dama_review_decision, margins=True))
pd.crosstab(dall.dama_critical_category, dall.dama_review_decision, margins=True).to_csv(OUT / 'rule_category_by_clinician_decision.csv')
NUM['kappa_rule_clinician_all'] = AG.iloc[0]['Kappa (95% CI)']
NUM['agreement_rule_clinician_all'] = float(AG.iloc[0]['Agreement, %'])


def find_input(name):
    for p in [HERE / name, HERE / 'output_dama_followup' / name]:
        if p.exists():
            return p
    return None


def code(x):
    return str(x).split(' ')[0].strip() if pd.notna(x) and str(x).strip() else None


rev_path = find_input('dama_second_review_INTERNAL.xlsx')
R2 = None
if rev_path:
    R2 = pd.read_excel(rev_path, sheet_name='Review', dtype=str)
    dcol = [c for c in R2.columns if c.startswith('decision')][0]
    R2['r2'] = R2[dcol].map(code)
    R2 = R2[R2.r2.isin(['critical', 'not_critical', 'unsure'])].set_index('visit_id')
if R2 is None or len(R2) == 0:
    note('Second review: not performed (no completed review file; the label is validated against death within 30 days instead).')
    NUM['second_review'] = 'not performed'
else:
    m = dall.join(R2[['r2', 'part']], how='inner')
    both = m[m.dama_review_decision.isin(['critical', 'not_critical']) & m.r2.isin(['critical', 'not_critical'])]
    k2, ci2 = kappa(both.dama_review_decision, both.r2)
    note(f'Second review: {len(m)} admissions reviewed; inter-rater kappa {k2:.2f} ({ci2[0]:.2f}-{ci2[1]:.2f}) on {len(both)} with two definite decisions.')
    m['final_v13'] = np.where(m.dama_review_decision.eq('unsure'), m.r2, np.where(m.r2.eq('unsure'), m.dama_review_decision,
                              np.where(m.dama_review_decision == m.r2, m.dama_review_decision, 'needs_consensus')))
    print(m.final_v13.value_counts())
    m[m.final_v13 == 'needs_consensus'][['dama_review_decision', 'r2']].to_csv(OUT.parent / 'second_review_needs_consensus_INTERNAL.csv')
    NUM['second_review'] = {'reviewed': int(len(m)), 'kappa': f'{k2:.2f} ({ci2[0]:.2f}-{ci2[1]:.2f})',
                            'needs_consensus': int((m.final_v13 == 'needs_consensus').sum())}

# %% [markdown]
# ## 6. Validation of the critical label: later records and telephone follow-up
# Later records = a new encounter in the hospital data after the DAMA discharge (admission, outpatient or emergency visit, laboratory or
# examination report under another visit number). Telephone results are added when the completed call list is present. Labels were
# locked before any outcome was linked.

# %%
dall_period = dall
dall = A_ALL[A_ALL.dama.eq(True).fillna(False).astype(bool)].copy()   # linkage for every exported DAMA admission; restricted to the study period below
ov = L('outpatient_visits')
labs = L('labs')
lq = L('lab_qualitative')
ex = L('exams')
ev = [admx.reset_index()[['visit_id', 'patient_id', 'admit_time']].rename(columns={'admit_time': 't'}),
      ov[['visit_id', 'patient_id', 'visit_time']].rename(columns={'visit_time': 't'}),
      labs[['visit_id', 'patient_id', 'report_time']].rename(columns={'report_time': 't'}).dropna().drop_duplicates(),
      lq[['visit_id', 'patient_id', 'report_time']].rename(columns={'report_time': 't'}).dropna().drop_duplicates()]
vis2pat = pd.concat([admx.reset_index()[['visit_id', 'patient_id']], ov[['visit_id', 'patient_id']]]).drop_duplicates('visit_id').set_index('visit_id').patient_id
e = ex[['visit_id', 'exam_time']].rename(columns={'exam_time': 't'}).dropna()
e['patient_id'] = e.visit_id.map(vis2pat)
ev.append(e.dropna(subset=['patient_id']))
EV = pd.concat(ev, ignore_index=True).astype({'visit_id': str, 'patient_id': str})
last_after = {}
for vid, r in dall.iterrows():
    E = EV[(EV.patient_id == str(r.patient_id)) & (EV.visit_id != str(vid)) & (EV.t > r.discharge_time)]
    last_after[vid] = E.t.max() if len(E) else pd.NaT
dall['last_record_after'] = pd.Series(last_after)
dall['days_last_record'] = (dall.last_record_after - dall.discharge_time).dt.total_seconds() / 86400
dall['alive_30d_records'] = dall.days_last_record >= FOLLOWUP_DAYS
dall['label'] = np.where(dall.dama_critical_final.eq(True).fillna(False).astype(bool), 'Critical DAMA', 'Not-critical DAMA')

# Telephone follow-up (optional input). The completed call list is linked by patient_id, so columns removed from a working copy
# (sex, visit ids, dates) are not needed. A reported death follows the child's last discharge from this hospital (any admission, so a
# readmission after DAMA is handled), and the reported timing category gives the window. Calls are dated by the attempt columns or,
# when those were left empty, by CALL_DATE_IF_MISSING (the date the call list was handed out: the earliest possible call date).
CALL_DATE_IF_MISSING = pd.Timestamp('2026-10-12')
DEATH_WINDOW_DAYS = {'<=24h': (0, 1), '1-7d': (1, 7), '8-30d': (8, 30), '>30d': (30, np.inf)}
fu_path = find_input('dama_phone_followup_INTERNAL.xlsx')
FU = None
if fu_path:
    C = pd.read_excel(fu_path, sheet_name='Call_list', dtype=str)
    col = {c.split(' (')[0]: c for c in C.columns}

    def get(name):
        return C[col[name]] if name in col else pd.Series(np.nan, index=C.index, dtype=object)

    C['vital'] = get('vital_status').map(code)
    if C.vital.isin(['alive', 'died']).any():
        FU = pd.DataFrame({'patient_id': C.patient_id.astype(str), 'priority': get('priority'), 'vital': C.vital,
                           'timing': get('death_timing_after_leaving').map(code), 'place': get('death_place').map(code),
                           'consent': get('verbal_consent').map(code), 'respondent': get('respondent').map(code),
                           'pcpc': get('pcpc_current').map(code), 'care_after': get('care_after_leaving').map(code),
                           'death_note': get('death_date'), 'call_note': get('call_notes')})
        times = pd.concat([pd.to_datetime(get(f'attempt{i}_datetime'), errors='coerce') for i in (1, 2, 3)], axis=1)
        FU['call_date_recorded'] = times.notna().any(axis=1).values
        FU['call_date'] = times.max(axis=1).fillna(CALL_DATE_IF_MISSING).values
        results = pd.concat([get(f'attempt{i}_result').map(code) for i in (1, 2, 3)], axis=1)
        FU['declined'] = results.eq('refused').any(axis=1).values
        FU = FU.drop_duplicates('patient_id').set_index('patient_id')
        # Vital status from the calls is used only where the guardian gave verbal consent, as the ethics committee approved. For the
        # other children (guardian declined, or not interviewed) the 30-day status after DAMA is unknown unless hospital records show survival.
        FU_RAW = FU.copy()
        FU.loc[~FU.consent.eq('yes'), ['vital', 'timing', 'place', 'pcpc', 'care_after', 'death_note', 'call_note']] = np.nan
dall['pid'] = dall.patient_id.astype(str)
if FU is None:
    note('Telephone follow-up: PENDING (no completed rows found).')
    NUM['followup'] = 'PENDING'
    dall['died_30d'] = np.where(dall.alive_30d_records, 0, np.nan)
else:
    last_discharge_any = admx.groupby(admx.patient_id.astype(str)).discharge_time.max()
    j = dall.join(FU, on='pid')
    gap = (j.pid.map(last_discharge_any) - j.discharge_time).dt.total_seconds() / 86400      # days from this DAMA to the last discharge
    lo = j.timing.map(lambda t: DEATH_WINDOW_DAYS.get(t, (0, np.inf))[0]).astype(float)
    hi = j.timing.map(lambda t: DEATH_WINDOW_DAYS.get(t, (0, np.inf))[1]).astype(float)
    died = j.vital.eq('died') & (gap + hi <= FOLLOWUP_DAYS)
    survived = (dall.alive_30d_records | (j.vital.eq('alive') & ((j.call_date - j.discharge_time).dt.total_seconds() / 86400 >= FOLLOWUP_DAYS))
                | (j.vital.eq('died') & (gap + lo > FOLLOWUP_DAYS)))
    dall['died_30d'] = np.where(died, 1, np.where(survived, 0, np.nan))
    for c in ['vital', 'timing', 'place', 'consent', 'pcpc', 'care_after', 'death_note', 'call_note', 'call_date_recorded']:
        dall['fu_' + c] = j[c]
    dall['fu_called'] = j.vital.notna()
    dall['readmitted_before_death'] = died & (gap > 0)
    n_conflict = int((died & dall.alive_30d_records).sum())
    n_earlier_encounter = int((died & dall.last_record_after.notna()).sum())     # e.g. readmission, emergency visit, late report
    kids = FU[FU.vital.isin(['alive', 'died'])]
    NUM['followup'] = {'children_on_list': int(len(FU)), 'children_with_status': int(len(kids)), 'children_died': int(kids.vital.eq('died').sum()),
                       'children_alive': int(kids.vital.eq('alive').sum()), 'conflicts_with_records': n_conflict,
                       'dama_admissions_status_known': int(dall.died_30d.notna().sum()), 'dama_admissions': int(len(dall)),
                       'call_dates_recorded': int(FU.call_date_recorded.sum()),
                       'readmitted_before_death': int(dall.readmitted_before_death.sum()),
                       'died_with_earlier_encounter_after_dama': n_earlier_encounter,
                       'children_declined': int((~FU_RAW.consent.eq('yes') & FU_RAW.declined).sum()),
                       'children_not_interviewed': int((~FU_RAW.consent.eq('yes') & ~FU_RAW.declined).sum())}
    note(f"Telephone follow-up: {len(kids)} of {len(FU)} children with vital status ({NUM['followup']['children_died']} died); "
         f"30-day status known for {NUM['followup']['dama_admissions_status_known']} of {len(dall)} DAMA admissions; "
         f"conflicts with hospital records {n_conflict}; call dates recorded for {NUM['followup']['call_dates_recorded']} children "
         f"(otherwise {CALL_DATE_IF_MISSING.date()} assumed).")

DALL_FULL = dall.copy()                                                    # every exported DAMA admission (sensitivity analysis)
dall = DALL_FULL.loc[DALL_FULL.index.isin(dall_period.index)].copy()       # study period
if FU is not None:
    rem = dall[~dall.alive_30d_records]
    st_ = FU_RAW.reindex(rem.pid.unique())
    interviewed_ = st_.consent.eq('yes') & st_.vital.isin(['alive', 'died'])
    declined_ = ~st_.consent.eq('yes') & st_.declined.eq(True)
    NUM['followup_period'] = {'dama_discharges': int(len(dall)), 'alive_30d_by_records': int(dall.alive_30d_records.sum()),
                              'remaining_discharges': int(len(rem)), 'children_remaining': int(len(st_)),
                              'children_on_call_list': int(st_.index.isin(FU_RAW.index).sum()),
                              'children_interviewed_with_consent': int(interviewed_.sum()), 'children_declined': int(declined_.sum()),
                              'children_not_interviewed': int((~interviewed_ & ~declined_).sum()),
                              'status_known': int(dall.died_30d.notna().sum()),
                              'unknown_alive_at_call_within_30d': int((dall.died_30d.isna() & dall.fu_vital.eq('alive')).sum()),
                              'conflicts_with_records': int((dall.died_30d.eq(1) & dall.alive_30d_records).sum()),
                              'readmitted_before_death': int(dall.readmitted_before_death.sum()),
                              'died_with_earlier_encounter_after_dama': int((dall.died_30d.eq(1) & dall.last_record_after.notna()).sum())}
    note(f"Follow-up, study period: {NUM['followup_period']}")
val_rows = []
for lab in ['Critical DAMA', 'Not-critical DAMA']:
    g = dall[dall.label == lab]
    known = g.died_30d.notna()
    k = int(g.died_30d.eq(1).sum())
    val_rows.append({'Label': lab, 'Admissions': len(g), 'Later hospital record': int(g.last_record_after.notna().sum()),
                     f'Alive >= {FOLLOWUP_DAYS} d by records': int(g.alive_30d_records.sum()),
                     f'{FOLLOWUP_DAYS}-day status known': int(known.sum()), f'Died within {FOLLOWUP_DAYS} d': k,
                     f'{FOLLOWUP_DAYS}-day mortality among known, % (95% CI)': (f'{100 * k / known.sum():.0f} ({wilson(k, int(known.sum()))[0]:.0f}-{wilson(k, int(known.sum()))[1]:.0f})'
                                                                              if known.sum() and NUM.get('followup') != 'PENDING' else 'PENDING')})
VAL = pd.DataFrame(val_rows)
if NUM.get('followup') == 'PENDING':
    VAL[f'{FOLLOWUP_DAYS}-day status known'] = 'PENDING'
    VAL[f'Died within {FOLLOWUP_DAYS} d'] = 'PENDING'
VAL.to_csv(OUT / 'validation_critical_label.csv', index=False)
print(VAL.to_string(index=False))
NUM['validation_records'] = VAL.drop(columns=[c for c in VAL.columns if 'mortality' in c]).to_dict('records')


if FU is not None:
    cat1 = dall.dama_critical_category.astype(str).str[0]

    def accuracy(name, pos, base):
        k = base.died_30d.notna()
        y = base.died_30d[k].eq(1)
        x = pos.loc[base.index][k]
        tp, fp, fn, tn = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())

        def f(a, b):
            return pci(a, b)
        return {'Definition': name, 'DAMA with known 30-day status': int(k.sum()), 'Died, labelled critical': tp, 'Survived, labelled critical': fp,
                'Died, labelled not critical': fn, 'Survived, labelled not critical': tn, 'Positive predictive value, % (95% CI)': f(tp, tp + fp),
                'Negative predictive value, % (95% CI)': f(tn, tn + fn), 'Sensitivity, % (95% CI)': f(tp, tp + fn), 'Specificity, % (95% CI)': f(tn, tn + fp)}

    final_pos = dall.dama_critical_final.eq(True).fillna(False).astype(bool)
    defs = [('Final label (clinician; rule where unsure)', final_pos), ('Rule alone (category A or B)', cat1.isin(['A', 'B'])),
            ('Rule category A alone', cat1.eq('A'))]
    acc_rows = []
    for setname, base in [('All index admissions', dall), ('Primary cohort', dall[dall.index.isin(P.index)])]:
        for name, pos in defs:
            acc_rows.append({'Set': setname, **accuracy(name, pos, base)})
    ACC = pd.DataFrame(acc_rows)
    ACC.to_csv(OUT / 'label_accuracy_vs_30day_death.csv', index=False)
    print(ACC.to_string(index=False))
    NUM['label_accuracy_all'] = ACC.iloc[0].to_dict()
    NUM['label_accuracy_primary'] = ACC[(ACC.Set == 'Primary cohort') & (ACC.Definition == defs[0][0])].iloc[0].to_dict()
    NUM['label_accuracy_rule_alone_all'] = ACC.iloc[1].to_dict()
    FUp = FU[FU.index.isin(dall.pid)]                                         # children with a DAMA discharge in the study period
    FUp_raw = FU_RAW[FU_RAW.index.isin(dall.pid)]
    # 30-day deaths by rule category and first-clinician decision
    CD = dall.assign(rule_category=cat1).groupby(['rule_category', 'dama_review_decision']).died_30d.agg(
        admissions='size', status_known=lambda v: int(v.notna().sum()), died_30d=lambda v: int(v.eq(1).sum()))
    CD.to_csv(OUT / 'died30_by_rule_category_and_clinician.csv')
    print(CD)
    # what the calls obtained (children with a DAMA discharge in the study period)
    kids = FUp[FUp.vital.isin(['alive', 'died'])]
    dk, ak = kids[kids.vital == 'died'], kids[kids.vital == 'alive']
    COMP = pd.DataFrame([
        ('Guardian interviewed with verbal consent (vital status used)', len(kids), len(FUp)),
        ('Guardian declined', int((~FUp_raw.consent.eq('yes') & FUp_raw.declined).sum()), len(FUp)),
        ('Not interviewed', int((~FUp_raw.consent.eq('yes') & ~FUp_raw.declined).sum()), len(FUp)),
        ('Call date recorded', int(kids.call_date_recorded.sum()), len(kids)),
        ('Deaths: time from discharge to death (category)', int(dk.timing.notna().sum()), len(dk)),
        ('Deaths: exact date of death', int(pd.to_datetime(dk.death_note, errors='coerce').notna().sum()), len(dk)),
        ('Deaths: place of death', int(dk.place.notna().sum()), len(dk)),
        ('Survivors: current PCPC', int(ak.pcpc.notna().sum()), len(ak))], columns=['Item', 'Obtained', 'Of'])
    COMP['%'] = (100 * COMP.Obtained / COMP.Of).round(1)
    COMP.to_csv(OUT / 'followup_completeness.csv', index=False)
    print(COMP.to_string(index=False))
    NUM['followup_completeness'] = COMP.set_index('Item')[['Obtained', 'Of']].to_dict('index')
    # deaths after DAMA: timing and place (children); free-text descriptors counted, text itself not exported
    TP_ = pd.crosstab(dk.timing.fillna('not recorded'), dk.place.fillna('not recorded'), margins=True)
    TP_.to_csv(OUT / 'deaths_after_dama_timing_place.csv')
    print(TP_)
    pdead = dall[dall.index.isin(P.index) & dall.died_30d.eq(1)]
    TPP = pd.crosstab(pdead.fu_timing.fillna('not recorded'), pdead.fu_place.fillna('not recorded'), margins=True)
    TPP.to_csv(OUT / 'deaths_after_dama_timing_place_primary.csv')
    print(TPP)
    NUM['primary_deaths_after_dama_timing'] = pdead.fu_timing.value_counts().to_dict()
    NUM['primary_deaths_after_dama_place'] = pdead.fu_place.value_counts().to_dict()
    NUM['primary_deaths_after_dama_children'] = int(pdead.pid.nunique())
    ptxt = pdead.fu_death_note.fillna('') + ' ' + pdead.fu_call_note.fillna('')
    NUM['primary_death_notes'] = {'brain_death_recorded': int(ptxt.str.contains('脑死亡').sum()),
                                  'family_asked_for_extubation': int(ptxt.str.contains('拔出气管插管|拔管|拔除气管').sum()),
                                  'death_certificate_from_this_hospital': int(ptxt.str.contains('死亡证').sum()),
                                  'died_before_leaving': int(ptxt.str.contains('院内已死亡').sum())}
    NUM['primary_died_before_leaving_place'] = pdead.loc[ptxt.str.contains('院内已死亡'), 'fu_place'].tolist()
    # Guardians reported the time from leaving the hospital to death. A later registration at this hospital (admission, or outpatient or
    # emergency visit) means the interval may refer to that later departure rather than to the DAMA; the death still falls within 30 days.
    reg = pd.concat([admx.reset_index()[['visit_id', 'patient_id', 'admit_time']].rename(columns={'admit_time': 't'}),
                     ov[['visit_id', 'patient_id', 'visit_time']].rename(columns={'visit_time': 't'})],
                    ignore_index=True).astype({'visit_id': str, 'patient_id': str})
    last_reg = {}
    for vid, r in pdead.iterrows():
        E = reg[(reg.patient_id == str(r.patient_id)) & (reg.visit_id != str(vid)) & (reg.t > r.discharge_time)]
        last_reg[vid] = E.t.max() if len(E) else pd.NaT
    days_reg = ((pd.Series(last_reg) - pdead.discharge_time).dt.total_seconds() / 86400).dropna()
    NUM['primary_deaths_with_later_registration'] = {'n': int(len(days_reg)), 'after_readmission': int(pdead.readmitted_before_death.sum()),
                                                     'days_min': round(float(days_reg.min()), 1) if len(days_reg) else None,
                                                     'days_max': round(float(days_reg.max()), 1) if len(days_reg) else None}
    readm = []
    for vid, r in pdead[pdead.readmitted_before_death].iterrows():   # readmissions after DAMA: days from DAMA to the next admission and its discharge
        a_ = admx.reset_index()
        a_ = a_[(a_.patient_id.astype(str) == str(r.patient_id)) & (a_.admit_time > r.discharge_time)].sort_values('admit_time')
        if len(a_):
            readm.append({'days_to_readmission': round((a_.admit_time.iloc[0] - r.discharge_time).total_seconds() / 86400, 1),
                          'days_to_its_discharge': round((a_.discharge_time.iloc[0] - r.discharge_time).total_seconds() / 86400, 1)})
    NUM['primary_deaths_with_later_registration']['readmissions'] = readm
    note(f"Deaths after DAMA (primary cohort) with a later registration at this hospital: {NUM['primary_deaths_with_later_registration']}")
    # care received, by 30-day outcome (eFigure 2)
    d30 = dall.died_30d.reindex(P.index)
    isdama = P.dama.eq(True).fillna(False).astype(bool)
    P['outcome_30d'] = np.select([P.outcome_group.eq('In-hospital death'), isdama & d30.eq(1), isdama & d30.eq(0), isdama & d30.isna()],
                                 ['Died in hospital', 'Died after DAMA', 'Survived DAMA', 'DAMA, status unknown'], default='Discharged alive')
    tf = lambda v: pd.Series(v, index=P.index).eq(True).fillna(False).astype(bool)
    FEATS = {'Cardiopulmonary resuscitation': tf(P.cpr_ever), 'Invasive ventilation': tf(P.imv_ever), 'Vasoactive infusion': tf(P.vasoactive_ever),
             'Left directly from the PICU': tf(P.discharge_unit.isin(['picu_main', 'picu_west'])),
             'Admitted through the emergency department': tf(P.admission_source.eq('emergency')),
             'Transferred from another hospital': tf(P.admission_source.eq('transfer_other_hospital'))}
    chg = []
    for g in ['Died in hospital', 'Died after DAMA', 'Survived DAMA', 'Discharged alive']:
        m = P.outcome_30d.eq(g)
        r = {'group': g, 'admissions': int(m.sum()), 'length_of_stay_median_days': round(float(P.loc[m, 'los_days'].median()), 1)}
        for k, v in FEATS.items():
            r[k + ', %'] = round(100 * v[m].mean(), 1)
            r[k + ', n'] = int(v[m].sum())
        chg.append(r)
    CHG = pd.DataFrame(chg)
    CHG.to_csv(OUT / 'characteristics_by_30day_outcome.csv', index=False)
    print(CHG.to_string(index=False))
    # Table 1 (main text): characteristics by 30-day outcome, the grouping used in the figures (blue, red, grey)
    T1_COLS = {'Died in hospital': 'Died in hospital', 'Died after DAMA': 'Died within 30 days after DAMA',
               'Survived DAMA': 'Alive 30 days after DAMA', 'Discharged alive': 'Discharged alive'}
    by30 = {'All admissions': P, **{lab: P[P.outcome_30d == g] for g, lab in T1_COLS.items()}}
    T1M = pd.DataFrame([{'Characteristic': lab, **vals} for lab, vals in
                        t1_rows(by30, dama_groups=('Died within 30 days after DAMA', 'Alive 30 days after DAMA'))])
    T1M.to_csv(OUT / 'table1_characteristics_by_30day_outcome.csv', index=False)
    print(T1M.to_string(index=False))
    NUM['dama_status_unknown_primary'] = int(P.outcome_30d.eq('DAMA, status unknown').sum())
    # comparisons quoted in the text and the Figure 3 legend: died after DAMA vs died in hospital, and vs survived DAMA
    cmp30 = []
    for other in ['Died in hospital', 'Survived DAMA']:
        a_, b_ = P.outcome_30d.eq('Died after DAMA'), P.outcome_30d.eq(other)
        feats_ = {'PICU stay': tf(P.picu), **FEATS}
        if other == 'Survived DAMA':   # assessed for DAMA only
            feats_['Organ support in the 24 h before discharge'] = tf(P.dama_support_last_24h)
        for k, v in feats_.items():
            t = [[int(v[a_].sum()), int((~v[a_]).sum())], [int(v[b_].sum()), int((~v[b_]).sum())]]
            cmp30.append({'comparison': f'Died after DAMA vs {other}', 'characteristic': k, 'P (Fisher)': stats.fisher_exact(t).pvalue})
        for lab_, col_ in [('Age', 'age_years'), ('Length of stay', 'los_days')]:
            x_, y_ = P.loc[a_, col_].dropna(), P.loc[b_, col_].dropna()
            cmp30.append({'comparison': f'Died after DAMA vs {other}', 'characteristic': lab_, 'P (Mann-Whitney)': stats.mannwhitneyu(x_, y_).pvalue})
    CMP30 = pd.DataFrame(cmp30)
    CMP30.to_csv(OUT / 'comparisons_by_30day_outcome.csv', index=False)
    print(CMP30.to_string(index=False))
    txt = dk.death_note.fillna('') + ' ' + dk.call_note.fillna('')
    NUM['death_notes'] = {'brain_death_recorded': int(txt.str.contains('脑死亡').sum()),
                          'family_asked_for_extubation': int(txt.str.contains('拔出气管插管|拔管|拔除气管').sum()),
                          'death_certificate_from_this_hospital': int(txt.str.contains('死亡证').sum()),
                          'died_before_leaving': int(txt.str.contains('院内已死亡').sum())}
    note(f"Deaths after DAMA (children): {len(dk)}; timing {dk.timing.value_counts().to_dict()}; place {dk.place.value_counts().to_dict()}; "
         f"free-text descriptors {NUM['death_notes']}.")
    # survivors' current function (children alive at the call), by the label of their last DAMA admission
    lastlab = dall.sort_values('discharge_time').groupby('pid').tail(1).set_index('pid').label
    SP = pd.crosstab(ak.index.map(lastlab), ak.pcpc.fillna('not recorded'), margins=True)
    SP.index.name = 'label of last DAMA'
    SP.to_csv(OUT / 'survivor_pcpc_by_label.csv')
    print(SP)

# %% [markdown]
# ## 7. Follow-up-corrected mortality: in hospital or within 30 days after DAMA
# Deaths in hospital plus deaths within 30 days after DAMA (telephone follow-up and hospital records), over admissions with a recorded
# disposition. DAMA admissions with unknown 30-day status count as survivors (a minimum; their number is shown), and deaths after a
# routine discharge were not ascertained. Without the follow-up, scenario estimates are shown instead.

# %%
def fu_row(name, df):
    n_k = int((df.disposition.notna() | df.death.notna()).sum())
    d_in = int(df.death.eq(True).fillna(False).astype(bool).sum())
    dd = DALL_FULL.loc[DALL_FULL.index.intersection(df.index)]
    d_out, unk = int(dd.died_30d.eq(1).sum()), int(dd.died_30d.isna().sum())
    tot = d_in + d_out
    lo1, hi1 = wilson(d_in, n_k)
    lo2, hi2 = wilson(tot, n_k)
    lo3, hi3 = wilson(d_out, tot) if tot else (np.nan, np.nan)
    return {'Group': name, 'Admissions': len(df), 'In-hospital deaths': d_in,
            'Recorded mortality, % (95% CI)': pci(d_in, n_k),
            'Deaths within 30 d after DAMA': d_out, 'DAMA with unknown 30-day status': unk,
            'Follow-up-corrected 30-day mortality, % (95% CI)': pci(tot, n_k),
            'Deaths after DAMA per in-hospital death': round(d_out / d_in, 1) if d_in else np.nan,
            'Deaths that occurred after DAMA, % (95% CI)': pci(d_out, tot),
            '_tot': tot, '_nk': n_k, '_rec': 100 * d_in / n_k if n_k else np.nan, '_fc': 100 * tot / n_k if n_k else np.nan}


if FU is not None:
    T2F = pd.DataFrame([fu_row(name, df) for name, df in SUBGROUPS])
    T2F.drop(columns=[c for c in T2F.columns if c.startswith('_')]).to_csv(OUT / 'table2b_followup_corrected_mortality.csv', index=False)
    print(T2F.drop(columns=[c for c in T2F.columns if c.startswith('_')]).to_string(index=False))
    r0 = T2F.iloc[0]
    NUM['fu_primary'] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in r0.items() if not k.startswith('_')}
    d_unk = int(r0['DAMA with unknown 30-day status'])
    NUM['fu_primary_max_if_unknown_died'] = round(100 * (r0['_tot'] + d_unk) / r0['_nk'], 1)
    note(f"Follow-up-corrected 30-day mortality, primary cohort: {r0['In-hospital deaths']} in hospital + {r0['Deaths within 30 d after DAMA']} after DAMA = "
         f"{r0['Follow-up-corrected 30-day mortality, % (95% CI)']}% vs recorded {r0['Recorded mortality, % (95% CI)']}%; "
         f"{r0['Deaths after DAMA per in-hospital death']} deaths after DAMA per in-hospital death; {r0['Deaths that occurred after DAMA, % (95% CI)']}% of deaths "
         f"occurred after DAMA; unknown 30-day status {d_unk} (at most {NUM['fu_primary_max_if_unknown_died']}% if all had died).")
    for nm in ['  PICU stay', '  Invasive ventilation', '  Vasoactive infusion', 'Neonatal encephalopathy', 'All index admissions (sensitivity)',
               'Acute encephalopathy, first admission per child (sensitivity)']:
        r = T2F[T2F.Group == nm].iloc[0]
        NUM['fu_' + nm.strip().split(',')[0].split(' (')[0].replace(' ', '_').lower()] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in r.items() if not k.startswith('_')}
else:
    sc = []
    n_k = int((P.disposition.notna() | P.death.notna()).sum())
    deaths, crit, noncrit = int(P.death.eq(True).fillna(False).astype(bool).sum()), int(P.dama_critical_final.eq(True).fillna(False).astype(bool).sum()), int((P.dama.eq(True).fillna(False).astype(bool) & ~P.dama_critical_final.eq(True).fillna(False).astype(bool)).sum())
    for p_c in [0.5, 0.7, 0.9, 1.0]:
        for p_n in [0.0, 0.05]:
            est = deaths + p_c * crit + p_n * noncrit
            sc.append({'Assumed death rate after critical DAMA': p_c, 'Assumed death rate after not-critical DAMA': p_n,
                       'Estimated mortality, %': round(100 * est / n_k, 1), 'Ratio to recorded mortality': round(est / deaths, 1) if deaths else np.nan})
    SC = pd.DataFrame(sc)
    SC.to_csv(OUT / 'scenario_true_mortality_primary.csv', index=False)
    print(SC.to_string(index=False))
    NUM['scenario_range'] = f"{SC['Estimated mortality, %'].min()}-{SC['Estimated mortality, %'].max()}"

# %% [markdown]
# ## 8. Trends by calendar year (primary cohort)

# %%
TR = P.groupby('admit_year').agg(admissions=('patient_id', 'size'), deaths=('death', 'sum'), critical_dama=('dama_critical_final', 'sum'),
                                 dama=('dama', 'sum')).astype(int)
TR['recorded_mortality_pct'] = (100 * TR.deaths / TR.admissions).round(1)
TR['death_or_critical_dama_pct'] = (100 * (TR.deaths + TR.critical_dama) / TR.admissions).round(1)
TR['dama_pct'] = (100 * TR.dama / TR.admissions).round(1)
if FU is not None:
    P['died_after_dama_30d'] = dall.died_30d.reindex(P.index).eq(1)
    TR['deaths_after_dama_30d'] = P.groupby('admit_year').died_after_dama_30d.sum().astype(int)
    TR['followup_corrected_mortality_pct'] = (100 * (TR.deaths + TR.deaths_after_dama_30d) / TR.admissions).round(1)
# Age and influenza by year of admission (eTable 5; context for the 2022-2023 peak)
TR['age_median_iqr'] = P.groupby('admit_year').age_years.apply(med_iqr)
TR['aged_6y_or_older_pct'] = (100 * P.groupby('admit_year').age_years.apply(lambda a: (a >= 6).mean())).round(1)
TR['influenza_admissions'] = P.groupby('admit_year')['Influenza'].sum().astype(int)
if FU is not None:
    P['died_any_30d'] = P.death.eq(True).fillna(False).astype(bool) | P.died_after_dama_30d
    TR['influenza_deaths'] = P[P['Influenza']].groupby('admit_year').died_any_30d.sum().reindex(TR.index, fill_value=0).astype(int)
TR.to_csv(OUT / 'trends_by_year_primary.csv')
print(TR)
# Poisson test for trend in critical DAMA per admission (year as continuous), descriptive
import statsmodels.api as sm  # noqa: E402
X = sm.add_constant(TR.index.values.astype(float) - TR.index.min())
trend_series = [(TR.critical_dama, 'critical DAMA'), (TR.deaths, 'deaths')]
if FU is not None:
    trend_series += [(TR.deaths_after_dama_30d, 'deaths after DAMA'), (TR.deaths + TR.deaths_after_dama_30d, 'all deaths')]
for y, lab in trend_series:
    fit = sm.GLM(y.values, X, family=sm.families.Poisson(), offset=np.log(TR.admissions.values)).fit()
    rr, lo, hi = np.exp(fit.params[1]), np.exp(fit.conf_int()[1][0]), np.exp(fit.conf_int()[1][1])
    note(f'Annual rate ratio, {lab}: {rr:.2f} ({lo:.2f}-{hi:.2f}), P = {fit.pvalues[1]:.3f}')
    NUM[f'trend_rr_{lab.replace(" ", "_")}'] = f'{rr:.2f} ({lo:.2f}-{hi:.2f}); P = {fit.pvalues[1]:.3f}'

# By month of admission (supplementary table; the December 2022 SARS-CoV-2 Omicron wave is quoted in the Results)
P['admit_month'] = P.admit_time.dt.to_period('M').astype(str)
agg = dict(admissions=('patient_id', 'size'), deaths=('death', 'sum'), critical_dama=('dama_critical_final', 'sum'))
if FU is not None:
    agg['deaths_after_dama_30d'] = ('died_after_dama_30d', 'sum')
TM = P.groupby('admit_month').agg(**agg).astype(int)
TM.to_csv(OUT / 'trends_by_month_primary.csv')
if '2022-12' in TM.index:
    dec = TM.loc['2022-12']
    NUM['dec2022'] = {k: int(v) for k, v in dec.items()}
    note(f"December 2022: {dec.admissions} admissions, {dec.critical_dama} critical DAMA, {dec.deaths} in-hospital deaths"
         + (f", {dec.deaths_after_dama_30d} deaths within 30 days after DAMA." if 'deaths_after_dama_30d' in dec else '.'))
print(TM.sort_values('critical_dama', ascending=False).head(6))

# End of China's zero-COVID policy (December 7, 2022): admissions, deaths, age, and coded SARS-CoV-2 infection in December 2022.
# Percentages follow AMA style: one decimal place, or whole numbers when the denominator is under 100.
covid_visits = set(dxc[dxc.diagnosis_cn.str.contains(r'新冠|新型冠状|COVID|SARS-CoV|冠状病毒', na=False, regex=True)].visit_id)
P['sars_cov_2_coded'] = P.index.isin(covid_visits)
ym_ = P.admit_time.dt.to_period('M')
dec22 = ym_.eq(pd.Period('2022-12', 'M'))
jan_nov22 = P.admit_year.eq(2022) & ~dec22


def pct_ama(k, n):
    return f'{100 * k / n:.1f}' if n >= 100 else half_up(100 * k / n)


if FU is not None:
    died_any = P.died_any_30d
    old6 = P.age_years >= 6
    RO = {'admissions_oct2022': int(ym_.eq(pd.Period('2022-10', 'M')).sum()), 'admissions_nov2022': int(ym_.eq(pd.Period('2022-11', 'M')).sum()),
          'admissions_dec2022': int(dec22.sum()), 'inhospital_deaths_dec2022': int(P.death.eq(True)[dec22].sum()),
          'deaths_after_dama_dec2022': int(P.died_after_dama_30d[dec22].sum()), 'deaths_dec2022': int(died_any[dec22].sum()),
          'deaths_dec2022_pct': pct_ama(int(died_any[dec22].sum()), int(dec22.sum())),
          'aged_6y_or_older_dec2022': int(old6[dec22].sum()), 'aged_6y_or_older_dec2022_pct': pct_ama(int(old6[dec22].sum()), int(dec22.sum())),
          'aged_6y_or_older_other_months_pct': pct_ama(int(old6[~dec22].sum()), int((~dec22).sum())),
          'deaths_aged_6y_or_older_dec2022': int((old6 & died_any)[dec22].sum()),
          'sars_cov_2_coded_dec2022': int(P.sars_cov_2_coded[dec22].sum()),
          'sars_cov_2_coded_among_deaths_dec2022': int((P.sars_cov_2_coded & died_any)[dec22].sum()),
          'admissions_jan_nov_2022': int(jan_nov22.sum()), 'deaths_jan_nov_2022': int(died_any[jan_nov22].sum()),
          'corrected_mortality_jan_nov_2022_pct': pct_ama(int(died_any[jan_nov22].sum()), int(jan_nov22.sum())),
          'influenza_admissions_2023': int(TR.loc[2023, 'influenza_admissions']), 'influenza_deaths_2023': int(TR.loc[2023, 'influenza_deaths'])}
    NUM['reopening_2022'] = RO
    note(f'Around the end of the zero-COVID policy (primary cohort): {RO}')

# %% [markdown]
# ## 9. Figure settings and eFigure 1 (study flow)
# Style: no titles, panel letters in the upper left aligned with the y-axis title, Arial (falls back to Liberation Sans), bold axis titles,
# value labels, no gridlines. Numbers follow AMA style (no thousands separator below 10 000). Files go to output_paper_b/figures/.

# %%
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

fams = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({'font.family': 'Arial' if 'Arial' in fams else ('Liberation Sans' if 'Liberation Sans' in fams else 'DejaVu Sans'),
                     'font.size': 12, 'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': False, 'legend.frameon': False, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def panel_letter(ax, letter, fig):
    fig.canvas.draw()
    ylab = ax.yaxis.label.get_window_extent(renderer=fig.canvas.get_renderer())
    x = fig.transFigure.inverted().transform((ylab.x0, 0))[0]
    y = ax.get_position().y1
    fig.text(x, y + 0.02, letter, fontsize=16, fontweight='bold', ha='left', va='bottom')


# Years, partial years and the data cutoff (used by the figures and their legends)
yrs = TR.index.values
export_end = max(admx.admit_time.max(), admx.discharge_time.max())
partial = export_end.year == int(yrs[-1]) and (export_end.month, export_end.day) < (12, 31)
first_partial = (STUDY_START.month, STUDY_START.day) != (1, 1) and int(yrs[0]) == STUDY_START.year
NUM['data_cutoff'] = f'{export_end:%B} {export_end.day}, {export_end.year}'
if first_partial:
    note(f'Figure legend: * {int(yrs[0])} covers admissions from {STUDY_START:%B} {STUDY_START.year} (partial year).')
if partial:
    note(f'Figure legend: † {int(yrs[-1])} covers admissions through {NUM["data_cutoff"]} (partial year).')


def nfmt(n):
    """AMA style: no separator below 10 000, thin space from 10 000."""
    n = int(n)
    return f'{n:,}'.replace(',', ' ') if n >= 10000 else str(n)


FIGDIR = OUT / 'figures'
FIGDIR.mkdir(exist_ok=True)

# eFigure 1: study flow. Connectors run only horizontally or vertically, and every arrow ends exactly on the edge of its box.
from matplotlib.patches import Rectangle as _Rect  # noqa: E402

fl = flow
n_epi = int(A.loc[A.cohort == 'Other non-acute', 'epileptic_encephalopathy_code'].sum())
NUM['other_nonacute_epileptic_encephalopathy'] = n_epi
fig = plt.figure(figsize=(11.5, 11))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
fig.canvas.draw()
R_ = fig.canvas.get_renderer()
PAD_X, PAD_Y, LS = 1.8, 1.4, 1.35


def text_wh(s, fs):
    t = ax.text(50, 50, s, fontsize=fs, ha='center', va='center', linespacing=LS)
    bb = t.get_window_extent(renderer=R_)
    t.remove()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return x1 - x0, y1 - y0


def box(cx, top, s, fs=11, fill='white', ls='-', w=None, left=None):
    tw, th = text_wh(s, fs)
    W = max(w or 0, tw + 2 * PAD_X)
    H = th + 2 * PAD_Y
    if left is not None:
        cx = left + W / 2
    ax.add_patch(_Rect((cx - W / 2, top - H), W, H, facecolor=fill, edgecolor='black', lw=1.0, ls=ls, zorder=2))
    ax.text(cx, top - H / 2, s, fontsize=fs, ha='center', va='center', linespacing=LS, zorder=3)
    return dict(l=cx - W / 2, r=cx + W / 2, t=top, b=top - H, cx=cx, cy=top - H / 2, w=W, h=H)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=12))


def seg(xs_, ys_):
    ax.plot(xs_, ys_, color='black', lw=1.0, solid_capstyle='butt', zorder=1)


exp_start = A_ALL.admit_time.min()
MAIN_X, SIDE_L, GAP_V = 36, 64, 12
b1 = box(MAIN_X, 98, f"Admissions with encephalopathy-related diagnoses in the export,\n{exp_start:%B %Y} to {export_end:%B %Y}: "
         f"{nfmt(NUM['n_export_admissions'])} admissions, {nfmt(NUM['n_export_patients'])} patients")
b2 = box(MAIN_X, b1['b'] - GAP_V, f"Index admissions, {STUDY_START:%B %Y} to {export_end:%B %Y}\n"
         f"{nfmt(len(A))} admissions, {nfmt(A.patient_id.nunique())} children")
b3 = box(MAIN_X, b2['b'] - GAP_V, f"Acute encephalopathy beyond the neonatal period (primary cohort)\n"
         f"{nfmt(fl.loc['Acute encephalopathy', 'admissions'])} admissions, {nfmt(fl.loc['Acute encephalopathy', 'children'])} children")
for upper, lower, s in [
        (b1, b2, f"Excluded\nAdults (maternal service): {nfmt(NUM['n_adult_excluded'])}\n"
                 f"Admitted before {STUDY_START:%B %Y}: {nfmt(NUM['n_before_study_start'])}"),
        (b2, b3, f"Neonatal encephalopathy (analyzed separately): {nfmt(fl.loc['Neonatal encephalopathy', 'admissions'])}\n"
                 f"Rehabilitation: {nfmt(fl.loc['Rehabilitation', 'admissions'])}; sequelae: {nfmt(fl.loc['Sequelae', 'admissions'])}\n"
                 f"Other nonacute: {nfmt(fl.loc['Other non-acute', 'admissions'])} (epileptic encephalopathy {nfmt(n_epi)})")]:
    ym = (upper['b'] + lower['t']) / 2
    _, th = text_wh(s, 10.5)
    sb = box(0, ym + (th + 2 * PAD_Y) / 2, s, fs=10.5, left=SIDE_L)
    arrow(MAIN_X, ym, sb['l'], ym)
arrow(MAIN_X, b1['b'], MAIN_X, b2['t'])
arrow(MAIN_X, b2['b'], MAIN_X, b3['t'])
outs = [(g, int((P.outcome_group == g).sum())) for g in GROUPS]
labels_ = {'In-hospital death': 'Died in hospital', 'Critical DAMA': 'Critical DAMA', 'Not-critical DAMA': 'Not-critical DAMA', 'Discharged alive': 'Discharged alive'}
cols = [11, 36, 61, 86]
y_bus = b3['b'] - 5
seg([MAIN_X, MAIN_X], [b3['b'], y_bus])
seg([cols[0], cols[-1]], [y_bus, y_bus])
ob = {}
for (g, n), xx in zip(outs, cols):
    ob[g] = box(xx, y_bus - 5, f'{labels_[g]}\n{nfmt(n)}', fs=11, fill='#d9d9d9' if g == 'Critical DAMA' else 'white', w=21)
    arrow(xx, y_bus, xx, ob[g]['t'])
if FU is not None:
    pd_ = dall[dall.index.isin(P.index)]
    fb, death_x = {}, [cols[0]]
    for lab, xx in [('Critical DAMA', cols[1]), ('Not-critical DAMA', cols[2])]:
        g = pd_[pd_.label == lab]
        nd, na, nu = int(g.died_30d.eq(1).sum()), int(g.died_30d.eq(0).sum()), int(g.died_30d.isna().sum())
        fb[lab] = box(xx, ob[lab]['b'] - 5, f"Within 30 days:\ndied {nfmt(nd)}, alive {nfmt(na)}" + (f", unknown {nfmt(nu)}" if nu else ''), fs=10.5, w=21)
        arrow(xx, ob[lab]['b'], xx, fb[lab]['t'])
        if nd:
            death_x.append(xx)
    r0 = T2F.iloc[0]
    s = (f"Died in hospital or within 30 days after DAMA: {nfmt(r0['_tot'])} ({100 * r0['_tot'] / r0['_nk']:.1f}%)\n"
         f"recorded as deaths {nfmt(r0['In-hospital deaths'])} ({r0['_rec']:.1f}%); recorded as discharges {nfmt(r0['Deaths within 30 d after DAMA'])}")
    top_s = min(fb[l]['b'] for l in fb) - 6
    tw, _ = text_wh(s, 11)
    W = max(tw + 2 * PAD_X, max(death_x) - min(death_x) + 12)
    cx = (min(death_x) + max(death_x)) / 2
    cx = max(cx, W / 2 + 1)
    sbx = box(cx, top_s, s, fs=11, ls='--', w=W)
    for xx in death_x:
        y0 = ob['In-hospital death']['b'] if xx == cols[0] else fb['Critical DAMA' if xx == cols[1] else 'Not-critical DAMA']['b']
        arrow(xx, y0, xx, sbx['t'])
from matplotlib.transforms import Bbox  # noqa: E402
ext_ = Bbox.union([a.get_window_extent(R_) for a in list(ax.patches) + list(ax.texts)]).transformed(fig.dpi_scale_trans.inverted())
ext_ = Bbox.from_extents(ext_.x0 - 0.1, ext_.y0 - 0.1, ext_.x1 + 0.1, ext_.y1 + 0.1)
for ext in ['png', 'pdf']:
    fig.savefig(FIGDIR / f'efigure1_study_flow.{ext}', dpi=600, bbox_inches=ext_)
plt.close(fig)
print('eFigure 1 written.')

# %% [markdown]
# ## 10. Figures 1-4 and eFigure 2
# Three colors only: blue = death recorded in hospital; red = death after DAMA (recorded as a discharge); gray = survivors and context.
# Figure 1: where the deaths occurred. Figure 2: recorded and follow-up-corrected mortality by severity and cause. Figure 3: the
# record-based label. Figure 4: mortality over time. eFigure 2: care received by status 30 days after discharge.
# Every value is read from the tables computed above. Files go to output_paper_b/figures/.

# %%
if FU is not None:
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch, Rectangle, Patch
    from matplotlib.lines import Line2D

    BLUE, RED, GRAY, LIGHT, INK, MUTED = '#256abf', '#e34948', '#c3c2b7', '#e1e0d9', '#0b0b0b', '#52514e'
    FIGV2 = FIGDIR
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.labelweight': 'bold', 'axes.edgecolor': MUTED,
                         'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 0.8})

    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(FIGV2 / f'{name}.{ext}', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def letters(fig, items, same_x=(), same_y=()):
        # panel letters at the upper left of each panel, aligned with its leftmost element (the y-axis title when there is one);
        # same_x / same_y: groups of letters that share a column or a row
        fig.canvas.draw()
        r_, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
        pos = {}
        for ax_, t in items:
            bb = ax_.get_tightbbox(r_)
            pos[t] = [inv.transform((bb.x0, 0))[0], inv.transform((0, bb.y1))[1]]
        for grp in same_x:
            m_ = min(pos[t][0] for t in grp)
            for t in grp:
                pos[t][0] = m_
        for grp in same_y:
            m_ = max(pos[t][1] for t in grp)
            for t in grp:
                pos[t][1] = m_
        for t, (x_, y_) in pos.items():
            fig.text(x_, y_ + 0.01, t, fontsize=16, fontweight='bold', ha='left', va='bottom')

    r0 = T2F.iloc[0]
    n_in, n_after = int(r0['In-hospital deaths']), int(r0['Deaths within 30 d after DAMA'])
    pd_ = dall[dall.index.isin(P.index)]

    # ---------------- Story 1: where the deaths were ----------------
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.45], height_ratios=[1.35, 0.75], hspace=0.42, wspace=0.12)
    # A: every death as one square
    ax = fig.add_subplot(gs[0, 0]); ax.set_aspect('equal'); ax.axis('off')
    rows_ = 10
    def block(x0, n, color, ncol):
        for i in range(n):
            c, r = divmod(i, rows_)
            ax.add_patch(Rectangle((x0 + c, r), 0.86, 0.86, facecolor=color, edgecolor='none'))
    ncol_in = int(np.ceil(n_in / rows_)); ncol_after = int(np.ceil(n_after / rows_))
    block(0, n_in, BLUE, ncol_in)
    block(ncol_in + 1.6, n_after, RED, ncol_after)
    ax.text(ncol_in / 2 - 0.07, -0.9, f'{n_in}', ha='center', va='top', fontsize=22, fontweight='bold', color=BLUE)
    ax.text(ncol_in / 2 - 0.07, -3.0, 'in hospital', ha='center', va='top', fontsize=10, color=INK)
    xa = ncol_in + 1.6 + ncol_after / 2 - 0.07
    ax.text(xa, -0.9, f'{n_after}', ha='center', va='top', fontsize=22, fontweight='bold', color=RED)
    ax.text(xa, -3.0, 'after DAMA', ha='center', va='top', fontsize=10, color=INK)
    ax.set_xlim(-0.3, ncol_in + 1.6 + ncol_after + 0.2); ax.set_ylim(-4.6, rows_ + 0.3)
    axA = ax

    # B: recorded disposition -> status at 30 days (admissions ending in death or DAMA)
    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    crit = pd_[pd_.label == 'Critical DAMA']; ncrit = pd_[pd_.label == 'Not-critical DAMA']
    flows = [('death', 'died_hosp', n_in, BLUE),
             ('crit', 'died_after', int(crit.died_30d.eq(1).sum()), RED),
             ('crit', 'unknown', int(crit.died_30d.isna().sum()), LIGHT),
             ('crit', 'alive', int(crit.died_30d.eq(0).sum()), GRAY),
             ('ncrit', 'died_after', int(ncrit.died_30d.eq(1).sum()), RED),
             ('ncrit', 'unknown', int(ncrit.died_30d.isna().sum()), LIGHT),
             ('ncrit', 'alive', int(ncrit.died_30d.eq(0).sum()), GRAY)]
    flows = [f for f in flows if f[2] > 0]
    left = [('death', f'Died in hospital  {n_in}', BLUE), ('crit', f'Critical DAMA  {len(crit)}', MUTED), ('ncrit', f'Not-critical DAMA  {len(ncrit)}', GRAY)]
    rtot = {k: sum(f[2] for f in flows if f[1] == k) for k in ['died_hosp', 'died_after', 'unknown', 'alive']}
    right = [('died_hosp', f'{rtot["died_hosp"]}  died in hospital', BLUE), ('died_after', f'{rtot["died_after"]}  died within 30 days', RED),
             ('unknown', f'{rtot["unknown"]}  unknown', LIGHT), ('alive', f'{rtot["alive"]}  alive at 30 days', GRAY)]
    right = [r for r in right if rtot[r[0]] > 0]
    total = sum(f[2] for f in flows); gap = 0.035; unit = (1 - gap * 3) / total
    def stack(nodes, sizes):
        pos, y = {}, 1.0
        for key, _, _ in nodes:
            h = sizes[key] * unit
            pos[key] = [y - h, y]
            y -= h + gap
        return pos
    lsize = {k: sum(f[2] for f in flows if f[0] == k) for k, _, _ in left}
    lpos, rpos = stack(left, lsize), stack(right, rtot)
    xl, xr, w = 0.46, 1.0, 0.035
    lcur = {k: v[1] for k, v in lpos.items()}; rcur = {k: v[1] for k, v in rpos.items()}
    for src, dst, n, color in flows:
        h = n * unit
        ya1, yb1 = lcur[src], rcur[dst]; ya0, yb0 = ya1 - h, yb1 - h
        lcur[src] -= h; rcur[dst] -= h
        xm = (xl + w + xr - w) / 2
        verts = [(xl + w, ya1), (xm, ya1), (xm, yb1), (xr - w, yb1), (xr - w, yb0), (xm, yb0), (xm, ya0), (xl + w, ya0), (xl + w, ya1)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor='none', alpha=0.55 if color != LIGHT else 0.9))
    for key, lab, color in left:
        y0, y1 = lpos[key]
        ax.add_patch(Rectangle((xl, y0), w, y1 - y0, facecolor=color, edgecolor='none'))
        ax.text(xl - 0.02, (y0 + y1) / 2, lab, ha='right', va='center', fontsize=10, color=INK)
    for key, lab, color in right:
        y0, y1 = rpos[key]
        ax.add_patch(Rectangle((xr - w, y0), w, max(y1 - y0, 0.004), facecolor=color, edgecolor='none' if color != LIGHT else MUTED, lw=0.6))
        ax.text(xr + 0.02, (y0 + y1) / 2, lab, ha='left', va='center', fontsize=10, color=INK)
    ax.text(xl + w / 2, 1.05, 'At discharge', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=INK)
    ax.text(xr - w / 2, 1.05, 'At 30 days', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=INK)
    ax.set_xlim(0.0, 1.5); ax.set_ylim(-0.02, 1.1)
    axB = ax

    # C: place and time of the deaths after DAMA
    ax = fig.add_subplot(gs[1, :])
    places = [('en_route', 'On the way home'), ('home', 'At home'), ('other_hospital', 'In another hospital')]
    tp = TPP.drop(index='All', errors='ignore')
    yv = np.arange(len(places))[::-1]
    for (pk, plab), y in zip(places, yv):
        a = int(tp.loc['<=24h', pk]) if ('<=24h' in tp.index and pk in tp.columns) else 0
        b = int(tp.loc['1-7d', pk]) if ('1-7d' in tp.index and pk in tp.columns) else 0
        if a:
            ax.barh(y, a, height=0.62, color=RED, edgecolor='white', linewidth=2)
            ax.text(a / 2, y, f'{a}', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
        if b:
            ax.barh(y, b, left=a, height=0.62, facecolor='white', edgecolor=RED, hatch='////', linewidth=1.2)
            ax.text(a + b + 0.8, y, f'{b}', ha='left', va='center', color=INK, fontsize=11)
    ax.set_yticks(yv); ax.set_yticklabels([p[1] for p in places])
    ax.set_xlabel('Deaths after DAMA, n'); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.legend(handles=[Patch(facecolor=RED, label='Within 24 h of leaving'), Patch(facecolor='white', edgecolor=RED, hatch='////', label='1 to 7 days after leaving')],
              loc='lower right', frameon=False, fontsize=10)
    axC = ax
    letters(fig, [(axA, 'A'), (axB, 'B'), (axC, 'C')], same_x=[('A', 'C')], same_y=[('A', 'B')])
    save(fig, 'figure1_deaths_in_hospital_and_after_dama')

    # ---------------- Story 2: the gap by severity and by cause ----------------
    T2Fi = T2F.assign(name=T2F.Group.str.strip()).set_index('name')
    sev = [('No PICU stay', 'No PICU stay'), ('Acute encephalopathy, all', 'All admissions'), ('PICU stay', 'PICU stay'),
           ('Invasive ventilation', 'Invasive ventilation'), ('Cardiac arrest', 'Cardiac arrest'), ('Vasoactive infusion', 'Vasoactive infusion')]
    causes = [k for k in ETIOLOGY if k in T2Fi.index and T2Fi.loc[k, '_tot'] > 0]
    short = {'Acute encephalopathy syndrome or encephalopathy, unspecified': 'Acute encephalopathy syndrome or unspecified',
             'Toxic (infection-associated) encephalopathy': 'Toxic encephalopathy',
             'Metabolic, hypoglycemic or hypertensive encephalopathy, PRES': 'Metabolic, hypoglycemic, or hypertensive; PRES',
             'Viral or autoimmune encephalitis': 'Viral or autoimmune encephalitis', 'Hypoxic-ischemic encephalopathy': 'Hypoxic-ischemic encephalopathy'}
    cause_rows = sorted([(c, short.get(c, c)) for c in causes if c != 'Cardiac arrest'], key=lambda t: T2Fi.loc[t[0], '_fc'])
    sev_rows = sorted(sev, key=lambda t: T2Fi.loc[t[0], '_fc'])
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 8.4), gridspec_kw={'height_ratios': [len(sev_rows), len(cause_rows)], 'hspace': 0.28}, sharex=True)
    for ax, rows in zip(axes, [sev_rows, cause_rows]):
        for i, (key, lab) in enumerate(rows):
            rec, fc = T2Fi.loc[key, '_rec'], T2Fi.loc[key, '_fc']
            ax.plot([rec, fc], [i, i], color=GRAY, lw=3, solid_capstyle='round', zorder=1)
            ax.scatter([rec], [i], s=70, color=BLUE, zorder=3, edgecolor='white', linewidth=1.5)
            ax.scatter([fc], [i], s=70, color=RED, zorder=3, edgecolor='white', linewidth=1.5)
            nk_ = int(T2Fi.loc[key, '_nk'])
            lab_ = (lambda v: f'{v:.1f}') if nk_ >= 100 else half_up      # AMA: whole numbers when the group is under 100
            ax.text(rec - 1.2, i, lab_(rec), ha='right', va='center', fontsize=9.5, color=BLUE)
            ax.text(fc + 1.2, i, lab_(fc), ha='left', va='center', fontsize=9.5, color=RED)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{lab} ({nfmt(T2Fi.loc[key, 'Admissions'])})" for key, lab in rows])
        ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
        ax.set_ylim(-0.7, len(rows) - 0.3)
    axes[1].set_xlabel('Mortality, %')
    axes[1].set_xlim(-6, max(T2Fi.loc[[k for k, _ in sev_rows], '_fc']) + 8)
    for ax in axes:
        ax.spines['bottom'].set_bounds(0, 50)
    axes[0].tick_params(axis='x', labelbottom=False)
    axes[0].spines['bottom'].set_visible(False); axes[0].tick_params(axis='x', length=0)
    axes[0].legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, markersize=8, label='Recorded in-hospital mortality'),
                            Line2D([], [], marker='o', ls='', color=RED, markersize=8, label='Died in hospital or within 30 days after DAMA')],
                   loc='lower right', frameon=False, fontsize=10)
    letters(fig, [(axes[0], 'A'), (axes[1], 'B')], same_x=[('A', 'B')])
    save(fig, 'figure2_mortality_by_severity_and_cause')

    # ---------------- Story 3: two kinds of dying ----------------
    CHi = CHG.set_index('group')
    groups = [('Died in hospital', BLUE, 'o'), ('Died after DAMA', RED, 'o'), ('Survived DAMA', GRAY, 'o')]   # survived = alive 30 days after DAMA
    feats = ['Cardiopulmonary resuscitation', 'Invasive ventilation', 'Vasoactive infusion', 'Left directly from the PICU',
             'Admitted through the emergency department', 'Transferred from another hospital']
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={'width_ratios': [1.45, 1], 'wspace': 0.32})
    ax = axes[0]
    yv = np.arange(len(feats))[::-1]
    offs = {'Died in hospital': 0.22, 'Died after DAMA': 0.0, 'Survived DAMA': -0.22}
    for g, color, mk in groups:
        vals = [CHi.loc[g, f + ', %'] for f in feats]
        ax.scatter(vals, yv + offs[g], s=60, color=color, edgecolor='white', linewidth=1.2, zorder=3,
                   label=f"{g} ({int(CHi.loc[g, 'admissions'])})")
        n_g = int(CHi.loc[g, 'admissions'])
        for f_, v, y in zip(feats, vals, yv + offs[g]):
            k_ = int(CHi.loc[g, f_ + ', n'])
            txt = f'{100 * k_ / n_g:.1f}' if n_g >= 100 else half_up(100 * k_ / n_g)     # AMA: whole numbers when the group is under 100
            ax.text(v + 2, y, txt, ha='left', va='center', fontsize=8.5, color=INK)
    ax.set_yticks(yv); ax.set_yticklabels(feats); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 108); ax.spines['bottom'].set_bounds(0, 100); ax.set_xlabel('Admissions, %')
    ax.legend(loc='lower right', frameon=False, fontsize=9.5, bbox_to_anchor=(1.0, -0.02))
    ax = axes[1]
    rng = np.random.default_rng(20261013)
    for i, (g, color, mk) in enumerate(groups):
        v = P.loc[P.outcome_30d.eq(g), 'los_days'].dropna().clip(lower=0.05)
        y = len(groups) - 1 - i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(v, y, s=14, color=color, alpha=0.75, edgecolor='none')
        med = float(v.median())
        ax.plot([med, med], [len(groups) - 1 - i - 0.3, len(groups) - 1 - i + 0.3], color=INK, lw=2)
        ax.text(med, len(groups) - 1 - i + 0.36, f'{med:.1f}', ha='center', va='bottom', fontsize=9.5, color=INK)
    ax.set_xscale('log'); ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100]); ax.set_xticklabels(['0.1', '0.3', '1', '3', '10', '30', '100'])
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g for g, _, _ in groups][::-1]); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False); ax.set_xlabel('Length of stay, days'); ax.set_ylim(-0.6, len(groups) - 0.2)
    ax.legend(handles=[Line2D([], [], color=INK, lw=2, label='Median')], loc='lower right', frameon=False, fontsize=9.5)
    letters(fig, [(axes[0], 'A'), (axes[1], 'B')], same_y=[('A', 'B')])
    save(fig, 'efigure2_care_by_30day_status')

    # ---------------- Story 4: hidden deaths over time ----------------
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.6), gridspec_kw={'height_ratios': [1, 1.05], 'hspace': 0.42})
    ax = axes[0]
    yrs = TR.index.values.astype(int)
    ax.fill_between(yrs, TR.recorded_mortality_pct, TR.followup_corrected_mortality_pct, color=RED, alpha=0.14, lw=0, label='Deaths after DAMA')
    ax.plot(yrs, TR.followup_corrected_mortality_pct, color=RED, lw=2.2, marker='o', ms=6, label='Died in hospital or within 30 days after DAMA')
    ax.plot(yrs, TR.recorded_mortality_pct, color=BLUE, lw=2.2, marker='o', ms=6, label='Recorded in-hospital mortality')
    for x_, y_ in zip(yrs, TR.followup_corrected_mortality_pct):
        ax.text(x_, y_ + 0.8, f'{y_:.1f}', ha='center', va='bottom', fontsize=9.5, color=RED)
    for x_, y_ in zip(yrs, TR.recorded_mortality_pct):
        ax.text(x_, y_ - 0.8, f'{y_:.1f}', ha='center', va='top', fontsize=9.5, color=BLUE)
    ax.set_xticks(yrs)
    ax.set_xticklabels([str(v) + ('*' if (v in GAP_YEARS or (first_partial and v == yrs[0])) else '') + ('†' if partial and v == yrs[-1] else '') for v in yrs])
    ax.set_ylim(-2.5, max(TR.followup_corrected_mortality_pct) * 1.35); ax.set_ylabel('Mortality, %'); ax.set_xlabel('Year of admission')
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)
    ax = axes[1]
    months = pd.period_range(max(pd.Period(f'{yrs[0]}-01', 'M'), STUDY_START.to_period('M')), TM.index.max(), freq='M')
    TMf = TM.reindex([str(m) for m in months], fill_value=0)
    xs_ = np.arange(len(months))
    gap_idx = [i for i, m in enumerate(months) if m in set(GAP)]
    if gap_idx:
        ax.axvspan(gap_idx[0] - 0.5, gap_idx[-1] + 0.5, facecolor='white', edgecolor=GRAY, hatch='////', lw=0, label='Records deleted (COVID-19)')
    ax.bar(xs_, TMf.deaths, width=0.8, color=BLUE, label='Died in hospital')
    ax.bar(xs_, TMf.deaths_after_dama_30d, bottom=TMf.deaths, width=0.8, color=RED, label='Died within 30 days after DAMA')
    top = TMf.deaths + TMf.deaths_after_dama_30d
    k = int(np.argmax(top.values))
    ax.text(xs_[k], top.values[k] + 0.3, f'{months[k].strftime("%b %Y")}: {int(TMf.deaths_after_dama_30d.iloc[k])} after DAMA, {int(TMf.deaths.iloc[k])} in hospital',
            ha='center', va='bottom', fontsize=9.5, color=INK)
    jan = [i for i, m in enumerate(months) if m.month == 1]
    if months[0].month != 1:
        jan = [0] + jan
    ax.set_xticks(jan); ax.set_xticklabels([months[i].strftime('%b %Y') if (i == 0 and months[0].month != 1) else str(months[i].year) for i in jan])
    ax.set_xlim(-1, len(months)); ax.set_ylim(0, top.max() + 2.2)
    ax.set_ylabel('Deaths, n'); ax.set_xlabel('Month of admission')
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)
    letters(fig, [(axes[0], 'A'), (axes[1], 'B')], same_x=[('A', 'B')])
    save(fig, 'figure4_mortality_over_time')

    # ---------------- Story 5: the record-based label ----------------
    CDi = CD.reset_index()
    order_dec = {'critical': 0, 'unsure': 1, 'not_critical': 2}
    CDi['o'] = CDi.dama_review_decision.map(order_dec)
    CDi = CDi.sort_values(['rule_category', 'o']).reset_index(drop=True)
    dec_lab = {'critical': 'critical', 'unsure': 'unsure', 'not_critical': 'not critical'}
    final_crit = lambda r: (r.dama_review_decision == 'critical') or (r.dama_review_decision == 'unsure' and r.rule_category in ('A', 'B'))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={'width_ratios': [1.1, 1], 'wspace': 0.62})
    ax = axes[0]
    yv = np.arange(len(CDi))[::-1]
    for y, (_, r) in zip(yv, CDi.iterrows()):
        kn, dd = int(r.status_known), int(r.died_30d)
        pdie = 100 * dd / kn if kn else 0
        ax.barh(y, pdie, height=0.66, color=RED, edgecolor='white', linewidth=2)
        ax.barh(y, 100 - pdie, left=pdie, height=0.66, color=GRAY, edgecolor='white', linewidth=2)
        ax.text(102, y, f'{dd}/{kn}', ha='left', va='center', fontsize=10, color=INK)
    ax.set_yticks(yv)
    ax.set_yticklabels([f"{r.rule_category}, {dec_lab[r.dama_review_decision]}" for _, r in CDi.iterrows()])
    for tl, (_, r) in zip(ax.get_yticklabels(), CDi.iterrows()):
        if final_crit(r):
            tl.set_fontweight('bold')
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 100); ax.set_xlabel('DAMA with known status, %'); ax.set_ylabel('Rule category, clinician decision')
    ax.legend(handles=[Patch(facecolor=RED, label='Died within 30 days'), Patch(facecolor=GRAY, label='Alive at 30 days')],
              loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10)
    ax = axes[1]
    acc_all = ACC[ACC.Set == 'All index admissions'].reset_index(drop=True)
    metrics = [('Sensitivity, % (95% CI)', 'Sensitivity'), ('Specificity, % (95% CI)', 'Specificity'),
               ('Positive predictive value, % (95% CI)', 'Positive predictive value'), ('Negative predictive value, % (95% CI)', 'Negative predictive value')]
    styles = [(RED, 'o', 'Final label (clinician; rule where unsure)', RED), (MUTED, 's', 'Rule alone (category A or B)', MUTED),
              (MUTED, 'D', 'Rule category A alone', 'white')]
    def parse(cell):
        m_ = re.match(r'([\d.]+) \(([\d.]+)-([\d.]+)\)', str(cell))
        return tuple(float(g) for g in m_.groups())
    yv = np.arange(len(metrics))[::-1] * 1.0
    for j, (color, mk, lab, face) in enumerate(styles):
        row = acc_all.iloc[j]
        for y, (col, _) in zip(yv, metrics):
            est, lo_, hi_ = parse(row[col])
            yy = y + 0.22 - 0.22 * j
            ax.plot([lo_, hi_], [yy, yy], color=color, lw=1.6, zorder=2)
            ax.scatter([est], [yy], marker=mk, s=46, facecolor=face, zorder=3, edgecolor=color, linewidth=1.3)
            ax.text(hi_ + 0.4, yy, f'{est:.1f}', ha='left', va='center', fontsize=8.5, color=INK)
    ax.set_yticks(yv); ax.set_yticklabels([m[1] for m in metrics]); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(75, 103); ax.spines['bottom'].set_bounds(75, 100); ax.set_xlabel('% (95% CI)')
    ax.legend(handles=[Line2D([], [], color=c, marker=m, markerfacecolor=f, lw=1.6, label=l) for c, m, l, f in styles], loc='upper center', bbox_to_anchor=(0.45, -0.13),
              ncol=1, frameon=False, fontsize=9.5)
    letters(fig, [(axes[0], 'A'), (axes[1], 'B')], same_y=[('A', 'B')])
    save(fig, 'figure3_record_based_label')
    print('Figures written to', FIGV2)

# %% [markdown]
# ## 11. Save the key numbers

# %%
NUM['outcome_groups_primary'] = {g: int((P.outcome_group == g).sum()) for g in GROUPS}
(OUT / 'numbers.json').write_text(json.dumps(NUM, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
(OUT / 'summary.md').write_text('# Paper B analysis summary\n\n' + '\n'.join(f'- {s}' for s in SUMMARY) + '\n', encoding='utf-8')
print(json.dumps(NUM, indent=1, ensure_ascii=False, default=str))
