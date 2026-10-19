# DAMA_AE

Analysis code for the study **Deaths After Discharge Against Medical Advice Among Children With Acute Encephalopathy in China** (manuscript in preparation).

The study measured how many children with acute encephalopathy died after discharge against medical advice (DAMA) from a tertiary children's hospital in Wuhan, China (April 2020 to October 2026), by linking hospital records with telephone follow-up, and validated a record-based definition of critical DAMA against death within 30 days. A post hoc check examined what a public pediatric intensive care database, the Paediatric Intensive Care (PIC) database, records about these deaths.

## What the notebooks do

`paper_b_dama_analysis.ipynb` reproduces every Wuhan number, table, and figure in the article from a frozen, versioned research data set (`AE_freeze_v1.2`) and the completed telephone follow-up list:

1. Cohorts and the study period
2. Recorded vs follow-up-corrected mortality, by subgroup (including sex), with sensitivity analyses
3. Characteristics by discharge outcome and by 30-day status (Table 1, eTable 1)
4. The record-based label of critical DAMA (rule and clinician decision) and its accuracy against death within 30 days, together with 2 definitions that use no text: organ support in the 24 hours before discharge, and the PIC-compatible definition (eTable 4)
5. Trends by year and month, including the post hoc analysis around December 2022
6. Study flow (eFigure 1), Figures 1-4, and eFigure 2

`paper_b_pic_check.ipynb` runs the post hoc check in PIC version 1.1.0 (eTable 6). It checks whether PIC records a discharge disposition, deaths after discharge, or ventilation, and counts admissions that ended in discharge alive with an order for dopamine, dobutamine, norepinephrine, milrinone, or vasopressin on the day of discharge or the day before, by age, discharge unit, and chapter of the first-listed diagnosis.

## Data

The Wuhan data are not included in this repository because they are patient-level health records. Deidentified data are available from the corresponding author on reasonable request, after approval of the request by the Medical Ethics Committee of Wuhan Children's Hospital.

PIC is not included either. It is available from PhysioNet to credentialed users under the PhysioNet Credentialed Health Data Use Agreement (https://doi.org/10.13026/32x9-wv38), and `paper_b_pic_check.ipynb` must be run by a credentialed user on their own computer.

## How to run

**Wuhan analysis**

1. Put `paper_b_dama_analysis.ipynb` in a folder that contains the frozen data set folder `AE_freeze_v1.2/` (which includes its loader, `code/load_frozen.py`) and the completed follow-up workbook `dama_phone_followup_INTERNAL.xlsx`.
2. Install the packages in `requirements.txt` (Python 3.13).
3. Run all cells. Outputs are written to `output_paper_b/`: tables as CSV (including `p_values_exact.csv`, every reported P value at full precision), figures as PNG (600 dpi) and PDF, `numbers.json`, and `summary.md`. No row-level identifiers are written to the output folder.

**PIC check**

1. Put `paper_b_pic_check.ipynb` in the folder that holds the PIC files (.csv or .csv.gz; subfolders are searched).
2. Run all cells (pandas is the only package needed; the article's counts were produced with Python 3.12 and pandas 2.3).
3. Outputs are written to `output_pic_paper_b/`: `pic_summary.md` and `etable_pic.csv` hold aggregate counts, with counts of 1 to 4 admissions shown as "<5"; `pic_numbers.json` holds the unmasked aggregate counts and stays on the credentialed user's computer.

The `.py` files hold the same code as percent-format scripts.

## Notes

- Vital status from the telephone follow-up is used only where the guardian gave verbal consent; for other children, 30-day status after DAMA is unknown unless hospital records show survival.
- `CALL_DATE_IF_MISSING` is the date the call list was handed out; it dates calls whose times were not recorded.
- PIC: only aggregate counts (`pic_summary.md`) left the credentialed author's computer; no row-level PIC data were shared with anyone, including the artificial intelligence tool used to write the code.
- `paper_b_pic_check` version 1.1 differs from version 1, which produced the counts in the article, only in the shared summary: rates are not shown when a count is below 5, and the discharged-alive column is left out. The counts are the same.
- The code was written with the assistance of Claude (Anthropic); the authors reviewed and ran all of it.

## License

MIT (see `LICENSE`).
