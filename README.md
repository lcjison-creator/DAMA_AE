# DAMA_AE

Analysis code for the study **Deaths After Discharge Against Medical Advice Among Children With Acute Encephalopathy in China** (manuscript in preparation).

The study measured how many children with acute encephalopathy died after discharge against medical advice (DAMA) from a tertiary children's hospital in Wuhan, China (April 2020 to October 2026), by linking hospital records with telephone follow-up, and validated a record-based definition of critical DAMA against death within 30 days.

## What the notebook does

`paper_b_dama_analysis.ipynb` reproduces every number, table, and figure in the article from a frozen, versioned research data set (`AE_freeze_v1.2`) and the completed telephone follow-up list:

1. Cohorts and the study period
2. Recorded vs follow-up-corrected mortality, by subgroup, with sensitivity analyses
3. Characteristics by discharge outcome and by 30-day status (Table 1, eTable 1)
4. The record-based label of critical DAMA (rule and clinician decision) and its accuracy against death within 30 days
5. Trends by year and month, including the post hoc analysis around December 2022
6. Study flow (eFigure 1), Figures 1-4, and eFigure 2

## Data

The data are not included in this repository because they are patient-level health records. Deidentified data are available from the corresponding author on reasonable request, after approval of the request by the Medical Ethics Committee of Wuhan Children's Hospital.

## How to run

1. Put `paper_b_dama_analysis.ipynb` in a folder that contains the frozen data set folder `AE_freeze_v1.2/` (which includes its loader, `code/load_frozen.py`) and the completed follow-up workbook `dama_phone_followup_INTERNAL.xlsx`.
2. Install the packages in `requirements.txt` (Python 3.13).
3. Run all cells. Outputs are written to `output_paper_b/`: tables as CSV, figures as PNG (600 dpi) and PDF, `numbers.json`, and `summary.md`. No row-level identifiers are written to the output folder.

`paper_b_dama_analysis.py` holds the same code as a percent-format script.

## Notes

- Vital status from the telephone follow-up is used only where the guardian gave verbal consent; for other children, 30-day status after DAMA is unknown unless hospital records show survival.
- `CALL_DATE_IF_MISSING` is the date the call list was handed out; it dates calls whose times were not recorded.
- The code was written with the assistance of Claude (Anthropic); the authors reviewed and ran all of it.

## License

MIT (see `LICENSE`).
