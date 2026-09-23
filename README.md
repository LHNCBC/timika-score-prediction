# Timika Score Prediction

## Overview

This notebook implements an automated deep learning pipeline to estimate the **Timika score**—a clinically established semi-quantitative measure of tuberculosis (TB) disease severity on chest radiographs (CXR), scored on a 0–140 scale. This work is part of a pilot study assessing machine learning (ML) approaches to predict TB treatment outcomes using the IeDEA-Southern Africa (SA) cohort.

## Background / Study Context

Post-treatment morbidity and disengagement from TB care remain major challenges in high HIV-prevalence settings. This pilot study evaluated tabular, image-only, and multimodal ML approaches for predicting treatment continuity and post-treatment radiographic sequelae using data from the **IeDEA-Southern Africa consortium**, a multi-country cohort combining structured clinical/radiological data with linked CXRs from **1,438 adults (≥15 years)** with pulmonary TB enrolled between **October 2022 and September 2024** across five countries: **Malawi, Mozambique, South Africa, Zambia, and Zimbabwe**.

The Timika score pipeline was specifically developed to automate radiographic severity scoring, which is normally assessed manually, and to test whether models trained on public datasets generalize to a real-world, multi-country African cohort.

## Pipeline Architecture

The Timika score is decomposed into three clinically interpretable sub-tasks executed in sequence:

1. **Lung Detection / Segmentation** — a U-Net-based model identifies and delineates lung fields to define total lung area.
2. **Lesion Detection** — an ensemble of nnU-Net and YOLOv8 identifies TB-related abnormalities within the segmented lungs, from which the percentage of affected lung area is calculated.
3. **Cavitation Detection** — a DenseNet121 model classifies the presence/absence of lung cavitation.

The final predicted Timika score combines the estimated **percentage of affected lung area** with a **fixed increment of +40 points for cavitation**, consistent with the clinical scoring definition. Implementation details build on prior work by Kantipudi et al.:

> Kantipudi K, et al. Automated Pulmonary Tuberculosis Severity Assessment on Chest X-rays. J Imaging Inform Med 37(5), 2173–2185, 2024.

## Usage

`timika_pipeline.ipynb` runs the three-model TB chest-X-ray pipeline end-to-end and computes the **Timika score** per image:

1. **Lung segmentation** (heart excluded) → binary lung mask
2. **Lesion segmentation** (nnU-Net + YOLOv8m ensemble) → binary lesion mask
3. **Cavity classification** → cavity probability + 0/1 label

Then:

```
PAL    = (lung_mask ∩ lesion_mask).sum() * 100 / lung_mask.sum()   # percent of lung affected
Timika = PAL + 40 * cavity_label                                   # cavity adds 40
```

**Self-contained & portable.**
- `timika_lib/` - model code
- `weights/` - all model weights
- `sample.csv` + `data/` - inputs
- Run the install cell once (installs deps into `./pylibs`), then run the cells top to bottom.

## Data Used

**Training data** (public datasets, chosen for larger sample size to support model development):
- Shenzhen chest X-ray dataset (NLM)
- Montgomery County (MC) chest X-ray dataset (NLM) — combined with Shenzhen: 802 images
- NIAID TB Portals dataset: 13,763 images

**Independent test/evaluation data:**
- IeDEA-Southern Africa cohort: 1,583 CXRs, used as an external, out-of-domain test set (not seen during training) to assess real-world generalization rather than in-sample performance. This subset was drawn primarily from Malawi, South Africa, and Zambia, as image quality (low-exposure images, digitized film scans) limited usability from other sites.

## Results Summary

- **Mean Absolute Error (MAE): 27.21** (SD 21.86) on the 0–140 Timika scale when evaluated against reference scores on the independent IeDEA-SA test set — indicating **moderate agreement**.
- **Signed error analysis**: mean bias of **−11.36** (SD 33.00), indicating a slight tendency toward **underestimating disease severity**, with both under- and overestimation observed across cases.
- Despite domain shift from differences in projection, exposure, positioning, and image quality across sites, the pipeline demonstrated **modest but clear agreement** with reference scores, supporting the feasibility of radiology-focused deep learning within IeDEA-SA.

## Interpretation Notes

- This is a **pilot/feasibility** result, not a validated clinical tool. Performance was likely affected by domain shift between curated public training data and heterogeneous, routine-care IeDEA-SA images.
- Findings suggest that improved **cross-site metadata capture** (e.g., projection type, equipment type) and potential **local calibration/adaptation** would likely improve future performance.

## Data Access

IeDEA-SA raw tabular and image data require a project concept form submitted to and approved by the IeDEA-SA consortium (per its Principles of Collaboration and IRB requirements): <https://www.iedea-sa.org/contact-us/>

Public training datasets:
- TB Portals: <https://tbportals.niaid.nih.gov>
- Shenzhen dataset: [NLM LHNCBC data portal](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html)
- Montgomery County dataset: [NLM LHNCBC data portal](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html)

## Funding & Acknowledgments

This work was supported by:
- The **Office of AIDS Research (OAR), National Institutes of Health (NIH)** — primary funder of this work.
- The **Lister Hill National Center for Biomedical Communications (LHNCBC), National Library of Medicine (NLM), NIH**.
- Federal funds from the **National Institute of Allergy and Infectious Diseases (NIAID), NIH**, Department of Health and Human Services, under BCBB Support Services Contract HHSN316201300006W/75N93022F00001 to Guidehouse Digital.
- Computation was performed on the **Biowulf Linux cluster** at NIH, Bethesda, MD (<http://biowulf.nih.gov>).

Contributions of NIH authors are considered Works of the United States Government. Findings and conclusions are those of the authors and do not necessarily reflect the views of the NIH or U.S. Department of Health and Human Services. The authors declare no conflicts of interest.

## Ethics

All participating IeDEA-SA sites obtained approval from local institutional review boards/ethics committees, and all study participants provided written informed consent.

## Citation

> Cid V, Bui V, Kantipudi K, Yaniv Z, Jaeger S. Assessing the Potential of Machine Learning for Predicting Tuberculosis Treatment Outcome in a Multi-Country Cohort in Southern Africa. 54th Annual Applied Imagery Pattern Recognition Workshop, Washington, DC. Springer LNCS, 2026.

## Point of Contact

Vy Bui, Stefan Jaeger  
National Library of Medicine  
National Institutes of Health  
8600 Rockville Pike  
Bethesda, MD 20894  
Email: {vy.bui, stefan.jaeger}@nih.gov
