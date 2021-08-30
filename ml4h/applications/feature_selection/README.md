# ML4HEN-COX

## Description

This repo contains code to perform feature selection in very large datasets --- in both number of samples and number of covariates --- using survival data.

## Citation 

### Selection of 51 predictors from 13,782 candidate multimodal features using machine learning improves coronary artery disease prediction

Saaket Agrawal, BS*, Marcus D. R. Klarqvist, PhD,  MSc*, Connor Emdin, DPhil, MD, Aniruddh P. Patel, MD, Manish D. Paranjpe, BA, Patrick T. Ellinor, MD, PhD, Anthony Philippakis, MD, PhD, Kenney Ng, PhD, Puneet Batra, PhD, Amit V. Khera, MD, MSc

#### Abstract

#### Background

Most current approaches to cardiovascular disease prediction use a relatively small number of predictors validated using Cox regression models. Here, we study the extent to which machine learning might: (a) enable principled selection of predictors from a large multimodal set of candidate variables; and (b) improve prediction of incident coronary artery disease (CAD) events compared to clinically used algorithms.

#### Methods

We studied 173,274 participants of the UK Biobank free of baseline cardiovascular disease,  of whom 5,140 (3.0%) developed CAD over a median follow-up of 11 years. Each participant was described by 13,782 candidate variables derived from a nurse interview, laboratory tests, and the electronic medical record. An elastic net-based Cox model (ML4HEN-COX)  was constructed in a development cohort of 80% of the participants and tested in the remaining 20% of participants.

#### Findings

In addition to most traditional risk factors, the model selected a polygenic score, waist and hip circumference, a marker of socioeconomic deprivation, and several hematologic indices. A more than 30-fold gradient in risk was noted across quintiles of the ML4HEN-COX distribution, with 10-year risk estimates ranging from 0.25% to 7.8%. ML4HEN-COX displayed improved discrimination of incident CAD (C-statistic = 0.796, 95% CI: 0.784-0.809) compared to the Framingham Risk Score, Pooled Cohort Equations, and QRISK3 (C-statistic range = 0.754-0.761). 

#### Conclusions

An elastic net-based Cox model (ML4HEN-COX) selected 51 predictors from a multimodal set of 13,782 candidate variables, demonstrating enhanced prediction of incident CAD versus the Framingham Risk Score, Pooled Cohort Equations, and QRISK3. This approach is readily generalizable to a broad range of large, complex datasets – likely to be increasingly available in the coming years – and disease endpoints.

