# Lubitz data mart
The Lubitz data mart is a download of structured data from RPDR and is available through Partners. Shaan Khurshid has created an excellent [overview of the mart](https://drive.google.com/drive/u/0/folders/1HftRHY29_-7AZXUdpunfFOR0NKOCyDVr) that explains how typical uses cases and how to get access. 

## Broadie Access
* First make sure you have an active partners' account
* Follow all the instruction in Shaan's document (getting access to the mart, installing citrix)
* you do *not* need to login to the VPN before starting
* go to workspace.partners.org and login with your partners account
* navigate to sqlserver 2016 (apps), click, download, open something that open
* login into server phsqlrpdr258, via windows authentication with your user, my user is PARTNERS\pb555
    
# Mart structure
* The main phenotype table is (lubitz_mart) datamart_schema.sk_ehr_cohort
    * creating sk_ehr_cohort:
        * TODO: Shaan, can you list/check-in the scripts that are needed to create sk_ehr_cohort?
        * curate_pcp_cohort_09159.sql
* adding phenotypes to sk_ehr_cohort
    * TODO: add all phenotype creation to files, or create subfolders with info
    * new phenotypes can be appended using append_phenotype.sql        
* raw data measurements
    * TODO: add all measurements to this file (or create subfolder and separate out)
    * tables listing MRN + clinical measurements are in datamart_schema.*sk*\_ehr\_cohort\_*ecgs*
* data summaries for plots
    * create_disease_summaries.sql does counting/grouping/incidence/prevalence by age for plotting
    * [google sheets plotting](https://docs.google.com/spreadsheets/d/16cvant-79Kfng0fCdAJWxV5SeKAw0qDWZeZkxIYnnPg/)

    
