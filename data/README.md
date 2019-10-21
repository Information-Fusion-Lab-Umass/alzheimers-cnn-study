# Metadata Files

This folder contains various metadata files that are downloaded from the ADNI website.

## MRILIST.csv

Contains the **IMAGEUID** column that can be matched to the MRI images. The **Visit** column can be matched to the 
**VISNAME** column in the *VISITS.csv* file.

## VISITS.csv

This file maps the **VISNAME** column, containing visit names in natural languagem to **VISCODE**, which can be mapped 
**VISCODE** in the *ADNIMERGE_relabeled.csv* file.

## ADNIMERGE_relabled.csv

This file is the *ADNIMERGE.csv* file downloaded from ADNI website, where the **DX** rows with empty values are filled 
based on previous or subsequent visits.
