# Hazardous Ingredients in Cosmetics Project


This project contains data from the California Department of Public Health where they maintain a database of all personal beauty products that contain ingredients that could potentially be hazardous. All products are self-reported by the manufacturers or companies, and reporting is required if the company:

   * Has annual aggregate sales of cosmetic products of one million dollars or more, and
   * Has sold cosmetic products in California on or after January 1, 2007.
   
Additional data from Kaggle was added to ensure that products that do not contain a hazardous ingredient are included in the project.

## Goals:
   * Determine features that could determine whether a product will contain a hazardous ingredient.
   * Build a model that can accurately predict if a product contains a hazardous ingredient.

## Steps to Reproduce:
   * Download the csv from CDPH at https://catalog.data.gov/dataset/chemicals-in-cosmetics-7d6ab.
   *  Download the csv from Kaggle at https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets.
   * Put the data in the same file as the cloned repo.
   * Run the final_notebook.
   
## Acquire
* Data acquired from California Department of Public Health (CDPH) and Kaggle.com
* Two separate dataframes are acquired before cleaning
    * DF1, the data from Kaggle, has 11 columns and 1472 rows before cleaning
    * DF2, the data from CDPH, has 22 columns and 114635 rows before cleaning
* Each row is a single product
* Each column contains information about the product

## Prepare
DF1 (Kaggle):
* Duplicates removed
* Ingredients extracted from single cell
* Target column, has_hazard_ingredient, created when comparing ingredients to hazard ingredient list.
* Unused columns dropped
* Product types renamed to match CHDP data.
    * Only Skincare products and Sunscreen products
* Columns renamed to match CDHP data.

DF2 (CDPH):
* Duplicates removed
* Target column, has_hazard_ingredient, created
* Unused columns dropped
* Columns renamed

Final dataframe:
* Combined Kaggle and CDPH dataframes
* All other types of products other than Skincare and Sunscreen removed:
    * These two types were the only ones that contained products with non-hazardous ingredients
* Split the data into train, validate, split in a 50, 30, 20 split, stratified on has_hazard_ingredient.
* 'Brand' and 'Type' features encoded:
    * LeaveOneOut Encoding used for dimensionality issues with a sigma = 0.5 to avoid overfitting.
        * Target excluded in test data
* Final dataframe has 4 columns and 7193 rows.


# Recommendations
* Continue developing the model.
    * Data too specified for accurate work in real world application


## Next Steps
Find more data.
   * Lack of products that don't contain hazard ingredients skewed the data
  
Features:

   * keep ingredients that aren't listed as hazardous
   * country the product is manufactured
    
Encoding:

   * Explore further encoding methods that can handle dimensionality (thousands of brands) without overfitting model on a binary target value (bayesian methods?)