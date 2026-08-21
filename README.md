# \# SDSS Stellar Classification and Data Analysis

# 

# \## Overview

# 

# This project investigates whether photometric and spectroscopic properties can be used to distinguish between stars, galaxies and quasars (QSOs) in data from the Sloan Digital Sky Survey (SDSS).

# 

# The analysis combines exploratory data analysis, astronomical colour indices, redshift analysis, SQL and machine learning to investigate the differences between the three object classes.

# 

# \## Dataset

# 

# The dataset contains \*\*100,000 observations\*\* of astronomical objects classified as:

# 

# \* Galaxy

# \* QSO (quasar)

# \* Star

# 

# The dataset contains photometric measurements in the `u`, `g`, `r`, `i` and `z` bands, as well as positional information, redshift and object classification.

# 

# \## Data Cleaning

# 

# The dataset uses `-9999` as a sentinel value for missing measurements. These values were identified and replaced with missing values before analysis.

# 

# Photometric colour indices were calculated from the measured magnitudes:

# 

# \* `u-g`

# \* `g-r`

# \* `r-i`

# \* `i-z`

# 

# \## Exploratory Analysis

# 

# The initial analysis investigated:

# 

# \* Class distribution

# \* Photometric magnitude distributions

# \* Colour-index distributions

# \* Colour-colour relationships

# \* Redshift distributions

# \* Numerical features by object class

# 

# \### Class distribution

# 

# The dataset is dominated by galaxies, with smaller numbers of stars and QSOs.

# 

# !\[Class distribution](figures/class\_distribution.png)

# 

# \### Photometric colours

# 

# The colour indices show clear differences between the three classes. Galaxies generally have larger colour indices, while QSOs tend to have smaller values.

# 

# !\[g-r distribution](figures/g\_r\_distribution\_normalised.png)

# 

# A colour-colour diagram was also used to investigate the degree of separation and overlap between the classes.

# 

# !\[Colour-colour diagram](figures/colour\_colour\_diagram.png)

# 

# \### Redshift

# 

# The redshift distributions show a strong distinction between the classes.

# 

# Stars are concentrated around zero redshift, while galaxies have intermediate redshifts and QSOs have substantially larger redshifts.

# 

# !\[Redshift distribution](figures/redshift\_distribution\_zoomed.png)

# 

# The SQL analysis also found that approximately \*\*79.6% of QSOs\*\* have a redshift greater than 1, compared with approximately \*\*1.5% of galaxies\*\* and essentially none of the stars.

# 

# \## SQL Analysis

# 

# The dataset was also loaded into a SQLite database to investigate the data using SQL.

# 

# Queries were used to examine:

# 

# \* Object counts and class proportions

# \* Mean redshift by class

# \* High-redshift objects

# \* Mean photometric colour indices

# 

# This provided an additional way of analysing the dataset alongside the Python-based analysis.

# 

# \## Machine Learning

# 

# A \*\*Random Forest classifier\*\* was trained to distinguish between galaxies, QSOs and stars.

# 

# The model used:

# 

# \* `u`

# \* `g`

# \* `r`

# \* `i`

# \* `z`

# \* `u-g`

# \* `g-r`

# \* `r-i`

# \* `i-z`

# 

# The data was split into training and testing sets using a stratified 80/20 split.

# 

# \### Classification performance

# 

# The model achieved:

# 

# | Metric        |     Result |

# | ------------- | ---------: |

# | Test accuracy | \*\*88.35%\*\* |

# | Macro F1      |  \*\*0.850\*\* |

# | Galaxy recall |    \*\*95%\*\* |

# | QSO recall    |    \*\*81%\*\* |

# | Star recall   |    \*\*76%\*\* |

# 

# !\[Confusion matrix](figures/confusion\_matrix.png)

# 

# The classifier performs particularly well for galaxies, while stars and QSOs show greater overlap.

# 

# \### Feature importance

# 

# The most important feature was the `r-i` colour index.

# 

# | Feature | Importance |

# | ------- | ---------: |

# | `r-i`   |  \*\*32.7%\*\* |

# | `u-g`   |  \*\*14.3%\*\* |

# | `g-r`   |  \*\*13.0%\*\* |

# | `i-z`   |  \*\*11.7%\*\* |

# 

# !\[Feature importance](figures/feature\_importance.png)

# 

# The results suggest that photometric colour contains substantial information for distinguishing between astronomical object classes.

# 

# \### Magnitudes vs colours

# 

# Three feature sets were compared:

# 

# | Features             |   Accuracy |  Macro F1 |

# | -------------------- | ---------: | --------: |

# | Magnitudes only      |     86.28% |     0.824 |

# | Colours only         |     86.47% |     0.827 |

# | Magnitudes + colours | \*\*88.35%\*\* | \*\*0.850\*\* |

# 

# Colour indices alone performed slightly better than the raw magnitudes, while combining both produced the best performance. This suggests that the two feature types provide complementary information.

# 

# \### Cross-validation

# 

# Five-fold stratified cross-validation was used to test the stability of the classifier.

# 

# The model achieved:

# 

# \*\*88.62% ± 0.27% accuracy\*\*

# 

# across the five folds, indicating that the classification performance is relatively stable across different subsets of the dataset.

# 

# \## Key Findings

# 

# The analysis found several clear differences between the three astronomical classes:

# 

# 1\. \*\*Redshift provides strong separation between the classes.\*\* Stars are concentrated around zero redshift, while QSOs generally have much larger redshifts.

# 2\. \*\*Photometric colours are useful for classification.\*\* The colour indices show clear class-dependent distributions.

# 3\. \*\*`r-i` was the most important feature\*\* in the Random Forest model, accounting for approximately 32.7% of the model's feature importance.

# 4\. \*\*Combining magnitudes and colour indices improves classification performance\*\*, achieving 88.35% accuracy compared with 86.28% using magnitudes alone and 86.47% using colours alone.

# 5\. \*\*Model performance was stable\*\*, with five-fold cross-validation producing 88.62% ± 0.27% accuracy.

# 

# \## Project Structure

# 

# ```text

# stellar-data-analysis/

# │

# ├── raw/

# │   ├── README.md

# │   └── data\_dictionary.md

# │

# ├── notebooks/

# │   └── 01\_data\_exploration.py

# │

# ├── sql/

# │   └── 01\_basic\_analysis.sql

# │

# ├── src/

# │   ├── create\_database.py

# │   ├── run\_sql.py

# │   ├── train\_classifier.py

# │   ├── compare\_features.py

# │   ├── confusion\_matrix.py

# │   └── cross\_validation.py

# │

# ├── figures/

# │   ├── class\_distribution.png

# │   ├── u\_band\_distribution.png

# │   ├── g\_r\_distribution\_normalised.png

# │   ├── colour\_colour\_diagram.png

# │   ├── redshift\_by\_class.png

# │   ├── redshift\_distribution\_zoomed.png

# │   ├── confusion\_matrix.png

# │   └── feature\_importance.png

# │

# └── README.md

# ```

# 

# \## Tools

# 

# \* Python

# \* Pandas

# \* Matplotlib

# \* Scikit-learn

# \* SQLite / SQL

# \* Git / GitHub

# 

# \## Future Work

# 

# Possible extensions include testing additional classification methods, investigating the physical interpretation of the most informative colour indices, and exploring whether incorporating additional SDSS features can improve classification performance.



