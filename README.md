# Distinct Electrophysiological Signatures of Intentional and Unintentional Mind-Wandering Revealed by Low-Frequency EEG Markers

This repository contains the code and data associated with the paper:

*Martel, A., Bruno, N., Robertson, I. H., Dockree, P. M., Sitt, J. D., & Valero-Cabré, A. (2023). Distinct Electrophysiological Signatures of Intentional and Unintentional Mind-Wandering Revealed by Low-Frequency EEG Markers. In bioRxiv (p. 2023.03.21.533634). https://doi.org/10.1101/2023.03.21.533634*

## Abstract
Mind-wandering is typically characterized by the common experience wherein attention veers off into thoughts unrelated to the task at hand. Recent research highlights the intentionality dimension of mind-wandering as a key predictor of adverse functional outcomes with intentional and unintentional task-unrelated thought (TUT) differentially linked to neural, behavioral, clinical, and functional correlates. We here aimed to elucidate the electrophysiological underpinnings of intentional and unintentional TUT by systematically examining the individual and collective discriminative power of a large set of EEG markers to distinguish between attentional states. Univariate and multivariate analyses were conducted on 54 predefined markers belonging to four conceptual families: ERP, spectral, information theory and connectivity measures, extracted from scalp EEG recordings prior to multidimensional reports of ongoing thought from participants performing a sustained attention task. We report here that on-task, intentional and unintentional TUT exhibit distinct electrophysiological signatures in the low frequency range. More specifically, increased features of the theta frequency range were found to be most discriminative between on-task and off-task states, while features within the alpha band were characteristic of intentional TUT when compared to unintentional TUT. This result is theoretically well aligned with contemporary accounts describing alpha activity as an index of internally oriented attention and a potential mechanism to shield internal processes from sensory input. Our study verifies the validity of the intentionality dimension of mind-wandering and represents a step forward towards real-time detection and mitigation of maladaptive mind-wandering. ### Competing Interest Statement The authors have declared no competing interest.

---
## Repository Structure

- `data/` (contains the dataset used in the study)
- `src/` (contains the source code used to perform the analysis and generate results)
- `results/` (contains the generated results, e.g., figures, tables, and other output files)

## Installation and Requirements
To run the code included in this repository, you will need to install the following Python packages:

- `mne`
- `nice` 
For `nice` installation refer to https://github.com/nice-tools/nice

---
## How to Run the Analysis

To reproduce the results of this study, please follow the steps listed below. Each step corresponds to a Jupyter notebook which should be run in the order specified.

### Step 1: Preprocessing
Navigate to the `Preprocessing` directory and run the `preprocessing_manual.ipynb` notebook. This notebook contains the code for preprocessing the raw EEG data. This involves steps such as filtering, epoching, artifact rejection, and channel interpolation.


### Step 2: Compute Markers
Next, navigate to the `Compute_markers` directory and run the `compute_markers.ipynb` notebook. This notebook computes the EEG markers of interest for the study, such as power spectral density, time-locked contrasts, and other connectivity measures.


### Step 3: Univariate Analysis
Now, navigate to the `Univariate_analysis` directory and run the `univariate_analysis.ipynb` notebook. This notebook includes the code to conduct univariate analysis of the computed EEG markers to identify significant differences across conditions.

For running comparison with the other mind state reports (e.g. 'about-task', 'distracted') run the `univariate_all_against_all.ipynb` notebook.

### Step 4: Multivariate Analysis
Finally, navigate to the `MVPA` directory and run the `multivariate_analysis.ipynb` notebook. This notebook includes the code to perform multivariate pattern analysis (MVPA) on the computed EEG markers, allowing us to evaluate the collective discriminative power of these markers.

Make sure you have installed all the necessary packages and dependencies before running the notebooks. If you encounter any issues, please create an issue in this repository.


The notebook for making the figures that compare between different analyses is `.src/plots.ipynb`. 

For the manuscript all the figures have been improved using Adobe Illustrator. 

---
## Data

This repository main datasets for the analysis:

### Raw Data
The raw EEG data used in the study is not included in this repository due to its size. Please refer here[LINK TO RAW DATA] to access the raw data.

Please note that all data used in this study is anonymized, with no personally identifiable information included. If you have any concerns about the data, please contact the authors.

### Preprocessed Data
The preprocessed data, which has been cleaned and organized for analysis, is also not included in this repository also because of size issues. Please refer here[LINK TO PREPROCESSED DATA] to access the preprocessed data.
This data is ready for further analysis using the provided notebooks. The preprocessing steps applied to the raw data are outlined in the `preprocessing_manual.ipynb` notebook under the `Preprocessing` directory. 

### Computed markers dataset
The `all_markers.csv` dataset, having undergone rigorous data cleaning and organization for analysis, can be found in this repository under the `Data/` directory. This dataset, with its calculated markers, is prepared for subsequent analysis using the notebooks provided. The detailed step-by-step marker computation methods, applied to the preprocessed data, are clearly demonstrated in the `compute_markers.ipynb` notebook located in the `Compute_markers` directory.

In this dataset, each row corresponds to an individual epoch (trial) from our database, and the columns provide various forms of identifiers and markers:

**Identifiers**
- `participant`: This identifier uniquely marks each participant. Not the same values as the real subject identifiers.
- `probe`: This variable signify the type of probe by which the participant made its report during the EEG recording session.  `SC`: Self-Caught and `PC`: Probe-Caught
- `mind`: This may indicate the participant's mental state reported during the EEG recording. `dMW`: deliberate Mind-wandering; `sMW`: spontaneous Mind-wanderin  
- `segment`: This represents the specific segment or phase of the study during which the recording took place.
- `stimuli`: This column indicates whether the presented stimulus was `go` or `no-go`.
- `correct`: This indicates if the response given by the participant to the go or no go stimuli was `correct` or `incorrect`.
- `prev_trial`: This varialble indicates the distance position of the trial to the probe (i.e. `1` is the last trial before the probe, `5` is five trials before the probe)  
- `preproc`: Indicates if the preprocessing for that involved subtracting the ERP or not. Not use.
- `epoch_type`: It should not to be regarded, it was not consider.

**Markers**
- `wSMI_1, wSMI_2, wSMI_4, wSMI_8`: These columns encapsulate the weighted symbolic mutual information (SMI) at various scales. SMI is a measure of the interdependence between two random variables at different scales or levels of detail.
- `p_e_1, p_e_2, p_e_4, p_e_8`: These columns represent permutation entropy at different scales. Permutation entropy serves as a measure of the complexity or randomness of a time series.
- `k`: This column denotes Kolgomorov Complexity.
- `se, msf, sef90, sef95`: These are spectral features extrapolated from the EEG signals. 'se' is an acronym for spectral entropy, 'msf' stands for mean spectral frequency, and 'sef90' and 'sef95' indicate spectral edge frequency at 90% and 95%, respectively.
- `b, b_n, g, g_n, t, t_n, d, d_n, a_n, a`: These columns quantify the power of different brainwave bands: Beta (b), Gamma (g), Theta (t), Delta (d), and Alpha (a). The '_n' suffix denotes the normalized power of these respective brainwaves.
- `CNV, P1, P3a, P3b`: These represent event-related potential (ERP) components. The Contingent Negative Variation (CNV) is a long-duration negative ERP. P1, P3a, and P3b are other ERP components, distinguished by their positive deflection and latency measured in milliseconds (ms) post-stimulus.

For an in-depth understanding of each marker, we recommend referring to the accompanying manuscript.


---
## Authors

- List all authors and their contact information, if applicable.

## Citation

If you use the code or data in this repository, please cite our paper:

```
Martel, A., Bruno, N., Robertson, I. H., Dockree, P. M., Sitt, J. D., & Valero-Cabré, A. (2023). Distinct Electrophysiological Signatures of Intentional and Unintentional Mind-Wandering Revealed by Low-Frequency EEG Markers. In bioRxiv (p. 2023.03.21.533634). https://doi.org/10.1101/2023.03.21.533634
```

## License



## Acknowledgments

