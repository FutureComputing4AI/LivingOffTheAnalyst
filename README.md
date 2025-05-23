# Living off the Analyst: Harvesting Features from Yara Rules for Malware Detection
https://arxiv.org/abs/2411.18516

## Notes:
* Environments used for portions of the pipeline are different due to versioning issues and varying package managers.
* As a result, the code is a bit disjoint but still straightforward.
* For the ML portion of the codebase we use a Conda environment and for the Yara portion we use a venv.
    * Yara: `pip install -r requirementsYara.txt`
    * ML: `conda create --name ML --file requirementsConda.txt`
* Dependencies in the ML environment are VERY finnicky.
    * As a result, the [Ember dependency itself](https://github.com/elastic/ember) is broken for us.
    * We have begun to load its `.dat` files without the usage of `Ember` but this still does not fully work.
    * It is up to you to get this set up properly.

## How to Run:
The pipeline is defined in the section below. How to run this is code is given here:
1. Create and activate pip `venv`.
2. `python3 main.py extract <path-to-yara-rules-to-harvest-from>`
    * Dumps harvested rules to `data/yar/rules.yar`.
3. Run the rules over malware with Yara.
4. `python3 main.py matrix --yara-outfile <your-yara-output-file-path>`
    * Dumps feature matrices to the following paths:
        * `data/npz/all_matrix.npz`: Unsplit matrix.
        * `data/npz/train_matrix.npz`
        * `data/npz/train_labels.npz`
        * `data/npz/test_matrix.npz`
        * `data/npz/test_labels.npz`
5. Create and activate conda `env`.
6. `python3 lasso_grids.py`
    * Use lasso-penalized logistic regression over a grid of lambda to select varying numbers of YARA rules as features.
    * We propose methods for selection which condition on the existing EMBER features to find YARA rules which add *additional* value.
7. `python3 xgb_models.py`
    * Classification accuracy on EMBER 2018 across varying numbers of used YARA rules. The used YARA rules are defined by the grids computed in `lasso_grids.py`.
8. `python3 compare_models.py`
    * Assessing how other models (LGBM and RF) compare with XGBoost.
9. All of the `plot_*.py` files.

## The Pipeline:
1. `lib.extract.strings_as_yara_rules`
    - Extract strings from existing Yara rules and create a new ruleset.
2. Your dataset
    - Run generated Yara rules over malware samples.
3. `lib.matrix.to_sparse -> lib.matrix.subset_split`
4. `ml/lasso_grids.py`
    - Generate lasso grids for your Yara output and the Ember data.
    - Required for the subsequent steps.
5. `ml/xgb_models.py`
6. `ml/compare_models.py`
7. `ml/plot_*.py`
    - Figure generation for each point.

## Cite us!
```
@misc{gupta2024livinganalystharvestingfeatures,
      title={Living off the Analyst: Harvesting Features from Yara Rules for Malware Detection}, 
      author={Siddhant Gupta and Fred Lu and Andrew Barlow and Edward Raff and Francis Ferraro and Cynthia Matuszek and Charles Nicholas and James Holt},
      year={2024},
      eprint={2411.18516},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2411.18516}, 
}
```

