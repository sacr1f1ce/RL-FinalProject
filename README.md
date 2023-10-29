RL-FinalProject
==============================

Optimizing grid bot trading strategy with RL. <br>
For now to run in you can simple do (right it uses the best parameters we were able to find by default)
```
grid_bot = GridBot()
grid_bot.trade(df_path)
```
It will contain all the relevant info inside.

To reproduce mentioned results with RL application, use venv (we used `python3.9`) defined by `requirements.txt` and run all cels in `src/models/grid_bot_rl/test.ipynb`

Project Organization
------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed           <- Market data with features.
    │   └── raw                 <- Market data.
    │
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                  <- Regular grid bot
    │
    ├── notebooks               <- Jupyter notebooks. (Visualization & metrics)
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                <- Scripts to download market data
    │   │   └── make_dataset.py
    │   │
    │   ├── features            <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models              <- Scripts to train models and then use trained models to make
    │   │   │                      predictions
    │   │   └── grid_bot_rl     <- Grid bot with adaptive orders.
    │   │       │
    │   │       └── test.ipynb  <- Notebook with experiment setup
    │   │
    │   └── visualization       <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io


Results for Regular bot 1 day
------------
![https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/total_1.png](https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/total_1.png)


Results for Regular bot 3 days
------------
![https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/total_3.png](https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/total_3.png)



Results for Regular bot 7 days
------------
![https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/total_7.png](https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/total_7.png)

Results of RL approach
------------
![https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/RL_3.png](https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/RL_3.png)
![https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/RL_1.png](https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/RL_1.png)
![https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/RL_act.png](https://github.com/sacr1f1ce/RL-FinalProject/blob/main/reports/figures/RL_act.png)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
