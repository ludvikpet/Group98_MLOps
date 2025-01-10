# Group98_MLOps

Adam Ledou s204216  <br>
Ludvik Petersen s_____  <br>
Martin Maximilian Ægidius s194119  <br>
Troels Ludwig s204227  <br>

Overall goal:
The main goal of the project is to classify the intent of sent customer queries, which potentially could improve efficiency for managing e-mail inboxes. The main goal of the project is to develop MLOps skill-sets.

# Frameworks
We intend to use the [HuggingFace Transformers](https://github.com/huggingface/transformers) framework.
We will use a base-model version as a baseline, and see if we can improve performance in a transfer-learning context.
We plan to use hydra for configurations, and wandb for monitoring experiments. We decided against liveshare2 in order to gain more familiarity with git.

# Dataset
We will use the [banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset consisting of 13083 english-languaged customer service queries with 77 intent labels.

# Model
We plan to use TinyBERT, which is a distilled version of BERT with approximately 15M parameters. We plan to use the CLS-token for the classification head (if they actually made it with a CLS-token). Alternatively, we will simply build the head on top of the encoded input.



# Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
