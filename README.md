# Group98_MLOps

*Adam Ledou, s204216*  <br>
*Ludvik Petersen, s194613*  <br>
*Martin Maximilian Ægidius, s194119*  <br>
*Troels Ludwig, s204227*  <br>

**Overall goal**:
The main goal of the project is to develop our MLOps skill-sets. Our intention is to implement the whole MLOps pipeline, with special emphasis on the operations part of the pipeline.

As a basis for our project, we've chosen to consider banking text data, with the purpose of classifying the intent of sent customer queries, which potentially could improve efficiency for managing e-mail inboxes.

# Frameworks
We intend to use the [HuggingFace Transformers](https://github.com/huggingface/transformers) framework.
We will use a base-model version as a baseline, and see if we can improve performance in a transfer-learning context.
We plan to use *hydra* for configurations, and *wandb* for monitoring experiments. We decided against using the tool Microsoft Live Share in order to gain more familiarity with *git*. Furthermore, we plan to use *invoke* to allow for streamlined command line commands, allowing for e.g. simplifying *git* command configurations by concatenating several commands into one *invoke* task. We beware hubris, and we'll therefore only implement such tasks for *git* once we're familiar with these commands.

# Dataset
We will use the [banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset consisting of 13083 english-languaged customer service queries with 77 intent labels. Query examples include:

<p style="text-align: center;"> 

**Query:** *What can I do if my card still hasn't arrived after 2 weeks?*

**Label:** *card_arrival*

</p>

# Model
We plan to use *TinyBERT*, which is a distilled version of BERT with approximately 15M parameters. We plan to use the CLS-token for the classification head (if they actually made it with a CLS-token). Alternatively, we will simply build the head on top of the encoded input.



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

# Martin & Ludviks thursday log 
added unittests for model outputs and dataset (note: skipped train unittest)
- refactored data.py
- added github actions with unittests for python 3.11 on ubuntu, mac, windows (model and text_dataset) 
- running docker build with torch no cuda version for quick debugging. Ensuring wandb auth works 
	- docker run -e WANDB_API_KEY=<your-api-key> wandb:latest
- made dvc init + added bucket banking77 to dvc. 
- Made cloud bucket in gcs - give max emails for invitation pls
	- todo: add Adam Troels storage object admin
- added docker artifact registry located at europe-west1-docker.pkg.dev/cleaninbox-448011/container-registry 
(had to add service account logging permissions: gcloud projects add-iam-policy-binding cleaninbox-448011 \ --member="serviceAccount:170780472924-compute@developer.gserviceaccount.com" \ --role="roles/logging.logWriter")

- active: make cloudbuild - 
current status: build fails when not running with .gcloudignore which explicitly forcing includes !dockerfiles/* !cloudbuild.yaml. Seems to be an issue with .gitignore, as .gitignore is sourced automatically during cloud build when no .gcloudignore is present.
Need to figure out why this is the case. ChatGPT says that it is probably because of repetitions in gitignore and cloudbuild. Need to check gitignore for multiple exclusions and ensure correctness of subdirs.

- Ludvik atm building API for online inference, I believe 

# To do friday
- Let us please clean up .gitignore, it looks pretty bad, don't know if it behaves as supposed - this could possibly mitigate the need of .gcloudignore for building docker
- invitations to bucket 
- (finish?) API
- Ensure model is fully deployable in cloud using vertex 
- Add wandb secrets to gcloud 
- Do a hyperparameter sweep on full trainset 
- Log samples during training with plot of input sentence and top-5 prediction class distributions. Losses seem too low. 


