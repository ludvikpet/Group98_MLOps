# Group98_MLOps
Now an org!

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


# Checklist

### Week 1

* [ ] Create a git repository (M5)
* [ ] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [ ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub
