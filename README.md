week4/
├──.gitignore - specifies files to be ignored by git
├──.dvcignore - specifies files to be ignored by DVC
├──.github
│   └── workflows
│       └── ci-dev.yml - GitHub Actions workflow for CI/CD executed when pushed to dev branch
│       └── ci-main.yml - GitHub Actions workflow for CI/CD executed when pushed to main branch
│       └── cml-report.yml - Workflow 
├── data
│   └── iris.csv - data file for the Iris dataset
├── requirements.txt - list of dependencies for the project
├── src
│   ├── evaluate.py - script for model evaluation
│   ├── __init__.py
│   └── train.py - script for model training
└── tests
    ├── __init__.py
    ├── test_data_validation.py - tests for data validation
    └── test_evaluation.py - tests for model evaluation