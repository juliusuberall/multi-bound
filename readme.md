Repo structure for model training, analysis and result plotting pipeline
--------------------------------------------------------------------------------
Multi-Bound
│── .vscode/                  # Launch settings
│   ├── launch.json           # Debugging profiles
│   ├── tasks.json            # Task definition for full pipeline execution
│── data/                     # Data to fit e.g. RGBA image
│── model_configs/            # Stores YAML configurations for all model architectures
│   ├── mlp.yaml              # MLP models
│   ├── xs.yaml               # Extra small models / Currently all model profiles to test
│── results/                  # NOT GIT TRACKED - Analysis data, plots, signal reconstruction
│   ├── models/               # Saved model parameters
│   ├── plot/                 # Analysis result plots
│   ├── raw/                  # Raw analysis results as .json
│   ├── reconstruction/       # Training signal reconstruction 
│── src/                      # All main scripts for execution
│   ├── analyze_inference.py  # Analyzes inference of models e.g. speed, accuracy 
│   ├── format_results.py     # Processes and visualizes analysis results
│   ├── train_models.py       # Instantiates models and trains until saturation
│── utils/                    # Utility functions
│   ├── model/                # Model implementations for MLP, MoE inlcuding forward(), loss(), deserialization()
│   │   ├── BaseModel.py      # Abstract base model
│   │   ├── MLP.py            # Multilayer Perceptron
│   │   ├── MoE.py            # Mixture of Experts, soft trained and soft inferred
│   │   ├── MoEH.py           # Mixture of Experts, soft trained and hard inferred
│   │   ├── registry.py       # Centralized registry for all models
│   ├── parameter/            # Model parameter implementations
│   │   ├── BaseParams.py     # Abstract base parameter
│   │   ├── MLPParams.py
│   │   ├── MoEParams.py 
│   ├── DataSampler.py        # Samples data to fit model to
│   ├── registry.py           # Centralised folder structure defintion and general pipeline settings
│   ├── Analyzer.py           # Functions for evaluating models
│── readme.md                 # Project documentation
│── .gitignore                # Ignore unnecessary files