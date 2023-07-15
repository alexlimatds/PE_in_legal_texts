# Leveraging Positional Encoding to Improve Fact Identification in Legal Documents

This repository holds code and artifacts of the extended abstract titled *Leveraging Positional Encoding to Improve Fact Identification in Legal Documents* accepted in the [First International Workshop on Legal Information Retrieval](https://tmr.liacs.nl/legalIR/).

For details about the models, check the paper in the [workshop proceedings](https://tmr.liacs.nl/legalIR/LegalIR2023_proceedings.pdf).

Files with a `report` prefix holds the results of a specific model.

### Running models

There is a running script for each model. For example, to run the BERT+PE(C) model we execute the `run_BERT_PE_C.py` file. Running a model yields the respective report file. There is no command line parameters.

We are not sure if we are allowed to provide the dataset since it was derived from three other ones. Although, it is easy to craft the dataset: kept the *Facts* labels and replace the other ones with the *Other* label. The original datasets are available [here](https://github.com/Law-AI/semantic-segmentation), [here](https://legal-nlp-ekstep.github.io/Competitions/Rhetorical-Role/) and [here](https://github.com/Exploration-Lab/CJPE). Check the `application_*.py` files to discover the required path for the dataset.

The hyperparameters of a model can be set in the respective run script. In the following we describe such hyperparameters.

TODO: describe hyperparameters
