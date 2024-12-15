#!/bin/bash
# Download the dataset
kaggle datasets download ameencaslam/ddp-v4-models

# Unzip the models
unzip ddp-v4-models.zip

# Rename the folder
mv ddp-v4-models converted_models

# Cleanup
rm -rf ddp-v4-models.zip