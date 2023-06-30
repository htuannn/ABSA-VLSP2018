# Multi-Tasks Learning ABSA on Vietnamese Hotel Reviews
This repository focuses on the task of Aspect-Based Sentiment Analysis (ABSA) for Vietnamese Hotel Reviews. It provides implementations of multi-task learning techniques to enhance the performance of ABSA on this specific domain.

## Introduction
Aspect-Based Sentiment Analysis is a natural language processing task that aims to identify and analyze sentiments expressed towards specific aspects within a text. In the context of this repository, ABSA is performed on Vietnamese hotel reviews, where the goal is to identify aspects (e.g., "FOOD#QUALITY," "HOTEL&ROOM#PRICES) and their corresponding sentiment polarities (e.g., positive, negative, neutral).

We focus on the task of text-level Aspect-Based Sentiment Analysis (ABSA).
## Annotate Custom Dataset

This repository provides a framework that simplifies the process of annotating data for the Aspect-Based Sentiment Analysis (ABSA) text-level problem. It includes an annotation tool and instructions on how to set it up and use it with your custom dataset.

### Annotation Tool Setup
please ensure that Streamlit is installed in your environment. You can install it by running the following command:

``` pip install streamlit ```

To run the annotation tool, execute the following command: ``` streamlit run Annotation_lab.py ```

### Dataset Format
Ensure that your custom data is structured as follows:


```
#1
text1
{FOOD#STYLE&OPTIONS, neutral}, {FOOD#QUALITY, neutral}

#2
text2  #reviews
       #label (blank if not annotated)

...
```

### Configuration

To configure the annotation tool to match your dataset, you can modify or create a new YAML config file.

_Note: Example config file for our dataset: `config/absa_config_anno.yaml_
## Prepare Environment
### Requirements
Install the necessary dependencies by running: ``` pip install -r requirements.txt ```

### Model Configuration Setup

Please organize your configuration file for training and evaluation in the following structure: 
```
Main-folder/
│
├── config/ 
│   ├── absa_model.yaml - This file contains configuration for training model
|   │
|	└── absa_anno_config.yaml
└── ...
```

## Usage

### Training
In this project, we focus on training Network based on pretrainded PhoBert embedding model. Executing the following command for trainning:

```
python trainer.py
```

If you want to fine-tune the embedding model, set `freeze_embedder: False` in `config/absa_model.yaml` file.

_Note: When making any changes or customizations for training on a custom dataset, remember to update the config/absa_model.yaml file according to your data._
### Evaluation
To evaluate the model on the test set, use the following command:

```
python eval.py
```

