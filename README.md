# Smørrebrød Image-to-Recipe Retrieval Project

## Overview
The Smørrebrød Image-to-Recipe Retrieval Project is an ambitious endeavor to bridge the gap between the culinary arts and artificial intelligence. By leveraging state-of-the-art computer vision and natural language processing techniques, this project aims to create a system that can recognize ingredients and reconstruct recipes from images and vice versa. Inspired by CLIP [12], the project utilizes a multimodal approach to understanding food by linking recipes and images.

## Goals
- To retrieve a recipe from a list of known recipes given an image query.
- To retrieve an image from a list of known images given a text recipe.

## Data
The project utilizes the Food Ingredients and Recipes Dataset from Kaggle, comprising 13,582 images, each paired with a title, a list of ingredients, and cooking instructions.

## Tasks
1. **Image-to-Recipe Retrieval**: Develop a model that maps food images to their corresponding recipes using an image encoder (CNN or visual transformer) and a text encoder (text transformer or BERT).
2. **Additional Text Modalities**: Enhance the model from Task 1 by incorporating extra textual information like instructions and ingredients during training.
3. **Comparison with CLIP**: Benchmark the developed models against the performance of the CLIP model [12].

## Project File Structure
```
.
food_bois/
│
├── DJ/ # Daniel & James project folder
│   ├── models/
│   ├── notebooks/
│   └── scripts/
│
├── EM/ # Emil & Magnus project folder
│   ├── models/
│   ├── notebooks/
│   └── scripts/
├── data
├── requirements.txt
├── README.md
└── .gitignore
```

## Virtual Environment
The project used python `3.10.7`. For testing reasons please use a virtual environment with the `requirements.txt` file.
Preferable either with the name `deep` or `venv` as the `.gitignore` filters it out. To use the virtual environment on windows use the command to generate the virtual environment.

```
python3 -m venv deep
```
To activate the environment do:

```
.\deep\Scripts\activate.ps1
```
To install the dependencies for the environment use:
```
pip3 install -r requirements.txt
```
and to stop the virtual environment.
```
deactivate
```
Note that if you want cuda to work with the software then use the following commands after installing `requirements.txt`:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118




### Directory Descriptions:
- `data/`: Contains the datasets required for each model.
- `models/`: Stores the trained model checkpoints.
- `notebooks/`: Used for storing Jupyter notebooks.
- `scripts/`: Stores various utility scripts or the main code to run the experiments.

## Virtual Environment
The project is developed using Python version `3.10.7`. To ensure compatibility, please use a virtual environment. The preferred environment names are `deep` or `venv`, as they are excluded by `.gitignore`.

### Environment Setup on Windows:
Create the virtual environment:
