# Shared_Task_Information_Science
The PreTENS shared task hosted at SemEval 2022 aims at focusing on semantic competence with specific attention on the evaluation of language models with respect to the recognition of appropriate taxonomic relations between two nominal arguments (i.e. cases where one is a supercategory of the other, or in extensional terms, one denotes a superset of the other).

<h2>To do:</h2>
<ul>
  <li>Add description of the script at the top and name of who made it</li>
  <li>Add comments to each script</li>
  <li>Make a file (or set of files) with all upsampled data</li>
  <li>Make a detailed overview of the difference between the original data and the upsampled data</li>
  <li>Add results in a txt file in the results folder! Also add the settings you used to create these results</li>
  <li>Remove redundant files. Only keep the files that currently serve a purpose or are important for reproducibility</li>
</ul>

<!-- GETTING STARTED -->
## Getting Started

Before you can start creating your models we firstly need to generate the training data we utilised for training our models.

### Prerequisites (only for classification task)

To create a train set using implimentation 1, run the following.
  ```
  GenerateSentence.py -i1
  ```
  To create a train set using implimentation 2, go to the Upsampling directory and run the following.
  ```
  Create_new_sentence_label. py
  ```
  These sentences would then need to be manually added the the train set created using implementation 1
  To create a train set using implimentation 3, run the following.
  ```
  GenerateSentence.py -i3
  ```
### Training your model (SubTask-A)
If you want to run the train sets from implementation 1 and 3, first go to the Models/LM_BERT_Balanced_labels/ directory.

* step 1: open config.json
* for implentation 1: set "training-set" value to "label_template_balanced_train"
* for implentation 3: set "training-set" value to "custom" and set "model" value to "ERNIE"
* step 2:run 
 ```
  train.py
  ```
  * step 3:run
 ```
  test.py
  ```
  * step 4:run
 ```
  evaluate.py
  ```
 
 If you want to run the train sets from implementation 2, first go to the Models/LM/src directory.

* step 1:run 
 ```
  train.py config_11.json
  ```
  * step 3:run
 ```
  test.py config_11.json
  ```
  * step 4:run
 ```
  evaluate.py config_11.json
  ```
  
  ### Training your model (SubTask-B)
To create your Regression model, follow these steps

First go to the Models/REG/ directory.
* step 1: open config.json
* for BERT (full dataset): set "model" value to "BERT"
* for ERNIE (full dataset): set "model" value to "ERNIE"
* step 2:run
 ```
  train_reg.py
  ```
  * step 3:run
 ```
  test.py
  ```
  Go to Output directory
  * step 4:run
 ```
  rho.py
  ```
To run the third BERT model (using only train set), first go to the Models/LM/src directory.
* step 1:run
 ```
  train_reg.py config_11_reg.json
  ```
  * step 3:run
 ```
  test_reg.py config_11_reg.json
  ```
  Go to Output directory
  * step 4:run
 ```
  rho.py 

