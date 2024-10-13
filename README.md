data_class.py contains code for preparing the data for CNN training by attaching images to their classes

functions.py contains simple functions used in this work such as for plotting light curves and training the model
lightkurve_data.py contains code for accessing the parquet files and plotting the images alongside creating their label csv files
main.py contains the main part of the code where the data, model is initialised and trained, plotting metric plots at the end
model_skeleton.py contains the architecture of the custom CNN
optuner.py contains code using optuna to tune hyperparameters
