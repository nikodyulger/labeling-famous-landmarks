# Labelling Famous Landmarks
This is my bachelor's final degree project. The main objective is the construction of Deep Learning models for landmark classification based on the **Google Landmark Recognition 2020** database. For its development an end to end data mining procedure has been followed.

# Structure of the project
The work has been divided in several jupyter notebooks, each one for each stage of the project.

* Exploratory Analysis
* Data Cleaning
* Model Construction
* Result Analysis
* Explainability

There are other additional files such as:

* The `.csv` files with the image sets extracted from the original database 
* The registry of all the result from the hyperparameter tuning process
* The ranking with the best models
* Best model files

Moreover, a Tensorboard is provided with all the model learning curves in this [link](https://tensorboard.dev/experiment/5gfr6pkJRDS6cPD6X8R99Q/)

# Development

This project has been mainly built with **Keras** and **Tensorflow**. Two main applications were chosen as cornestones for the  custome models: **VGG16** and **Xception**. Also, the transfer learning experiments carried out have been divided in two main tecniques: **feature extraction** and **fine tuning**. 


# Acknowledgements
This project as been developed under the tutorship of José Miguel Puerta Callejón and José Antonio Gámez at the University of Castilla La-Mancha
