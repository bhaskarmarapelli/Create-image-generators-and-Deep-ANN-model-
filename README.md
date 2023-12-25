Aim: Create image generators and Deep ANN model 
Project structure

Poject
  |_ Preprocess.py  
  |_ Classifcation_model.py
  |_ Main_method.py
 
Preprocess.py
class Dataprocess contains various methods for data processing, exploration, and preparation for a machine learning model, particularly for binary classification of diabetic retinopathy images.
1.	generate_images_dataset: This method takes a data_path as input and generates a DataFrame containing image paths and corresponding labels. It iterates through subdirectories in the data_path and extracts image paths and labels.
2.	Generate_new_feature_in_csv: This method takes a CSV file path as input, reads the data from the CSV, and generates new features based on mappings provided in the method.
3.	data_EDA: Performs exploratory data analysis (EDA) on the dataset by creating a count plot of image labels using the seaborn library.
4.	data_separation: Splits the dataset into training, validation, and testing sets using train_test_split.
5.	mkdir_separate_image: Creates working directories for train, validation, and test sets, and copies images to their respective directories based on the dataset split.
6.	ImageDataGenerator_Data: Uses Keras' ImageDataGenerator to generate batches of images for training, validation, and testing from the directory structure created by mkdir_separate_image.
Classifcation_model.py
‘Classifcation_model: Defines a class that will contain methods related to creating and working with classification models.
1.	simple_model: Defines a simple neural network model using the Sequential API. The model consists of a Flatten layer (for flattening the input), followed by several Dense layers with different numbers of neurons and ReLU activation functions. The output layer has 2 neurons with a softmax activation function, suitable for binary classification.

2.	compile_model: Takes a model as input and compiles it using the Adam optimizer, categorical cross-entropy loss function, and accuracy as the metric.

3.	summary_model:  The summary_model method uses matplotlib to plot the training history. The first plot shows the training and validation accuracy over epochs, and the second plot shows the training and validation loss over epochs.


Main_method.py

1.	Data Preprocessing:

data_path = r"C:\Users\Bhaskar Marapelli\Downloads\gaussian_filtered_images\gaussian_filtered_images": Specifies the path to the directory containing diabetic retinopathy images.

da=ps.Dataprocess(): Creates an instance of the Dataprocess class from the preprocess module.

dataset_df = da.generate_images_dataset(data_path): Generates a DataFrame (dataset_df) containing image paths and labels by calling the generate_images_dataset method of the Dataprocess class.

csv_filename = "diabetic_retinopathy_dataset.csv": Specifies the name of the CSV file to which the dataset information will be saved.

dataset_df.to_csv(csv_filename, index=False): Saves the dataset information to a CSV file named "diabetic_retinopathy_dataset.csv".

data=da.Generate_new_feature_in_csv(csv_filename): Reads the CSV file, generates new features, and creates a DataFrame (data) using the Generate_new_feature_in_csv method.

binary_csv="binary_dataset.csv": Specifies the name of the CSV file to which the binary dataset information will be saved.

data.to_csv(binary_csv, index=False): Saves the binary dataset information to a CSV file named "binary_dataset.csv".

da.data_EDA(binary_csv): Performs exploratory data analysis on the binary dataset by calling the data_EDA method.

2.	Image Data Preparation:

train_batches, val_batches, test_batches=da.ImageDataGenerator_Data(binary_csv): Generates batches of images for training, validation, and testing using the ImageDataGenerator_Data method.

3.	Model Creation and Training:

obj=cm.Classifcation_model(): Creates an instance of the Classifcation_model class from the classification_model module.

model=obj.simple_model(): Creates a simple neural network model using the simple_model method.

model=obj.compile_model(model): Compiles the model using the compile_model method.

history=model.fit(train_batches, epochs=10, validation_data=val_batches): Fits the model to the training data for 10 epochs using the training batches and validates it on the validation data.

obj.summary_model(history): Visualizes the training history using the summary_model method.

