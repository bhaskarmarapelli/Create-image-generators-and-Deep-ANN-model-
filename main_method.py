import preprocess as ps
import classification_model as cm

data_path = r"C:\Users\Bhaskar Marapelli\Downloads\gaussian_filtered_images\gaussian_filtered_images"
da=ps.Dataprocess()
dataset_df = da.generate_images_dataset(data_path)
# Save the DataFrame to a CSV file
csv_filename = "diabetic_retinopathy_dataset.csv"
dataset_df.to_csv(csv_filename, index=False)
print(f"Dataset information saved to {csv_filename}")
data=da.Generate_new_feature_in_csv(csv_filename)
binary_csv="binary_dataset.csv"
data.to_csv(binary_csv, index=False)
da.data_EDA(binary_csv)

train_batches, val_batches, test_batches=da.ImageDataGenerator_Data(binary_csv)

obj=cm.Classifcation_model()
model=obj.simple_model()
model=obj.compile_model(model)
history=model.fit(train_batches, epochs=10, validation_data=val_batches)
obj.summary_model(history)