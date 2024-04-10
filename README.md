# Predicting Barbell Exercises
- This project was based on [Dave Ebbelaar](https://github.com/daveebbelaar)'s tracking barbell exercises [project](https://github.com/daveebbelaar/tracking-barbell-exercises). He collected the data during gym workouts where participants were performing various barbell exercises using the [Mbientlab's WristBand Sensor Research Kit](https://mbientlab.com/).
- Also, the original code is associated with the book titled "Machine Learning for the Quantified Self" authored by Mark Hoogendoorn and Burkhardt Funk and published by Springer in 2017. The website of the book can be found on [ml4qs.org](https://ml4qs.org/).
  
![Basic Barbell Exercises](https://github.com/NicolasFaleiros/Predicting-Barbell-Exercises/assets/41973874/4002eb01-bddd-42fe-92e7-8ffd03594a6c)

# 1. Description
- This is an end-to-end machine learning project that uses Feedforward Neural Network, Random Forest and Simple Decision Tree to predict barbell exercises. It uses accelerometer and gyroscope data collected by a fitness device to predict which exercise was performed and how many repetitions.
- I organized the project development into components responsible for data ingestion, transformation, visualization, model training and evaluation. The entire analysis, from EDA to model training and evaluation, was made in these components, and everything is documented.

# 2. Technologies and tools
- The technologies and tools used were Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn, Feedforward Neural Networks, Random Forest, Decision Tree, Principal Component Analysis, Low-Pass Filter, Fourier Transformation), Jupyter Notebook, Git and Github (version control), machine learning classification algorithms, statistics, Anaconda (terminal) and Visual Studio Code (project development environment).

# 3. Business problem and project objective

**3.1 What is the business problem?**
- The management team at a fitness device company  is looking to optimize their clients' workout routines and track their progress more effectively. They want to enhance the user experience by accurately predicting the specific barbell exercises performed by the users based on accelerometer and gyroscope data collected from the fitness devices. This predictive tool will enable them to tailor workout plans according to individual needs, monitor exercise consistency, and provide timely feedback to clients for better results, ultimately improving user satisfaction and retention.

**3.2 What is the context?**
- When a person uses a fitness tracker device they collect data through time, such as data from accelerometer and gyroscope sensors, and it is to be expected that the realization of a specific exercise should result a particular pattern in the data.
    1. Accelerometer sensor measures acceleration (rate of change of velocity) forces acting on the device along three axes: X, Y, and Z. It can detect movement, tilt, vibration, and changes in speed.
    2. Gyroscope sensor measures the rate of rotation or angular velocity of the device around its three axes: X, Y, and Z. Provides information and can detect changes about the orientation, angular velocity, and rotational movement of the device.

  ![image](https://github.com/NicolasFaleiros/Predicting-Barbell-Exercises/assets/41973874/f0cc292e-1f17-4964-876c-4abeb134bc8a)

- We can use the data from these sensors to differentiate between barbell exercises, and build a machine learning model to predict what exercise is being performed by a new instance.
- In order to maintain customer satisfaction and retention, the company whises to accurately label new instances of barbell training exercises from all clients. This will ensure that our fitness device company builds a good relationship with their customers in the sense of helping them structure and track their training routine.

**3.3 What are the project objectives?**
1. Identify exactly which variables contribute to the prediction of barbell exercises.
2. Use feature engineering to perhaps enhance the prediction capabilities of our features.
3. Construct a classification model capable of accurately predicting the probability of the movement being a particular barbell exercise.
4. Offer action plans for the company to avoid mistakes when interpreting the data, to make accurate predictions and preserve customer satisfaction.

**3.4 What are the project benefits?**
- Development of groundbreaking technologies in fitness tracker devices.
- Improved customer satisfaction and retention.
- Improved customer experience.
- Targeted and competitive marketing.

**3.5 Conclusion**
- When deploying the model so that the device can make predictions, the primary objective is to generate probability scores for each set of exercises a customer performs. This is typically more valuable for businesses when compared to making binary predictions (1/0), as it enables better decision-making and more effective interpretation of the data.
- For instance, predicting the probability of it being any specific exercise provides more actionable insights. Instead of simply determining whether a customer is performing exercise A, B or C, you gain an understanding of how likely it is to be each exercise. This information enables the device company to allocate its efforts and resources more effectively.  For instance, if certain exercises are predicted to be less likely to be performed by a particular user demographic, the company can focus its resources on promoting or enhancing those exercises to encourage greater engagement. Conversely, exercises with high predicted probabilities may already be popular among users, prompting the company to invest more resources in refining or expanding related features or content, and so on.

# 4. Solution pipeline
1. Define the business problem.
2. Converting raw data, reading CSV files, cleaning.
3. Explore the data (exploratory data analysis)
4. Feature engineering, data cleaning and preprocessing.
5. Split the data into train and test sets.
6. Model training, comparison, feature selection and tuning.
7. Final production model testing and evaluation.
8. Conclude and interpret the model results.

Further explanation for each step can be found inside the python files, where I provide the rationaly for some decisions made. But here I'll give an overview of what was made in each step, right after I highlight the main business insights.

# 5. Main business insights


# 6. Explaining my approach

**6.1 Define the business problem**
- The definition of the business problem was made in the introduction.

**6.2 Converting raw data, reading CSV files, cleaning**
- The original data collected by the device was separated by participant (5 subjects), type of exercise (Bench Press, Overhead Press, Barbell Row, Squat, Deadlift), category of exercise (medium, 5 reps, or heavy, 10 reps), and whether it is accelerometer or gyroscope data. Each are stored in separate csv files. The first objctive in the `make_dataset.py` file was to extract all these features from the file names and join all of them into a single dataframe.
- The data was collected by the device multiple times a second. So I used a resample method to restructure the data in a way that every instance within 200ms is encompassed together by the mean and becomes the new observation.
- A "set" column was added to the dataframe in the proccess to distinguish between the sets performed by each subject.

** 6.3 Explore the data (exploratory data analysis)**
- 


