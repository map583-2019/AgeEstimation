# Age Estimation
Age estimation by Pytorch

### The SecondAttempt.ipynb file is the main file for the code
soft_cross_entropy_version.ipynb uses costomed cross-entropy function for soft label. This file is still under development.

### Other .ipynb files are supporting files
preprocessing.ipynb: This file is to preprocess the dataset. <br>
remove_outliers.ipynb: This file is to remove the empty images in the dataset. <br>
statistics.ipynb: This file is to draw statistics of the dataset. <br>
curve.ipynb: This file is to draw curves based on data. <br>
cp_part.ipynb: This file to generate a smaller dataset in order to debug the code more quickly <br>

### Data
The data could be found here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ <br>
Currently we are working on 1-GB WIKI Face-Only data set.

Data Description

There are 50386 images in total in this dataset. The figure below shows the data's distribution by ages. The dataset is split into training, validation and test dateset with the ration 90:5:5. 

![alt text](https://github.com/map583-2019/AgeEstimation/blob/master/pictures/AgeDistribution.png)

As shown in the figure above, the data's distribution across different ages is highly unbalanced. Images around 26-years-old photo are most abundant, while photos under 15 years' old or higher than 85 years' old is limited. 

As Azure platform budget had expired, I trained the model with my local CPU. To get result in a waitable time, I cropped the training dataset with a 50-image ceiling. The new cropped dataset's distribution by age is as follows:

![alt text](https://github.com/map583-2019/AgeEstimation/blob/master/pictures/cropped_dataset_distribution.png)


### Results

