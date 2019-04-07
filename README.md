# Age Estimation
Age estimation by Pytorch

### The principle code files
KLDiv_version.ipynb tries simple CNN structure and VGG16 structure with KLDiv loss. <br>
VGG16_soft_cross_entropy_version.ipynb uses costomed cross-entropy function for soft label.  <br>
img2vec_MSE_version.ipynb uses a pretrained ResNet18 to get a 512-length vector to represent the image, and we train the fully connected network upon it. 


### Other .ipynb files are supporting files
preprocessing.ipynb: This file is to preprocess the dataset. <br>
remove_outliers.ipynb: This file is to remove the empty images in the dataset. <br>
statistics.ipynb: This file is to draw statistics of the dataset. <br>
curve.ipynb: This file is to draw curves based on data. <br>
cp_part.ipynb: This file to generate a smaller dataset in order to debug the code more quickly <br>

### Data
The data could be found here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ <br>
Currently we are working on 1-GB WIKI Face-Only data set.

There are 50386 images in total in this dataset. The figure below shows the data's distribution by ages. The dataset is split into training, validation and test dateset with the ration 90:5:5. 

![alt text](https://github.com/map583-2019/AgeEstimation/blob/master/pictures/AgeDistribution.png)

As shown in the figure above, the data's distribution across different ages is highly unbalanced. Images around 26-years-old photo are most abundant, while photos under 15 years' old or higher than 85 years' old are limited. 

As Azure platform budget had expired, I trained the model with my local CPU. To get result in a waitable time, I cropped the training dataset with a 50-image ceiling. The new cropped dataset's distribution by age is as follows:

![alt text](https://github.com/map583-2019/AgeEstimation/blob/master/pictures/cropped_dataset_distribution.png)

The data is not very vlean. I removed the image files with no pixels and images with negative age label. Images in the dataset still have deficits such as <br>
&nbsp&nbsp the labeled age could be appararently not correct, <br>
&nbsp&nbsp The face may be very small in the image. 


### Results
On my cropped dataset, SimpleCNN-KLDivLoss version, VGG16-KLDivLoss version and VGG16-soft-cross-entropy version failed to converge during training. 
Based on the work of Samet Çetin ( https://github.com/cetinsamet/age-estimation ) I implement a third version. It uses ResNet18 to get a 512-length image representation, and then uses 3 fully connected layers to do regression. I trained this network on my cropped dataset. I did get a descending training loss process as shown below:

![alt text](https://github.com/map583-2019/AgeEstimation/blob/master/pictures/training_loss_evaluation_img2vec.png)

but the prediction is not good either. It seems that the trained model always tends to predict an age between 40 and 50, as shown below:

![alt text](https://github.com/map583-2019/AgeEstimation/blob/master/pictures/test_cetinsamet_trained_on_limited_imdb_data.png)

I also tested directly the model trained by Samet Çetin, which is better than mine but also limited on our dataset: 

![alt text](https://github.com/map583-2019/AgeEstimation/blob/master/pictures/test_cetinsament.png)

It predicts better for images between 20 and 40 years old, while not so good outside this range. 

### Future works
The principle is relatively simple in this task, but my code does not reach a satisfying results. The reason might be the lack of data (at most 50 images for each age in my dataset, at the dataset is not very clean), the unoptimized nework structure, the unoptimized data preprocessing jobs, or something else. <br>
In the future, if computing resources are provided, the priority job is to enlarge the dataset as well as upgrade network structure accordingly. 