# Fundus Image-Based Diagnosis of Multiple Ocular Diseases Using Deep Convolutional Neural Networks

The following study examines one of the most substantial problems in the medical field, the diagnosis of ocular diseases. According to the WHO, over 2.2 billion people live with a vision impairment. Leading causes of vision complications include conditions such as diabetes, cataracts, glaucoma, and myopia. Unfortunately, many of these ailments only present symptoms at advanced and irreversible stages. Consequently, it is imperative to diagnose these diseases early to prevent long-term health-related burdens. We propose a novel detection system for multiple ocular diseases using a deep convolutional neural network. The model is capable of identifying several diseases from fundus images of the retina, achieving an AUC score of 0.75. This study aims to serve as an effective assistive diagnostic tool for eye-care professionals through the use of a CNN. We believe that our research shows an automated system for eye disease detection is possible for future clinical use.

# Results
| Model                      | AUROC | Recall | Precision | F1-Score |
|:--------------------------:|:-----:|:------:|:---------:|:--------:|
| SGD-VGG-19                 | 0.75  |  0.57  |   0.78    | 0.66     |
| SGD-VGG-19 (No thresh.)    | 0.72  |  0.68  |   0.66    | 0.67     |
| Nadam-VGG-19               | 0.74  |  0.59  |   0.76    | 0.66     |
| SGD-SE-ResNet-34           | 0.70  |  0.53  |   0.69    | 0.60     |
| Nadam-SE-ResNet-34         | 0.71  |  0.54  |   0.71    | 0.62     |
| Class-Weight-SE-ResNet-34  | 0.70  |  0.49  |   0.73    | 0.59     |
| No-ES-SE-ResNet-34         | 0.72  |  0.50  |   0.75    | 0.60     |
| Oversampled-SE-Resnet-34   | 0.60  |  0.42  |   0.70    | 0.53     |

Our models were trained on an 80/10/10 train/validation/test split, with the training dataset containing 11,020 images and the validation and test set containing 1377 and 1378 images respectively. We chose these splits to maximize the amount of training data available due to the relatively small size of our dataset. The validation set was used to monitor the generalization error while training and to tune the hyperparameters. Of the 7 model architectures and configurations that we tested, the VGG-19 model trained with the stochastic gradient descent optimizer achieved the performance, with an AUC score of 0.75, recall of 0.78,  precision of 0.57 and an F1-score of 0.66 on the test set. These metrics were chosen based on their effectiveness in evaluating the performance of a model on imbalanced datasets, which is the case for this scenario.

# Libraries Used
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* sci-kit learn

# ocularnn Webapp Design
This webapp is made with Flask and is meant as a proof of concept for a
potential virtual solution for ocular disease detection. This would be used by
medical professionals to help diagnose patients with certain ocular diseases

* Frontend (probably use Bootstrap)
    * Doctor login/registration page
    * A prediction page for user upload and viewing their image, display
      some stats about the image (std, mean, size).
        * Maybe two centered flexboxes, one for uploaded and second for 
        preprocessed image
        * Also have the option of randomly selecting an image from the test
        set for evaluation
        * Left and right eyes, allow for only the input of one of them as well
    * A dashboard for the "doctor's" patients
    * A view for each patient describing the model's predicted confidence of 
    each disease being present

* Backend
    * Add user authentication
    * Doctor uploads image sees the preprocessed output, then a prediction is made
    * SQLAlchemy DB
        * Each doctor will have patients which in turn will have data about
        them (ex. Name, age, gender, ID, their image data and their detected
        diseases)
    * Model serving predictions, display response to doctor and store the results
    in the DB

# References

Arjun, S., Saluja, K., & Biswas, P. (2021, January 04). Analysing ocular parameters for web browsing and graph visualization. Retrieved March 4, 2021, from https://arxiv.org/abs/2101.00794

Author Bill Holton, R., & Holton, B. (2017, April 12). Vision tech: Earlier eye disease detection may be possible thanks to new research. Retrieved March 27, 2021, from https://www.afb.org/aw/17/11/15390

Discovery Eye Foundation, D. (2016, March 24). The costs of eye care. Retrieved March 2, 2021, from https://discoveryeye.org/the-costs-of-eye-care/

Fu, Y., Li, F., Wang, W., Tang, H., Qian, X., Gu, M., & Xue, X. (2020, September 04). A new screening method FOR COVID-19 based ON OCULAR feature recognition by machine learning tools. Retrieved March 4, 2021, from https://arxiv.org/abs/2009.03184

He, K., Zhang, X., Ren, S., & Sun, J. (2015, December 10). Deep residual learning for image recognition. Retrieved March 2, 2021, from https://arxiv.org/abs/1512.03385

Hu, J., Shen, L., Albanie, S., Sun, G., & Wu, E. (2019, May 16). Squeeze-and-excitation networks. Retrieved March 3, 2021, from https://arxiv.org/abs/1709.01507

Iskander, J., & Hossny, M. (2020, August 12). An ocular biomechanics environment for reinforcement learning. Retrieved March 27, 2021, from https://arxiv.org/abs/2008.05088

Jung, Y., Park, J., Low, C., Tiong, L., & Teoh, A. (2020, December 12). Periocular in the Wild Embedding learning with cross-modal Consistent Knowledge Distillation. Retrieved March 5, 2021, from https://arxiv.org/abs/2012.06746

Krishnan, A., Almadan, A., & Rattani, A. (2020, November 17). Probing fairness of mobile ocular biometrics methods across gender on visob 2.0 dataset. Retrieved March 4, 2021, from https://arxiv.org/abs/2011.08898

Langholz, E. (2019, May 22). Oculum AFFICIT: Ocular Affect Recognition. Retrieved March 6, 2021, from https://arxiv.org/abs/1905.09240

Li, N., Li, T., Hu, C., Wang, K., & Kang, H. (2021, February 16). A benchmark of Ocular DISEASE INTELLIGENT recognition: One shot for MULTI-DISEASE DETECTION. Retrieved March 4, 2021, from https://arxiv.org/abs/2102.07978

Macantosh, H. (2015, April 12). Multiple eye disease detection using deep neural network. Retrieved March 27, 2021, from https://ieeexplore.ieee.org/document/8929666

Meller, G. (2020, December 21). Ocular disease recognition using convolutional neural networks. Retrieved March 4, 2021, from https://towardsdatascience.com/ocular-disease-recognition-using-convolutional-neural-networks-c04d63a7a2da

Nations, U. (2019, March 12). Eye care, vision impairment and blindness. Retrieved March 2, 2021, from https://www.who.int/health-topics/blindness-and-vision-loss#tab=tab_1

Pantanowitz, A., Kim, K., Chewins, C., Tollman, I., & Rubin, D. (2020, November 24). Addressing the eye-fixation problem in gaze tracking for human computer interface using the Vestibulo-ocular reflex. Retrieved March 4, 2021, from https://arxiv.org/abs/2009.02132

Simonyan, K., & Zisserman, A. (2015, April 10). Very deep convolutional networks for large-scale image recognition. Retrieved March 2, 2021, from https://arxiv.org/abs/1409.1556
University of British Columbia, U. (2018, March 12). Color fundus Photography. Retrieved March 2, 2021, from https://ophthalmology.med.ubc.ca/patient-care/ophthalmic-photography/color-fundus-photography/

Zanlorensi, L., Laroca, R., Luz, E., Britto Jr., A., Oliveira, L., & Menotti, D. (2019, November 21). Ocular recognition databases AND Competitions: A survey. Retrieved March 3, 2021, from https://arxiv.org/abs/1911.09646

Zanlorensi, L., Lucio, D., Britto Jr., A., Proença, H., & Menotti, D. (2019, November 21). Deep representations FOR Cross-spectral Ocular Biometrics. Retrieved March 4, 2021, from https://arxiv.org/abs/1911.09509

Zhang, Z., Srivastava, R., Liu, H., Chen, X., Duan, L., Kee Wong, D., . . . Liu, J. (2014, August 31). A survey on computer aided diagnosis for ocular diseases. Retrieved March 7, 2021, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4163681/

H. He and E. A. Garcia, "Learning from Imbalanced Data," in IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263-1284, Sept. 2009, Retrieved March 6th 2021 from doi: 10.1109/TKDE.2008.239.

Fan, R., & Lin, C. (2007). A Study on Threshold Selection for Multi-label Classification.

Smith, L. N. (2018). A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay. ArXiv:1803.09820 [Cs, Stat]. Retrieved March 6th 2021 from http://arxiv.org/abs/1803.09820 

Kingma, D. P., & Ba, J. (2017). Adam: A method for stochastic optimization. ArXiv:1412.6980 [Cs]. Retrieved March 6th 2021 from http://arxiv.org/abs/1412.6980

Hoffer, E., Hubara, I., & Soudry, D. (2018). Train longer, generalize better: Closing the generalization gap in large batch training of neural networks. ArXiv:1705.08741 [Cs, Stat]. Retrieved March 6th 2021 from http://arxiv.org/abs/1705.08741

Krause, J. et al. Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy. Ophthalmology (2018). Retrieved March 6th 2021 from doi:10.1016/j.ophtha.2018.01.034

DuEE, B. (2017, March 12). Baidu research Open-Access dataset - Introduction. Retrieved March 2, 2021, from http://ai.baidu.com/broad/introduction

Guillaume PATRY, G. (2021, February 18). Messidor-2. Retrieved March 2, 2021, from https://www.adcis.net/en/third-party/messidor2/

None, I. (2018, April 12). IDRiD (Indian diabetic RETINOPATHY Image Dataset). Retrieved March 2, 2021, from https://academictorrents.com/details/3bb974ffdad31f9df9d26a63ed2aea2f1d789405

Challenge, G. (2107, November 12). Riadd (isbi-2021) - grand challenge. Retrieved March 2, 2021, from https://riadd.grand-challenge.org/Home/

Diaz-Pinto, A., Morales, S., Naranjo, V. et al. CNNs for automatic glaucoma assessment using fundus images: an extensive validation. BioMed Eng OnLine 18, 29 (2019). https://doi.org/10.1186/s12938-019-0649-y 

