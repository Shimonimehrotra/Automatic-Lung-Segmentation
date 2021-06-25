# Automatic-Lung-Segmentation
Segmentation of Chest X-Rays (CXRs) plays a crucial role in computer-aided diagnosis of chest X-Ray (CXR) images. CXRs are one of the most commonly prescribed medical imaging procedures with the voluminous CXR scans placing significant load on radiologists and medical practitioners. Automated organ segmentation is a key step towards computer-aided detection and diagnosis of diseases from CXRs. In this project, we propose a deep convolutional network model trained in an end-to-end setting to facilitate fully automatic lung segmentation from anteroposterior (AP) or posteroanterior (PA) view chest X-Rays, which is based on the UNet++ model architecture, using Efficient Net B4 as the encoder and residual blocks from ResNet to create our decoder blocks. Our network learns to predict binary masks for a given CXR, by learning to discriminate regions of organ, in this case the lungs, from regions of no organ and achieves very realistic and accurate segmentation. In order to generalize our model to CXRs we also employ image pre-processing techniques like top-hat bottom-hat transform, Contrast Limited Adaptive Histogram Equalization (CLAHE), in addition to a couple of randomized augmentation techniques. The datasets chosen by us are also critical components in various computer-aided diagnosis of CXR algorithms. 

## Design and Implementation

1. **IMAGE AUGMENTATION:** 

Image Augmentation, since limited dataset is an obstacle in training deep convolutional neural networks, we can rectify this problem using image augmentation which is nothing but simply creating new images by making small modifications in the previous data set, to provide more images for better training and testing. Some of the modifications are:
 - Rotation – Rotating the image with a very small amount of angle probably 0.5 degree will create a new image. We just have to make sure that while doing this rotation the boundaries of lungs and edges do not go out of the image boundary 
 - Width shift- images are randomly shifted on the horizontal axis by a fraction of total width 
 - Height shift - Images are randomly shifted on the vertical axis by a fraction of the total height 
 - Shearing – It is basically slanting the image; it is different from rotation. In this we fix one axis, and then stretch the image at a certain angle. 
 - Horizontal flip – In horizontal flipping, images are flipped randomly, by doing this modification, the model will be able to better segment chest radiographs of front as well as back.
 

2. **OPTIMIZATION ALGORITHM:**

An optimization algorithm is arguably one of the most important tools of deep learning. These algorithms are responsible for training the deep convolutional neural network (DCNN) by updating the parameters of the network so that it learns to minimize an objective function aka the loss function. In our proposed approach, we employ the Nadam optimization algorithm.The Nadam optimizer has been shown to perform well for medical segmentation tasks and leads to a faster convergence to the local minima of the chosen loss function. 

Loss Function: Deep-learning segmentation frameworks rely not only on the design of the network architecture but also on the type and complexity of the driving loss function. When we train our DCNN, we are essentially trying to run an optimization algorithm (in our case,Nadam) to minimize the chosen loss function. We realized the need of a specialized loss function which appropriately assigns weights to each class in order to balance the bias so that the model can better segment the lungs by providing additional emphasis on learning to classify all lung-related voxels which are a minority. 

Our proposed novel loss function termed ‘Penalty Combo Loss’ (PCL) is defined as: **LPCL = αLBCE + βLPGDL**. 

**Learning Rate Schedule:**
The learning rate is considered to be one of the most important hyperparameters that directly affects the optimization process of a deep neural network (DNN) alongside model training and generalization. It is a positive scale factor which basically supervises the celerity of network convergence in order to reach the global minima by navigating through the non-convex loss function’s high-dimensional spatial surface. This convergence is often affected/delayed due to entrapment of the optimizer function at multiple aberrations such as local minima, saddle points etc.The proposed PCL loss is essentially a weighted sum of Binary Cross-Entropy loss and Penalty Generalized Dice Loss where α and β are the weights assigned to the binary cross-entropy loss and penalty generalized dice loss functions respectively. α and β are hyperparameters which require fine-tuning depending on the application domain. The fine-tuning can be trivially accomplished using a holdout/validation set. 
 - Binary cross-entropy loss (LBCE) is defined as: **𝑳𝑩𝑪𝑬= (− 𝟏/𝑵) NΣi=1𝒈𝒊𝒍𝒐𝒈(𝒑𝒊) + (𝟏−𝒈𝒊)𝒍𝒐𝒈(𝟏−𝒑𝒊)**
 - Penalty Generalized Dice Loss (PGDL) is defined as: **𝑳𝑷𝑮𝑫𝑳= 𝑳𝑮𝑫𝑳/𝟏+𝒌(𝟏−𝑳𝑮𝑫𝑳)**



3. **MODEL ARCHITECTURE:**

Our proposed deep convolutional neural network is inspired from the UNet++ architecture and follows a similar design integrating multiple encoder and decoder blocks in a network ensemble of varying depths wherein all encoder and decoder blocks at the same level are densely connected with skip connections, thereby alleviating the problem of choosing the a-priori unknown optimal depth. We employ EfficientNet B4 as the encoder network and makes use of residual blocks inspired from ResNets in order to incorporate residual learning in the decoder blocks. The network architecture used in this work consists of 4 encoding and 4 decoding layers. Every encoder layer reduces the input feature map size by factor of 2, thus resulting in the combined sub-sampling rate being equal to 16. It is known that large scaling factors can potentially improve desired properties of displacement, rotation and scale invariance of the convolution network being considered in the spatial domain. 


4. **RESULT**

![image](https://user-images.githubusercontent.com/58876793/123479537-b4771c80-d61e-11eb-94d3-e41f9efaa4c0.png)       ![image](https://user-images.githubusercontent.com/58876793/123479554-bb059400-d61e-11eb-9105-50068797c40c.png)       ![image](https://user-images.githubusercontent.com/58876793/123479573-c22ca200-d61e-11eb-915e-8e88e7faeed4.png)

