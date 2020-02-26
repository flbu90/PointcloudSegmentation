# PointcloudSegmentation
Object segmentation in Pointcloud Rooms (Masterarbeit)

This master thesis proposes the development of a fully automatic point cloud
segmentation pipeline using clustering and convolutional neural networks
(CNN). Therefor different CNNs were trained to identify the number of
objects in a point cloud scene. Experiments showed that training the CNN
with scene-based data sets lead to a high accuracy of prediction. Using the
information about the number of objects in the point cloud, several cluster
methods were tested for their segmentation quality. Results showed that the
Gaussian Mixture models provide the best but also the worst scores. To
address this problem, a cluster validation was developed that identifies the
best clustering of multiple clustering candidates.
