from pyntcloud import PyntCloud
import os
import numpy as np
import random
from random import randint, uniform
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

# ---------------------------------------------- Variables -------------------------------------------------------------

DATADIR = "scannet_ply_labels\scans\\"

# Scene Container creating the Batches for Traning
scene_container_x = {}
scene_container_y = {}

# dict for evaluating past losses
loss_dict = []

# Counter on how many scenes the Network has been trained
counter_trainingobjects = 0

# Number of scenes for Batch-Training
number_scenes = 32

# Voxel Size
voxel_size = 192


# ---------------------------------------------- Functions/Classes ----------------------------------------------------


class LossHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

        
# --------------------------------------------- Network Setup ---------------------------------------------------------


# To activate type in console "tensorboard --logdir=logs\\" and then start training and open link from consoleinput
tensorboard = TensorBoard(log_dir="logs\\{}".format(time()))

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((voxel_size, voxel_size, voxel_size, 1), input_shape=(voxel_size, voxel_size, voxel_size)),  # 3D CNN expects a 5 dim Tensor as Input
    tf.keras.layers.Conv3D(filters=8, kernel_size=(2,2,2), strides=2, activation='relu'),
    tf.keras.layers.Conv3D(filters=16, kernel_size=(2,2,2), strides=2, activation='relu'),
    tf.keras.layers.Conv3D(filters=32, kernel_size=(2,2,2), strides=2, activation='relu'),
    tf.keras.layers.Conv3D(filters=64, kernel_size=(2,2,2), strides=2, activation='relu'),
    tf.keras.layers.Conv3D(filters=128, kernel_size=(2,2,2), strides=2, activation='relu'),
    tf.keras.layers.Flatten(),
    
    ## create an MLP architecture with dense layers : 2048 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(30, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

# ------------------------------------- Begin Trainig --------- ------------------------------------------


# Main-Loop for Training till User interruption
while True: 
    
    #try: 
        
	v = 0  # Till number of Scene in Scene Container

	while v < number_scenes: # Adds Scenes to the Scene Container

		
		# Creates an empty Pointcloud Scene
		#pointcloudscene = np.empty((0, 3), dtype=float)

		folder = random.choice(os.listdir(DATADIR))
		path = DATADIR + folder
		file = next(os.walk(path))[2][0]
		path = path +"\\"+ file

		# Load Data to PyntCloud Object
		pointcloud = PyntCloud.from_file(path)

		# Convert to npy for manipulation
		npy_pointcloud = pointcloud.points.to_numpy()

		# Create pandas dateframe to convert x,y,z scene to pyntcloud Object
		p_x = npy_pointcloud[:, 0]
		p_y = npy_pointcloud[:, 1]
		p_z = npy_pointcloud[:, 2]

		df_pointcloud = pd.DataFrame(zip(p_x, p_y, p_z), columns=['x', 'y', 'z'])
		pyntcloud_scene = PyntCloud(df_pointcloud)

		# Get Number of Objects in the Scene
		category = npy_pointcloud[:, 3]
		category = np.unique(category) # counts only each unique number
		number_objects = category.shape[0]

		# ------------------------------------------ Voxilisation -----------------------------------------------------

		# create Voxelgrid from Pointcloudscene
		voxelgrid_id = pyntcloud_scene.add_structure("voxelgrid", n_x=voxel_size, n_y=voxel_size, n_z=voxel_size)
		voxelscene = pyntcloud_scene.structures[voxelgrid_id]
		
		# Create binary array from Voxelscene 
		binary_voxelscene = voxelscene.get_feature_vector(mode="binary")
	   
		# Create a Voxelscene Container of Scenes
		scene_container_x[v]=binary_voxelscene
		scene_container_y[v]=number_objects

		v = v + 1    

	v = 0
	print("Scene Creation done")
	
	
	#---------------------------------------- Prepare data for Network input -------------------------------------

	# X = datainput for Network
	X = np.zeros((number_scenes, voxel_size, voxel_size, voxel_size), dtype=float) 

	k = 0
	for key, value in scene_container_x.items():
		temp = [value]
		X[k] = np.concatenate(temp)
		k = k + 1

	# Y = label input for Network
	Y = []
	for key, value in scene_container_y.items():
		temp = [value]
		Y.append(temp)
	Y = np.asarray(Y)
	Y = tf.keras.utils.to_categorical(Y, 30)
	
	print("Data preparation done")

	# --------------------- Training the Network by incrementally call fit function -------------------------------

	history = LossHistory()
	model.fit(X, Y, batch_size=8, epochs=4, verbose=1, callbacks=[tensorboard, history])
	counter_trainingobjects = counter_trainingobjects + 1

	if (history.losses[0] < 0.001):
		model.save('savedmodel/scannet_model_192.h5')
		print("Model has been saved and trained on", counter_trainingobjects*32, "Voxelscenes")
            
    #except: continue
