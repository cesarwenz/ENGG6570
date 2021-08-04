
## Test trained GAN models 
#%%

from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from numpy.random import randn
import cv2
import numpy as np

# Overlay trajectories (data_y) to the image (data_x)
def create_trajectory(data_x, data_y, obs_len=10):
    # Calibration parameter to overlay for a 1280x360 resolution image
    K = np.array([[537.023764, 0, 640 , 0], 
                    [0 , 537.023764, 180, 0], 
                    [0, 0, 1, 0]])
    # Rotation matrix to obtain egocentric trajectory
    Rt = np.array([[0.028841, 0.007189, 0.999558, 1.481009],
                    [-0.999575,  0.004514,  0.028809,  0.296583],
                    [ 0.004305,  0.999964, -0.007316, -1.544537],
                    [ 0.      ,  0.      ,  0.      ,  1.      ]])

    # Resize data back to 1280x360
    data_x = cv2.resize(data_x, (1280,360))
    # Add column of ones for rotation matrix multiplication
    data_y = np.hstack((data_y, np.ones((len(data_y),1))))
    # Draw points
    for m in range(obs_len, data_y.shape[0]):
        # Rotation matrix multiplication of trajectory 
        A = np.matmul(np.linalg.inv(Rt), data_y[m, :].reshape(4, 1))
        # Egocentric view of trajectory
        B = np.matmul(K, A)
        # Circle location of trajectories 
        x = int(B[0, 0] * 1.0 / B[2, 0])
        y = int(B[1, 0] * 1.0 / B[2, 0])
        if (x < 0 or x > 1280 - 1 or y > 360 - 1):
            continue
        # Use opencv to overlay trajectories
        data_x = cv2.circle(data_x, (x, y), 3, (0, 0, 255), -1)
    return data_x

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_drvact_text(drvact):
    text = '[warnings] drvact label is not defined ...'
    if (drvact == 1):
        text = 'Go'
    elif (drvact == 2):
        text = 'Turn Left'
    elif (drvact == 3):
        text = 'Turn Right'
    elif (drvact == 4):
        text = 'U-turn'
    elif (drvact == 5):
        text = 'Left LC'
    elif (drvact == 6):
        text = 'Right LC'
    elif (drvact == 7):
        text = 'Avoidance'
    return text

# plot source, generated and target images
def plot_images(X_realA, X_fakeB, X_realB, filename, n_samples=1):
    X_realA = (X_realA + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']
    pyplot.figure(figsize=(32.0, 20.0))
    # plot real source images
    for i in range(n_samples):
        orig_image = (X_realA[i]* 255).astype(np.uint8)
        orig_image = cv2.resize(orig_image, (1280,360))
        pyplot.subplot(3, n_samples, 1 + i)
        # pyplot.axis('off')
        pyplot.imshow(orig_image)
        pyplot.title(titles[i])
    # plot generated target image
    pyplot.text(10,20, 'Driver Action: ' + generate_drvact_text(label+1), color='red', fontsize=12, fontweight='extra bold')
    for i in range(n_samples):
        fake_sample = create_trajectory((X_realA[i]* 255).astype(np.uint8), X_fakeB[i])
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(fake_sample)
        pyplot.title(titles[1 + i])
    # plot real target image
    for i in range(n_samples):
        true_sample = create_trajectory((X_realA[i]* 255).astype(np.uint8), X_realB[i])
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(true_sample)
        pyplot.title(titles[i + 2])
    # save plot to file
    pyplot.subplots_adjust(hspace=0.4)
    pyplot.show()
    pyplot.savefig(filename)
    
    # save the generator model

#%%

# load dataset

data = load('data/dual_condition_dataset_test.npz')
# unpack arrays
X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']
X3 = X3-1

#%%
# load model
model = load_model('model_056010.h5')

#%%

# select random example
latent_dim = 512
z_input = generate_latent_points(latent_dim, 1)
ix = int(randint(0, len(X1), 1))

src_image, tar_traj, label = X1[ix].reshape((1, 256, 256, 3)), X2[ix].reshape((1,40,3)), X3[ix].reshape(1)
# generate image from source
gen_traj = model.predict([src_image, z_input, label])
plot_images(src_image, gen_traj, tar_traj, 'sample20.png')

