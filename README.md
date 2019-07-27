# CNN for face-anti-spoofing
The face anti-spoofing is an technique that could prevent face-spoofing attack. For example, an intruder might use a photo of the legal user to "deceive" the face recognition system. Thus it is important to use the face anti-spoofint technique to enhance the security of the system.

All the face images listed below are in the dataset of CASIA-FASD

#Why MobileNet?
MobileNet is an architecture which is more suitable for mobile and embedded based vision applications where there is lack of compute power. This architecture was proposed by Google.

 
## Usage
### Main Dependencies
 ```
 Python 3 and above
 tensorflow 1.3.0
 numpy 1.13.1
 tqdm 4.15.0
 easydict 1.7
 matplotlib 2.0.2
 pillow 5.0.0
 ```
# Prepare Dataset
CASIA-FASD datasets are consist of videos, each of which is made of 100 to 200 video frames. For each video, I captured 30 frames (with the same interval between each frame). Then, with the Haar_classifier, I was able to crop a person’s face from an image. These images make up training datasets and test datasets. 

### Train and Test
1. Prepare your data, and modify the data_loader.py/DataLoader/load_data() method.
2. Modify the config/test.json to meet your needs.

Note: If you want to test that the model is pretrained and working properly, I've added some test images from different classes in directory 'data/'. All of them are classified correctly.

### Run
```
python3 main.py --config config/test.json

```
The file 'test.json' is just an example of a file. If you run it as is, it will test the model against the images in directory 'data/test_images'. You can create your own configuration file for training/testing.

## Benchmarking
In my implementation, I have achieved approximately 1140 MFLOPS. The paper counts multiplication+addition as one unit. My result verifies the paper as roughly dividing 1140 by 2 is equal to 569 unit.

To calculate the FLOPs in TensorFlow, make sure to set the batch size equal to 1, and execute the following line when the model is loaded into memory.
```
tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')
```
I've already implemented this function. It's called ```calculate_flops()``` in `utils.py`. Use it directly if you want.

## Updates
* Inference and training are working properly.
# Test

 <img src="https://i.ibb.co/G2784c7/test.png" alt="test" border="0">

 <img src="https://paperswithcode.com/media/tasks/facial-anti-spoofing_gHfingq.png" alt="facial-anti-spoofing" border="0">
