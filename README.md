# Music Classification using Deep Learning

### Google Colab
https://colab.research.google.com/drive/1W1MrTu8CpuKiB7d_h5ihwWT4PRvTzvr3?usp=sharing#scrollTo=zJi2TeZcDXQW

### Description
Music is a form of art that is ubiquitous and has a rich history. Different composers have created
music with their unique styles and compositions. However, identifying the composer of a
particular piece of music can be a challenging task, especially for novice musicians or listeners.
The proposed project aims to use deep learning techniques to identify the composer of a given
piece of music accurately.


Dataset
-----------------------------

The project used a dataset consisting of musical scores from various composers. The 
dataset contains MIDI files of compositions from well-known classical composers like Bach, 
Beethoven, Chopin, Mozart, Schubert, etc. (9 composers).


Data Cleaning, EDA, and Preprocessing
-----------------------------

Data was obtained through class resources and included three separate folders: “dev”, “test”, and
“train”. Each folder contains samples in midi format from nine various composers including Bach, Bartok, Byrd, Chopin, Handel, Hummel, Mendelssohn, Mozart, and Schumann. The training example directory has 369 sample files, the test directory 35 sample files, and the dev directory another 35 sample files. Using the “pretty_midi” library, the files were loaded into the Jupyter notebook. Most composers contributed approximately 40 samples each, suggesting a
balanced distribution within the dataset. To further understand the dataset, visualizations were
plotted. A distribution plot for the length of the midi files revealed that most samples were under
500 seconds, with a few extending beyond 1500 seconds. Tempo distribution appeared mostly
normal around 200 beats per minute, but displayed a slight skew to the left, indicating some
slower-paced sample

### Approach 1: 

The first approach involves the conversion of the raw midi files into wav files using
“midi2audio”. The features extracted included the following:
*  MFCC (Mel-frequency cepstral coefficients)
*  Mel Spectrogram
*  Chroma Vector
*  Tonnetz (Tonal centroid features)
The concatenated features were inputted into a Feedforward Neural Network, more specifically a
Multilayer Perceptron. The two hidden layers used batch normalization, ReLU activations,
dropout, “Adam” optimizer, early stopping and reduced learning rate callbacks to achieve
approximately 74.29% accuracy with a SparseCategoricalCrossentropy loss of 1.0977.

### Approach 2: 

The second approach involves a total of 12 features - extracted and engineered from raw MIDI
files using pretty_midi library . We tried deep Convolutional Neural Networks on the train
dataset and achieved a Test Accuracy: 0.7143 using AdamW optimizer and
SparseCategoricalCrossentropy loss function.
The feature list includes the following.
* The estimated tempo (beats per minute) of the MIDI file.
*  The number of time signature changes in the MIDI file.
*  The resolution of the MIDI file.
*  The numerator of the time signature (e.g., 4 in 4/4 time) of the MIDI file. If no time
signature changes are present, it defaults to 4.
*  The denominator of the time signature (e.g., 4 in 4/4 time) of the MIDI file. If no time
signature changes are present, it defaults to 4.
*  The number of different instruments used in the MIDI file.
*  The number of different instruments used in the MIDI file.
*  Note Distribution: Number of notes (total count of MIDI notes in the file).
*  Average note pitch (average MIDI note number).
*  Range of note pitches (highest MIDI note number - lowest MIDI note number).
*  Standard deviation of note pitches (measure of pitch variation). Note density (number of
notes per unit of time).

Model Design and Training
-----------------------------

### Approach 1:
The constructed model began with an input layer of shape 498, corresponding to the expected
number of features. The first layer is a densely connected neural network layer with 128 neurons,
L2 regularization of ‘0.001’, batch normalization, a ‘relu’ activation function, and a dropout
layer that nullifies 50% of the inputs during training to minimize overfitting. The second layer
mirrors the first but consists of 64 neurons. The output layer has 9 neurons, which aligns with the
classification of the nine composers, using a ‘softmax’ activation function. The model was
optimized with an “Adam” optimizer at a learning rate of ‘0.001’. Additionally, an early stopping
callback with a patience of 10 epochs and a learning rate reduction monitoring validation loss
plateaus (with a minimum of ‘0.0001’) were employed.

Training was initiated for 300 epochs, but early stopping trimmed this to 43 epochs completed in
12 steps or batches. A separate validation dataset was used during training to gauge the model's
ability to generalize. Performance metrics included sparse categorical accuracy.

### Approach 2:
The architecture combines convolutional layers for feature extraction from the input data,
followed by fully connected layers for high-level pattern recognition and decision-making.
Dropout layers help prevent overfitting, and a learning rate scheduler adjusts the learning rate
during training. This architecture is designed to capture relevant features in the input data and
make accurate predictions for the binary classification problem at hand.

We started our training with an initial learning rate of 0.01. Optimizer AdamW and
SparseCategoricalCrossentropy loss function was used with a learning rate decay in the callback
function and accuracy as the metric to be observed for 250 epochs with a batch size of 128.

Model Evaluation
-----------------------------

### Approach 1:
Performance metrics were drawn from the evaluate function using the test and label features. The
test accuracy stood at 74.29%, with a test loss of 1.098. To visually interpret the model's training
progression, learning curves depicting accuracy and loss were plotted. These curves revealed the
model’s commendable accuracy on the validation dataset, without any evident signs of
overfitting 

![Model Evaluation](https://github.com/tmcgirr/AAI-511-Music/assets/59525360/f0851121-2d01-4eed-9522-32e68b28229b)

### Approach 2:
We reached an accuracy of 97% on the training dataset and loss of 0.0776 and a Test Loss:
1.2994, Test Accuracy: 0.7143.

![model2](https://github.com/tmcgirr/AAI-511-Music/assets/59525360/095818b5-282b-4bfa-acc1-19ef606cda1b)
