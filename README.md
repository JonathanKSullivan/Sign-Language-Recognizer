# Artificial Intelligence Engineer Nanodegree
## Probabilistic Models
### Project: Sign Language Recognition System
In this project, I built a system that can recognize words communicated using the American Sign Language (ASL). I was provided with a preprocessed dataset of tracked hand and nose positions extracted from video. My goal was to train a set of Hidden Markov Models (HMMs) using part of this dataset to try and identify individual words from test sequences.

As an optional challenge, I incorporated Statistical Language Models (SLMs) that capture the conditional probability of particular sequences of words occurring. This helps improve the recognition accuracy of the system.

## Getting Started

To get this code on your machine you can fork the repo or open a terminal and run this command.
```sh
git clone https://github.com/JonathanKSullivan/Sign-Language-Recognizer.git
```

### Prerequisites

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [jupyter](http://ipython.org/notebook.html)
- [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)

Notes: 
1. It is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python and load the environment included in the "Your conda env for AI ND" lesson.
2. The most recent development version of hmmlearn, 0.2.1, contains a bugfix related to the log function, which is used in this project.  In order to install this version of hmmearn, install it directly from its repo with the following command from within your activated Anaconda environment:
```sh
pip install git+https://github.com/hmmlearn/hmmlearn.git
```
I used pygame to help me visualize mu programs so that I have beautiful visualizations of AI I can share with others in your portfolio. However, pygame is optional as it can be tricky to install. 

### Installing
#### Mac OS X and Linux
1. Download the `aind-environment-unix.yml/aind-environment-unix.yml`/`aind-environment-osx.yml` file at the bottom of this section.
2. Run `conda env create -f aind-environment-unix.yml`(or `aind-environment-osx.yml`) to create the environment.
then source activate aind to enter the environment.
3. Install the development version of hmmlearn 0.2.1 with a source build: `pip install git+https://github.com/hmmlearn/hmmlearn.git`. 

#### Windows
1. Download the `aind-environment-windows.yml` file at the bottom of this section.
2. `conda env create -f aind-environment-windows.yml` to create the environment.
then activate aind to enter the environment.
3. Install the development version of hmmlearn 0.2.1 in one of the following ways. 
    ##### Source build
    1. Download the Visual C++ Build Tools [here](http://landinghub.visualstudio.com/visual-cpp-build-tools).
    `pip install git+https://github.com/hmmlearn/hmmlearn.git`
    ##### Precompiled binary wheel
    1. Download the appropriate `hmmlearn-0.2.1-yourpythonwindows.whl` file from here
    2. Install with `pip install hmmlearn-0.2.1-yourpythonwindows.whl`.

#### Optional: Install Pygame
I used pygame to help you visualize my programs so that I have beautiful visualizations of AI I can share with others in my portfolio. 
##### Mac OS X
1. Install [homebrew](http://brew.sh/)
2. `brew install sdl sdl_image sdl_mixer sdl_ttf portmidi mercurial`
3. `source activate aind`
4. `pip install pygame`
Some users have reported that pygame is not properly initialized on OSX until you also run `python -m pygame.tests`.

Windows and Linux
1. `pip install pygame`
2. In Windows, an alternate method is to install a precompiled binary wheel:
    1. Download the appropriate `pygame-1.9.3-yourpythonwindows.whl` file from here
    2. Install with `pip install pygame-1.9.3-yourpythonwindows.whl`.


Download the one of the following yml files.
[aind-environment-osx.yml](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58ee7e68_aind-environment-macos/aind-environment-macos.yml)
[aind-environment-unix.yml](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58ee7eff_aind-environment-unix/aind-environment-unix.yml)
[aind-environment-windows.yml](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58ee7f6c_aind-environment-windows/aind-environment-windows.yml)

## Running the tests

Test are included in notebook. To run test from terminal, navigate to project directory and run 
```sh
    asl_test.py
```

## Deployment
To run simply navigate to project directory and run 
```sh
    jupyter notebook asl_recognizer.ipynb
```

## Built With

* [Jupyter](http://www.http://jupyter.org/) - The Document Editor used
* [Anaconda](https://www.continuum.io/downloads) - The data science platform used
* [hmmlearn](https://github.com/hmmlearn/hmmlearn) - Python Hidden Markov Models API used 

## Authors
* **Udacity** - *Initial work* - [AIND-Recognizer](https://github.com/udacity/AIND-Recognizer)
* **Jonathan Sulivan** - *Build Model* -

## Acknowledgments
* Hackbright Academy
* Udacity
 
### Additional Information
##### Provided Raw Data

The data in the `asl_recognizer/data/` directory was derived from 
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php). 
The handpositions (`hand_condensed.csv`) are pulled directly from 
the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). The three markers are:

*   0  speaker's left hand
*   1  speaker's right hand
*   2  speaker's nose
*   X and Y values of the video frame increase left to right and top to bottom.

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/~dreuw/download/021.avi)
to see how the hand locations are tracked.

The videos are sentences with translations provided in the database.  
For purposes of this project, the sentences have been pre-segmented into words 
based on slow motion examination of the files.  
These segments are provided in the `train_words.csv` and `test_words.csv` files
in the form of start and end frames (inclusive).

The videos in the corpus include recordings from three different ASL speakers.
The mappings for the three speakers to video are included in the `speaker.csv` 
file.