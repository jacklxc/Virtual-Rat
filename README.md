# Virtual Rat

##### Virtual Rat project is a computaional neuroscience project  conducted by Xiangci Li as a research assistant at Erlich lab under the guidance of New York University Shanghai Professor Jeffrey Erlich. We showed that the task switch cost of rat's ProAnti orienting task can be modeled by an Elman Recurrent Neural Network. The experimental data are from Dr. Carlos Brody's lab at Princeton.

For any question, please contact

##### Xiangci Li (xiangci.li@nyu.edu)

##### Jeffrey Erlich (jerlich@nyu.edu)

### Briefly about the implementation

From the perspective of implementation, Virtual Rat project consists of two parts: Virtual Rat (RNN model) and data analysis of rats' behavioral and electrophysiology data.

### How to reproduce all results:

## Environment setup:

### Python:

1. Visit https://github.com/dmlc/minpy to setup Minpy and MXNet.
2. Other requirements: Python 2.7, Numpy, Matplotlib, Scikit-learn, Scipy

### MATLAB:

MATLAB 2017a or later.

## Note:

The codes for the final version of this project are all under `publication`  folder.

`figures`  folder saves figures produced. `mats` folder saves `.mat` files and `pkls` folder  saves `.pkl` files.

## Procedures:

### Virtual Rat part:

#### 1. Experiments for the most basic RNNs:

1. Run `sbatch trainingTime.sh` on HPC to evoke `TrainingTime.py`. All weights after certain training epochs will be saved.
2. Run `sbatch testTrainingTime.sh` on HPC to evoke `TestTrainingTime.py`. Similarly run `sbatch testTrainingTimeFine.sh` to evoke `TestTrainingTimeFine.py`. The weights saved will be loaded to test the performance of the models and the results will be saved in the folder `TrainingTime`.
3. Run jupyter notebook `TrainingTime.ipynb`. This notebook finds out best epochs to plot later figures.
4. Run jupyter notebook `VirtualRat.ipynb` and `DiluteActivation.ipynb`to obtain figures. 
5. Run jupyter notebook `PETH.ipynb` to choose sample PETH.

#### 2. Experiments for varying block sizes:

1. Run `sbatch trainBlock.sh` on HPC to evoke `BlockTime.py`. All weights after certain training epochs will be saved.
2. Run `sbatch testBlockTime.sh` on HPC to evoke `TestBlockTime.py`.  The weights saved will be loaded to test the performance of the models and the results will be saved in the folder `BlockTime`.
3. Run jupyter notebook `Block.ipynb`.

#### 3. Experiments for varying pro to anti ratio:

1. Run `sbatch RatioTime.sh` on HPC to evoke `TrainRatioTime.py`. All weights after certain training epochs will be saved.
2. Run `sbatch testTrainingTimeRatio.sh` on HPC to evoke `TestTrainingTimeRatio.py`.  The weights saved will be loaded to test the performance of the models and the results will be saved in the folder `RatioTime`.
3. Run jupyter notebook `TrainingTimeRatio.ipynb` to check the performace over time.
4. Run jupyter notebook `Ratio.ipynb`.



### Data analysis part (MATLAB & Python):

1. Run `countSpikes.m` to produce spike counts.
2. Run `convertBehavior.m` to extract necessary information from the original huge data file for running Python scripts.
3. Run `realRatEphysTrainClfs.py` and `realRatEphysTrainClfsTarget.py`
4. Run `ratSGDrule.m`,  `ratSGDTarget.m`,  `RNN_SGD_rule.m` and `RNN_SGD_target.m`.
5. Run jupyter notebook `realRatEphys.ipynb`.
   . Run jupyter notebook `realRatEphysTarget.ipynb`.	
6. Run jupyter notebook `realRatBehavior.ipynb`.

