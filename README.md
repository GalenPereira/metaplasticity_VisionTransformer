
# Metaplasticity in Binarized Vision Transformers to solve Catestrophic Forgetting in Continual Learning

My project is built on the work of Metaplasticity in Binarized Neural Networks from the [CVPR paper](https://arxiv.org/abs/2003.03533), where I explored the possibility to use metaplasticity in vision Transformers to see if it improves accuracy for the problem of catestropic forgetting while performing continual learning.

I binarized the Vision Trasformer using the functions BinarizeLinear and BinarizeConv2d form the code repository of the paper ([Link here](https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN)) and also used Pytorch is moduledict to maintain a dictionary of all the layers to plot the distribution of weights wherever requred to see how the network is behaving.

I have updated certain files in requirements.txt to work on any Linux Distro( Tried and Tested on Ubuntu, Debian, Arch, Gnome and Fedora)

# Installation
First Step like in any project set up a seperate environment
```bash
conda config --add channels conda-forge  
conda create --name environment_name --file requirements.txt  
conda activate environment_name  
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```



## Command Line Arguments
I changed the main.py to main_vit.py (I have still put the old file in the repo for referrance) of the paper to be more optimal as I was not using many of the command line arguments these are the arguments I have used 


`net` - which network you want to run option are vit, bvit and hybrid

`task-sequence` - how you want the dataset to load in which sequence

`lr` - the learning rate for the optimizer

`gamma` - for decay

`epochs-per-task` - no of runs

`decay` - decay rate

`device` - cuda or cpu



## Running Tests

After creation and activation of the environment run these commands in the same folder where all the files are located and in task sequence you can change the no of subsets to match the output layer of the model these are the all three models with different metaplastic coefficient.
```bash
python main.py  --net 'vit' --lr 0.00099 --decay 1e-7 --meta 0.0 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'vit' --lr 0.00099 --decay 1e-7 --meta 0.5 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'vit' --lr 0.00099 --decay 1e-7 --meta 1.0 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'vit' --lr 0.00099 --decay 1e-7 --meta 1.35 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
```

```bash
python main.py  --net 'bvit' --lr 0.00099 --decay 1e-7 --meta 0.0 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'bvit' --lr 0.00099 --decay 1e-7 --meta 0.5 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'bvit' --lr 0.00099 --decay 1e-7 --meta 1.0 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'bvit' --lr 0.00099 --decay 1e-7 --meta 1.35 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
```
```bash
python main.py  --net 'hybrid' --lr 0.00099 --decay 1e-7 --meta 0.0 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'hybrid' --lr 0.00099 --decay 1e-7 --meta 0.5 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'hybrid' --lr 0.00099 --decay 1e-7 --meta 1.0 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
python main.py  --net 'hybrid' --lr 0.00099 --decay 1e-7 --meta 1.35 --epochs-per-task 100 --task-sequence 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5' 'imgnet-5'
```