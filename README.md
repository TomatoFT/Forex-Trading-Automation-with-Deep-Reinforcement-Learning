# Forex Trading Automation with Deep Reinforcement Learning
<h1>How to run</h1>
You can use the .ipynb file to run the project 

### Clone the project
```
git clone https://github.com/TomatoFT/Forex-Trading-Automation-with-Deep-Reinforcement-Learning
cd Forex-Trading-Automation-with-Deep-Reinforcement-Learning
```

### Create the Anaconda Environment
```
conda create --name Forex
conda activate Forex
```

### Install some package and dependancies
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
conda install python=3.7 anaconda=custom
pip install -r requirements.txt
```

### Run Deep Reinforcement learning methods
```
python run_DRL.py
```

### Exit the Anaconda Environment
```
conda deactivate
```

<h1>Publication</h1>
<p>We use this project to have the publication in RIVF 2022 conference. For more about the methods or the implementation of the project you can read the paper with information below.</p>

```
@INPROCEEDINGS{10013861,
  author={Chau, Tan and Nguyen, Minh-Tri and Ngo, Duc-Vu and Nguyen, Anh-Duc T. and Do, Trong-Hop},
  booktitle={2022 RIVF International Conference on Computing and Communication Technologies (RIVF)}, 
  title={Deep Reinforcement Learning methods for Automation Forex Trading}, 
  year={2022},
  volume={},
  number={},
  pages={671-676},
  doi={10.1109/RIVF55975.2022.10013861}}
```

 <h1>NOTICE</h1>
 <p>In Janary 9 2023, I known that Google Colab had suspended the Tensorflow 1.x. So the baseline on Colab file is not running anyways. To run this, you can use command in the baseline to deploy on other place (I recommend you to use Docker). Or you can update the whole codes to transform from tensorflow 1 to tensorflow 2. If you want to have any contributed, you can make an request and I will appreciate if you do that<p>
 
 <p>Feel free to clone my code and I will appreciate if you improve it</p>

