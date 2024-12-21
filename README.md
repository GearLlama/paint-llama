# Description
The goal of this project is to create a robotic painting machine that can accept an input image and produce a physical painting.

This project relies heavily on the "Learning to Paint" project by Zhewei Huang, Wen Heng, Shuchang Zhou. The stroke pattern and simulation environment are modified to simulate the robotic painting machine.
Additionally, the code is refactored to match patterns preferred by the author. Links to original project below:

* [repo](https://github.com/hzwer/ICCV2019-LearningToPaint/tree/master)
* [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Learning_to_Paint_With_Model-Based_Deep_Reinforcement_Learning_ICCV_2019_paper.pdf)

# Initial Setup
```
./init.sh
```

# Update Project Structure
```
projen synth
```

# Update Dependencies
```
projen
```

# Renderer Commands
Run renderer tests
```
projen renderer:test
```

Start a tensorboard server to monitor a training job
```
projen renderer:monitor
```

Execute a training job
```
projen renderer:train
```