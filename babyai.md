(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: README.md
================================================
[![Build Status](https://travis-ci.org/mila-iqia/babyai.svg?branch=master)](https://travis-ci.org/mila-iqia/babyai)

# 2023 update

All BabyAI environments are now part of the [Minigrid library](https://github.com/Farama-Foundation/Minigrid). **This repository is not actively maintained.**

Training RL agents on Minigrid (and BabyAI) environments can be done using [this repository](https://github.com/lcswillems/rl-starter-files).

This repository still contains scripts which, if adapted to the Minigrid library, could be used to:
- Produce demonstrations using the [BabyAI bot](babyai/bot.py),
- [Train Imitation Learning agents](babyai/imitation.py) using the bot-generated demonstrations as training trajectories.

# BabyAI 1.1


BabyAI is a platform used to study the sample efficiency of grounded language acquisition, created at [Mila](https://mila.quebec/en/).

The master branch of this repository is updated frequently.  If you are looking to replicate or compare against the [baseline results](http://arxiv.org/abs/2007.12770), we recommend you use the [BabyAI 1.1 branch](https://github.com/mila-iqia/babyai/tree/dyth-v1.1-and-baselines) and cite both:

```
@misc{hui2020babyai,
    title={BabyAI 1.1},
    author={David Yu-Tung Hui and Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Yoshua Bengio},
    year={2020},
    eprint={2007.12770},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

and the [ICLR19 paper](https://openreview.net/forum?id=rJeXCo0cYX), which details the experimental setup and BabyAI 1.0 baseline results.  Its source code is in the [iclr19 branch](https://github.com/mila-iqia/babyai/tree/iclr19):

```
@inproceedings{
  babyai_iclr19,
  title={Baby{AI}: First Steps Towards Grounded Language Learning With a Human In the Loop},
  author={Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Salem Lahlou and Lucas Willems and Chitwan Saharia and Thien Huu Nguyen and Yoshua Bengio},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=rJeXCo0cYX},
}
```

This README covers instructions for [installation](##installation) and [troubleshooting](##troubleshooting).  Other instructions are:

- [Instructions on how to contribute](CONTRIBUTING.md)
- [Codebase Structure](babyai/README.md)
- [Training, Evaluation and Reproducing Baseline Results](scripts/README.md)
- [BabyAI 1.0+ levels](docs/iclr19_levels.md) and [older levels](docs/bonus_levels.md).

## Installation

### Conda (Recommended)

If you are using conda, you can create a `babyai` environment with all the dependencies by running:

```
git clone https://github.com/mila-iqia/babyai.git
cd babyai
conda env create -f environment.yaml
source activate babyai
```

After that, execute the following commands to setup the environment.

```
cd ..
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip install --editable .
```

The last command installs the repository in editable mode. Move back to the `babyai` repository and install that in editable mode as well.

```
cd ../babyai
pip install --editable .
```

Finally, [follow these instructions](###babyai-storage-path)

### Manual Installation

Requirements:
- Python 3.6+
- OpenAI Gym
- NumPy
- PyTorch 0.4.1+
- blosc

First install [PyTorch](http://pytorch.org/) for on your platform.

Then, clone this repository and install the other dependencies with `pip3`:

```
git clone https://github.com/mila-iqia/babyai.git
cd babyai
pip3 install --editable .
```

Finally, [follow these instructions](###babyai-storage-path)

### BabyAI Storage Path

Add this line to `.bashrc` (Linux), or `.bash_profile` (Mac).

```
export BABYAI_STORAGE='/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>'
```

where `/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>` is the folder where you typed `git clone https://github.com/mila-iqia/babyai.git` earlier.

Models, logs and demos will be produced in this directory, in the folders `models`, `logs` and `demos` respectively.

### Downloading the demos

These can be [downloaded here](https://drive.google.com/file/d/1NeJX8ZCUEnhwO1rmefqkMEizhWxyQLEX/view?usp=sharing)

Ensure the downloaded file has the following md5 checksum (obtained via `md5sum`): `1df202ef2bbf2de768633059ed8db64c`

before extraction:
```
gunzip -c copydemos.tar.gz | tar xvf -
```


**Using the `pixels` architecture does not work with imitation learning**, because the demonstrations were not generated to use pixels.


## Troubleshooting

If you run into error messages relating to OpenAI gym, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/mila-iqia/babyai/issues/new) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a Mila machine?).

### Pixel Observations

Please note that the default observation format is a partially observable view of the environment using a compact encoding, with 3 input values per visible grid cell, 7x7x3 values total. These values are **not pixels**. If you want to obtain an array of RGB pixels as observations instead, use the `RGBImgPartialObsWrapper`. You can use it as follows:

```
import babyai
from gym_minigrid.wrappers import *
env = gym.make('BabyAI-GoToRedBall-v0')
env = RGBImgPartialObsWrapper(env)
```

This wrapper, as well as other wrappers to change the observation format can be [found here](https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/wrappers.py).



================================================
FILE: CONTRIBUTING.md
================================================
# Instructions for Contributors

To contribute to this project, you should first create your own fork, and remember to periodically [sync changes from this repository](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository). You can then create [pull requests](https://yangsu.github.io/pull-request-tutorial/) for modifications you have made. Your changes will be tested and reviewed before they are merged into this repository. If you are not familiar with forks and pull requests, we recommend doing a Google or YouTube search to find many useful tutorials on the topic.

Also, you can have a look at the [codebase structure](docs/codebase.md) before getting started.

A suggested flow for contributing would be:
First, open up a new feature branch to solve an existing bug/issue
```bash
$ git checkout -b <feature-branch> upstream/master
```
This ensures that the branch is up-to-date with the `master` branch of the main repository, irrespective of the status of your forked repository.

Once you are done making commits of your changes / adding the feature, you can:
(In case this is the first set of commits from this _new_ local branch)
```bash
git push --set-upstream origin 
```
(Assuming the name of your forked repository remote is `origin`), which will create a new branch `<feature-branch>`
tracking your local `<feature-branch>`, in case it hasn't been created already.

Then, create a [pull request](https://help.github.com/en/articles/about-pull-requests) in this repository.


================================================
FILE: Dockerfile
================================================
# To build this docker image:
# sudo docker build .
#
# To run the image:
# sudo nvidia-docker run -it <image_id>

# Note: a more recent nvidia/cuda image means users must
# have a more recent CUDA install on their systems
#FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y qt5-default qttools5-dev-tools git python3-pip
RUN pip3 install --upgrade pip

# Clone and install BabyAI git repo
RUN git clone https://github.com/mila-udem/babyai.git
WORKDIR babyai
RUN pip3 install --editable .

# Copy models into the docker image
COPY models models/



================================================
FILE: environment.yaml
================================================
name: babyai
channels:
    - pytorch
    - defaults
dependencies:
    - python=3.6
    - pytorch=1.4
    - numpy
    - blosc
    - pip
    - pip:
        - gym
        - scikit-build



================================================
FILE: LICENSE
================================================
BSD 3-Clause License

Copyright (c) 2017, Maxime Chevalier-Boisvert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



================================================
FILE: run_tests.py
================================================
#!/usr/bin/env python3

"""
Run basic BabyAI level tests
Note: there are other automated tests in .circleci/config.yml
"""

import babyai
from babyai import levels

# NOTE: please make sure that tests are always deterministic

print('Testing levels, mission generation')
levels.test()



================================================
FILE: setup.py
================================================
from setuptools import setup

setup(
    name="babyai",
    version="1.1.2",
    license="BSD 3-clause",
    keywords="memory, environment, agent, rl, openaigym, openai-gym, gym",
    packages=["babyai", "babyai.levels", "babyai.utils"],
    install_requires=[
        "gym>=0.9.6",
        "numpy>=1.17.0",
        "torch>=0.4.1",
        "blosc>=1.5.1",
        "gym_minigrid>=1.2.0",
    ],
)



================================================
FILE: .travis.yml
================================================
language: python
cache: pip
python:
    - "3.6"

before_install:
    - pip3 install --upgrade pip

# command to install dependencies
install:
    - pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    - pip3 install flake8
    - pip3 install scikit-build
    - pip3 install --editable .

# command to run tests
script:
    # Check the source code for obvious errors
    - python3 -m flake8 . --count --show-source --statistics --select=E901,E999,F821,F822,F823

    # Test the BabyAI levels
    - ./run_tests.py

    # Quickly exercise the RL training code
    - time python3 -m scripts.train_rl --env BabyAI-GoToObj-v0 --algo ppo --procs 4 --batch-size 80 --log-interval 1 --save-interval 2 --val-episodes 10 --frames 300 --arch original --instr-dim 16 --image-dim 16 --memory-dim 16

    # Check that the bot works on a few episodes of Boss Level
    - python3 -m scripts.eval_bot --level BossLevel --num_runs 50
    - python3 -m scripts.eval_bot --level BossLevel --num_runs 50 --advise_mode --non_optimal_steps 100 --bad_action_proba .3
    # Check that the bot works on a single episode from each level
    - python3 -m scripts.eval_bot --num_runs 1

    # Quickly test the generation of bot demos
    - python3 -m scripts.make_agent_demos --env BabyAI-GoToRedBallGrey-v0 --episodes 100 --valid-episodes 32

    # Quickly test the evaluation of bot demos
    - python3 -m scripts.evaluate --env BabyAI-GoToRedBallGrey-v0 --demos BabyAI-GoToRedBallGrey-v0_agent

    # Quick test for imitation learning
    - python3 -m scripts.train_il --env BabyAI-GoToRedBallGrey-v0 --demos BabyAI-GoToRedBallGrey-v0_agent --model GoToRedBallGrey-il --val-interval 1 --patience 0 --episodes 100 --val-episodes 50

    # Quickly test the evaluation of models
    - python3 -m scripts.evaluate --env BabyAI-GoToRedBallGrey-v0 --model GoToRedBallGrey-il

    # Quick test for imitation learning with multi env
    - python3 -m scripts.train_il --multi-env BabyAI-GoToRedBall-v0 BabyAI-GoToRedBallGrey-v0 --multi-demos BabyAI-GoToRedBallGrey-v0_agent BabyAI-GoToRedBallGrey-v0_agent --val-interval 1 --patience 0 --multi-episodes 100 100 --val-episodes 50

    # Quick test for train_intelligent_expert
    - python3 -m scripts.train_intelligent_expert --env BabyAI-GoToRedBallGrey-v0 --demos BabyAI-GoToRedBallGrey-v0_agent --val-interval 1 --patience 0 --val-episodes 50 --start-demos 10 --num-eval-demos 5 --phases 2



================================================
FILE: babyai/README.md
================================================
# BabyAI

There are three folders and eight other files

## Folders

- `levels` contains the code for all levels
- `rl` contains an implementation of the Proximal Policy Optimization (PPO) RL algorithm
- `utils` contains files for reading and saving logs, demos and models.  In this folder, `agent.py` defines an abstract class for an agent

## Files

- `arguments.py` contains the value of default arguments shared by both imitation and reinforcement learning
- `bot.py` is a heuristic stack-based bot that can solve all levels
- `efficiency.py` contains hyperparmeter configurations we use for imitation learning sample efficiency
- `evaluate.py` contains functions used by IL and RL to evaluate an agent
- `imitation.py` is our imitation learning implementation
- `model.py` contains the neural network code
- `plotting.py` is used in plotting.  It also contains Gaussian Process code used in measuring imitation learning sample efficiency



================================================
FILE: babyai/__init__.py
================================================
# Import levels so that the OpenAI Gym environments get registered
# when the babyai package is imported
from . import levels
from . import utils
import warnings


warnings.warn(
    "This code base is no longer maintained and is not expected to be maintained again. \n"
    "These environments are now maintained within Minigrid"
    "(see https://github.com/Farama-Foundation/Minigrid/tree/master/minigrid/envs/babyai). \n"
    "The maintained version includes documentation, support for current versions of Python, \n"
    "numerous bug fixes, support for installation via pip, and many other quality-of-life improvements. \n"
    "We encourage researchers to switch to the maintained version for all purposes other than comparing \n"
    "with results that use this version of the environments. \n"
)



================================================
FILE: babyai/arguments.py
================================================
"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default=None,
                            help="name of the environment to train on (REQUIRED)")
        self.add_argument("--model", default=None,
                            help="name of the model (default: ENV_ALGO_TIME)")
        self.add_argument("--pretrained-model", default=None,
                            help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
        self.add_argument("--seed", type=int, default=1,
                            help="random seed; if 0, a random random seed will be used  (default: 1)")
        self.add_argument("--task-id-seed", action='store_true',
                            help="use the task id within a Slurm job array as the seed")
        self.add_argument("--procs", type=int, default=64,
                            help="number of processes (default: 64)")
        self.add_argument("--tb", action="store_true", default=False,
                            help="log into Tensorboard")

        # Training arguments
        self.add_argument("--log-interval", type=int, default=10,
                            help="number of updates between two logs (default: 10)")
        self.add_argument("--frames", type=int, default=int(9e10),
                            help="number of frames of training (default: 9e10)")
        self.add_argument("--patience", type=int, default=100,
                            help="patience for early stopping (default: 100)")
        self.add_argument("--epochs", type=int, default=1000000,
                            help="maximum number of epochs")
        self.add_argument("--epoch-length", type=int, default=0,
                            help="number of examples per epoch; the whole dataset is used by if 0")
        self.add_argument("--frames-per-proc", type=int, default=40,
                            help="number of frames per process before update (default: 40)")
        self.add_argument("--lr", type=float, default=1e-4,
                            help="learning rate (default: 1e-4)")
        self.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
        self.add_argument("--recurrence", type=int, default=20,
                            help="number of timesteps gradient is backpropagated (default: 20)")
        self.add_argument("--optim-eps", type=float, default=1e-5,
                            help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--optim-alpha", type=float, default=0.99,
                            help="RMSprop optimizer apha (default: 0.99)")
        self.add_argument("--batch-size", type=int, default=1280,
                                help="batch size for PPO (default: 1280)")
        self.add_argument("--entropy-coef", type=float, default=0.01,
                            help="entropy term coefficient (default: 0.01)")

        # Model parameters
        self.add_argument("--image-dim", type=int, default=128,
                            help="dimensionality of the image embedding.  Defaults to 128 in residual architectures")
        self.add_argument("--memory-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--instr-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--no-instr", action="store_true", default=False,
                            help="don't use instructions in the model")
        self.add_argument("--instr-arch", default="gru",
                            help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
        self.add_argument("--no-mem", action="store_true", default=False,
                            help="don't use memory in the model")
        self.add_argument("--arch", default='bow_endpool_res',
                            help="image embedding architecture")

        # Validation parameters
        self.add_argument("--val-seed", type=int, default=int(1e9),
                            help="seed for environment used for validation (default: 1e9)")
        self.add_argument("--val-interval", type=int, default=1,
                            help="number of epochs between two validation checks (default: 1)")
        self.add_argument("--val-episodes", type=int, default=500,
                            help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")

    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args()

        # Set seed for all randomness sources
        if args.seed == 0:
            args.seed = np.random.randint(10000)
        if args.task_id_seed:
            args.seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
            print('set seed to {}'.format(args.seed))

        # TODO: more validation

        return args



================================================
FILE: babyai/bot.py
================================================
from gym_minigrid.minigrid import *
from babyai.levels.verifier import *
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                                    GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)


class DisappearedBoxError(Exception):
    """
    Error that's thrown when a box is opened.
    We make the assumption that the bot cannot accomplish the mission when it happens.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def manhattan_distance(pos, target):
    return np.abs(target[0] - pos[0]) + np.abs(target[1] - pos[1])


class Subgoal:
    """The base class for all possible Bot subgoals.

    Parameters:
    ----------
    bot : Bot
        The bot whose subgoal this is.
    datum : object
        The first parameter of the subgoal, e.g. a location or an object description.
    reason : str
        Why this subgoal was created. Subgoals created for different reasons require
        similar but different behaviour.

    """

    def __init__(self, bot=None, datum=None, reason=None):
        self.bot = bot
        self.datum = datum
        self.reason = reason

        self.update_agent_attributes()

        self.actions = self.bot.mission.actions

    def __repr__(self):
        """Mainly for debugging purposes"""
        representation = '('
        representation += type(self).__name__
        if self.datum is not None:
            representation += ': {}'.format(self.datum)
        if self.reason is not None:
            representation += ', reason: {}'.format(self.reason)
        representation += ')'
        return representation

    def update_agent_attributes(self):
        """Should be called at each step before the replanning methods."""
        self.pos = self.bot.mission.agent_pos
        self.dir_vec = self.bot.mission.dir_vec
        self.right_vec = self.bot.mission.right_vec
        self.fwd_pos = self.pos + self.dir_vec
        self.fwd_cell = self.bot.mission.grid.get(*self.fwd_pos)
        self.carrying = self.bot.mission.carrying

    def replan_before_action(self):
        """Change the plan if needed and return a suggested action.

        This method is called at every iteration for the top-most subgoal
        from the stack. It is supposed to return a suggested action if
        it is clear how to proceed towards achieving the current subgoal.
        If the subgoal is already achieved, or if it is not clear how it
        can be achieved, or if is clear that a better plan exists,
        this method can replan by pushing new subgoals
        from the stack or popping the top one.

        Returns:
        -------
        action : object
            A suggection action if known, `None` the stack has been altered
            and further replanning is required.

        """
        raise NotImplementedError()


    def replan_after_action(self, action_taken):
        """Change the plan when the taken action is known.

        The action actually taken by the agent can be different from the one
        suggested by `replan_before_action` is the bot can be used in
        advising mode. This method is supposed to adjust the plan in the view
        of the actual action taken.

        """
        pass


    def is_exploratory(self):
        """Whether the subgoal is exploratory or not.

        Exploratory subgoals can be removed from the stack by the bot, e.g.
        when no more exploration is required.

        """
        return False

    def _plan_undo_action(self, action_taken):
        """Plan how to undo the taken action."""
        if action_taken == self.actions.forward:
            # check if the 'forward' action was succesful
            if not np.array_equal(self.bot.prev_agent_pos, self.pos):
                self.bot.stack.append(GoNextToSubgoal(self.bot, self.pos))
        elif action_taken == self.actions.left:
            old_fwd_pos = self.pos + self.right_vec
            self.bot.stack.append(GoNextToSubgoal(self.bot, self.pos + self.right_vec))
        elif action_taken == self.actions.right:
            old_fwd_pos = self.pos - self.right_vec
            self.bot.stack.append(GoNextToSubgoal(self.bot, self.pos - self.right_vec))
        elif action_taken == self.actions.drop and self.bot.prev_carrying != self.carrying:
            # get that thing back, if dropping was succesful
            assert self.fwd_cell.type in ('key', 'box', 'ball')
            self.bot.stack.append(PickupSubgoal(self.bot))
        elif action_taken == self.actions.pickup and self.bot.prev_carrying != self.carrying:
            # drop that thing where you found it
            fwd_cell = self.bot.mission.grid.get(*self.fwd_pos)
            self.bot.stack.append(DropSubgoal(self.bot))
        elif action_taken == self.actions.toggle:
            # if you opened or closed a door, bring it back in the original state
            fwd_cell = self.bot.mission.grid.get(*self.fwd_pos)
            if (fwd_cell and fwd_cell.type == 'door'
                    and self.bot.fwd_door_was_open != fwd_cell.is_open):
                self.bot.stack.append(CloseSubgoal(self.bot)
                                      if fwd_cell.is_open
                                      else OpenSubgoal(self.bot))


class CloseSubgoal(Subgoal):

    def replan_before_action(self):
        assert self.fwd_cell is not None, 'Forward cell is empty'
        assert self.fwd_cell.type == 'door', 'Forward cell has to be a door'
        assert self.fwd_cell.is_open, 'Forward door must be open'
        return self.actions.toggle

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.toggle:
            self.bot.stack.pop()
        elif action_taken in [self.actions.forward, self.actions.left, self.actions.right]:
            self._plan_undo_action(action_taken)


class OpenSubgoal(Subgoal):
    """Subgoal for opening doors.

    Parameters:
    ----------
    reason : str
        `None`, `"Unlock"`, or `"UnlockAndKeepKey"`. If the reason is `"Unlock"`,
        the agent will plan dropping the key somewhere after it opens the door
        (see `replan_after_action`). When the agent faces the door, and the
        reason is `None`, this subgoals replaces itself with a similar one,
        but with with the reason `"Unlock"`. `reason="UnlockAndKeepKey` means
        that the agent should not schedule the dropping of the key
        when it faces a locked door, and should instead keep the key.

    """

    def replan_before_action(self):
        assert self.fwd_cell is not None, 'Forward cell is empty'
        assert self.fwd_cell.type == 'door', 'Forward cell has to be a door'

        # If the door is locked, go find the key and then return
        # TODO: do we really need to be in front of the locked door
        # to realize that we need the key for it ?
        got_the_key = (self.carrying and self.carrying.type == 'key'
            and self.carrying.color == self.fwd_cell.color)
        if (self.fwd_cell.is_locked and not got_the_key):
            # Find the key
            key_desc = ObjDesc('key', self.fwd_cell.color)
            key_desc.find_matching_objs(self.bot.mission)

            # If we're already carrying something
            if self.carrying:
                self.bot.stack.pop()

                # Find a location to drop what we're already carrying
                drop_pos_cur = self.bot._find_drop_pos()

                # Take back the object being carried
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))

                # Go back to the door and open it
                self.bot.stack.append(OpenSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))

                # Go to the key and pick it up
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))

                # Drop the object being carried
                self.bot.stack.append(DropSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))
            else:
                # This branch is will be used very rarely, given that
                # GoNextToSubGoal(..., reason='Open') should plan
                # going to the key before we get to stand right in front of a door.
                # But the agent can be spawned right in front of a open door,
                # for which we case we do need this code.

                self.bot.stack.pop()

                # Go back to the door and open it
                self.bot.stack.append(OpenSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))

                # Go to the key and pick it up
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))
            return

        if self.fwd_cell.is_open:
            self.bot.stack.append(CloseSubgoal(self.bot))
            return

        if self.fwd_cell.is_locked and self.reason is None:
            self.bot.stack.pop()
            self.bot.stack.append(OpenSubgoal(self.bot, reason='Unlock'))
            return

        return self.actions.toggle

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.toggle:
            self.bot.stack.pop()
            if self.reason == 'Unlock':
                # The reason why this has to be planned after the action is taken
                # is because if the position for dropping is chosen in advance,
                # then by the time the key is dropped there, it might already
                # be occupied.
                drop_key_pos = self.bot._find_drop_pos()
                self.bot.stack.append(DropSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_key_pos))
        else:
            self._plan_undo_action(action_taken)


class DropSubgoal(Subgoal):

    def replan_before_action(self):
        assert self.bot.mission.carrying
        assert not self.fwd_cell
        return self.actions.drop

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.drop:
            self.bot.stack.pop()
        elif action_taken in [self.actions.forward, self.actions.left, self.actions.right]:
            self._plan_undo_action(action_taken)


class PickupSubgoal(Subgoal):

    def replan_before_action(self):
        assert not self.bot.mission.carrying
        return self.actions.pickup

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.pickup:
            self.bot.stack.pop()
        elif action_taken in [self.actions.left, self.actions.right]:
            self._plan_undo_action(action_taken)


class GoNextToSubgoal(Subgoal):
    """The subgoal for going next to objects or positions.

    Parameters:
    ----------
    datum : (int, int) tuple or `ObjDesc` or object reference
        The position or the decription of the object or
        the object to which we are going.
    reason : str
        One of the following:
        - `None`: go the position (object) and face it
        - `"PutNext"`: go face an empty position next to the object specified by `datum`
        - `"Explore"`: going to a position, just like when the reason is `None`. The only
          difference is that with this reason the subgoal will be considered
          exploratory

    """

    def replan_before_action(self):
        target_obj = None
        if isinstance(self.datum, ObjDesc):
            target_obj, target_pos = self.bot._find_obj_pos(self.datum, self.reason == 'PutNext')
            if not target_pos:
                # No path found -> Explore the world
                self.bot.stack.append(ExploreSubgoal(self.bot))
                return
        elif isinstance(self.datum, WorldObj):
            target_obj = self.datum
            target_pos = target_obj.cur_pos
        else:
            target_pos = tuple(self.datum)

        # Suppore we are walking towards the door that we would like to open,
        # it is locked, and we don't have the key. What do we do? If we are carrying
        # something, it makes to just continue, as we still need to bring this object
        # close to the door. If we are not carrying anything though, then it makes
        # sense to change the plan and go straight for the required key.
        if (self.reason == 'Open'
                and target_obj and target_obj.type == 'door' and target_obj.is_locked):
            key_desc = ObjDesc('key', target_obj.color)
            key_desc.find_matching_objs(self.bot.mission)
            if not self.carrying:
                # No we need to commit to going to this particular door
                self.bot.stack.pop()
                self.bot.stack.append(GoNextToSubgoal(self.bot, target_obj, reason='Open'))
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))
                return

        # The position we are on is the one we should go next to
        # -> Move away from it
        if manhattan_distance(target_pos, self.pos) == (1 if self.reason == 'PutNext' else 0):
            def steppable(cell):
                return cell is None or (cell.type == 'door' and cell.is_open)
            if steppable(self.fwd_cell):
                return self.actions.forward
            if steppable(self.bot.mission.grid.get(*(self.pos + self.right_vec))):
                return self.actions.right
            if steppable(self.bot.mission.grid.get(*(self.pos - self.right_vec))):
                return self.actions.left
            # Spin and hope for the best
            return self.actions.left

        # We are facing the target cell
        # -> subgoal completed
        if self.reason == 'PutNext':
            if manhattan_distance(target_pos, self.fwd_pos) == 1:
                if self.fwd_cell is None:
                    self.bot.stack.pop()
                    return
                if self.fwd_cell.type == 'door' and self.fwd_cell.is_open:
                    # We can't drop an object in the cell where the door is.
                    # Instead, we add a subgoal on the stack that will force
                    # the bot to move the target object.
                    self.bot.stack.append(GoNextToSubgoal(
                        self.bot, self.fwd_pos + 2 * self.dir_vec))
                    return
        else:
            if np.array_equal(target_pos, self.fwd_pos):
                self.bot.stack.pop()
                return

        # We are still far from the target
        # -> try to find a non-blocker path
        path, _, _ = self.bot._shortest_path(
            lambda pos, cell: pos == target_pos,
        )

        # No non-blocker path found and
        # reexploration within the room is not allowed or there is nothing to explore
        # -> Look for blocker paths
        if not path:
            path, _, _ = self.bot._shortest_path(
                lambda pos, cell: pos == target_pos,
                try_with_blockers=True
            )

        # No path found
        # -> explore the world
        if not path:
            self.bot.stack.append(ExploreSubgoal(self.bot))
            return

        # So there is a path (blocker, or non-blockers)
        # -> try following it
        next_cell = path[0]

        # Choose the action in the case when the forward cell
        # is the one we should go next to
        if np.array_equal(next_cell, self.fwd_pos):
            if self.fwd_cell:
                if self.fwd_cell.type == 'door':
                    assert not self.fwd_cell.is_locked
                    if not self.fwd_cell.is_open:
                        self.bot.stack.append(OpenSubgoal(self.bot))
                        return
                    else:
                        return self.actions.forward
                if self.carrying:
                    drop_pos_cur = self.bot._find_drop_pos()
                    drop_pos_block = self.bot._find_drop_pos(drop_pos_cur)
                    # Take back the object being carried
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))

                    # Pick up the blocking object and drop it
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_block))
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, self.fwd_pos))

                    # Drop the object being carried
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))
                    return
                else:
                    drop_pos = self.bot._find_drop_pos()
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos))
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    return
            else:
                return self.actions.forward

        # The forward cell is not the one we should go to
        # -> turn towards the direction we need to go
        if np.array_equal(next_cell - self.pos, self.right_vec):
            return self.actions.right
        elif np.array_equal(next_cell - self.pos, -self.right_vec):
            return self.actions.left

        # If we reacher this point in the code,  then the cell is behind us.
        # Instead of choosing left or right randomly,
        # let's do something that might be useful:
        # Because when we're GoingNextTo for the purpose of exploring,
        # things might change while on the way to the position we're going to, we should
        # pick this right or left wisely.
        # The simplest thing we should do is: pick the one
        # that doesn't lead you to face a non empty cell.
        # One better thing would be to go to the direction
        # where the closest wall/door is the furthest
        distance_right = self.bot._closest_wall_or_door_given_dir(self.pos, self.right_vec)
        distance_left = self.bot._closest_wall_or_door_given_dir(self.pos, -self.right_vec)
        if distance_left > distance_right:
            return self.actions.left
        return self.actions.right

    def replan_after_action(self, action_taken):
        if action_taken in [self.actions.pickup, self.actions.drop, self.actions.toggle]:
            self._plan_undo_action(action_taken)

    def is_exploratory(self):
        return self.reason == 'Explore'


class ExploreSubgoal(Subgoal):
    def replan_before_action(self):
        # Find the closest unseen position
        _, unseen_pos, with_blockers = self.bot._shortest_path(
            lambda pos, cell: not self.bot.vis_mask[pos],
            try_with_blockers=True
        )

        if unseen_pos:
            self.bot.stack.append(GoNextToSubgoal(self.bot, unseen_pos, reason='Explore'))
            return None

        # Find the closest unlocked unopened door
        def unopened_unlocked_door(pos, cell):
            return cell and cell.type == 'door' and not cell.is_locked and not cell.is_open

        # Find the closest unopened door
        def unopened_door(pos, cell):
            return cell and cell.type == 'door' and not cell.is_open

        # Try to find an unlocked door first.
        # We do this because otherwise, opening a locked door as
        # a subgoal may try to open the same door for exploration,
        # resulting in an infinite loop.
        _, door_pos, _ = self.bot._shortest_path(
            unopened_unlocked_door, try_with_blockers=True)
        if not door_pos:
            # Try to find a locker door if an unlocked one is not available.
            _, door_pos, _ = self.bot._shortest_path(
            unopened_door, try_with_blockers=True)

        # Open the door
        if door_pos:
            door_obj = self.bot.mission.grid.get(*door_pos)
            # If we are going to a locked door, there are two cases:
            # - we already have the key, then we should not drop it
            # - we don't have the key, in which case eventually we should drop it
            got_the_key = (self.carrying
                and self.carrying.type == 'key' and self.carrying.color == door_obj.color)
            open_reason = 'KeepKey' if door_obj.is_locked and got_the_key else None
            self.bot.stack.pop()
            self.bot.stack.append(OpenSubgoal(self.bot, reason=open_reason))
            self.bot.stack.append(GoNextToSubgoal(self.bot, door_obj, reason='Open'))
            return

        assert False, "0nothing left to explore"

    def is_exploratory(self):
        return True


class Bot:
    """A bot that can solve all BabyAI levels.

    The bot maintains a plan, represented as a stack of the so-called
    subgoals. The initial set of subgoals is generated from the instruction.
    The subgoals are then executed one after another, unless a change of
    plan is required (e.g. the location of the target object is not known
    or there other objects in the way). In this case, the bot changes the plan.

    The bot can also be used to advice a suboptimal agent, e.g. play the
    role of an oracle in algorithms like DAGGER. It changes the plan based on
    the actual action that the agent took.

    The main method of the bot (and the only one you are supposed to use) is `replan`.

    Parameters:
    ----------
    mission : a freshly created BabyAI environment

    """

    def __init__(self, mission):
        # Mission to be solved
        self.mission = mission

        # Grid containing what has been mapped out
        self.grid = Grid(mission.width, mission.height)

        # Visibility mask. True for explored/seen, false for unexplored.
        self.vis_mask = np.zeros(shape=(mission.width, mission.height), dtype=np.bool)

        # Stack of tasks/subtasks to complete (tuples)
        self.stack = []

        # Process/parse the instructions
        self._process_instr(mission.instrs)

        # How many BFS searches this bot has performed
        self.bfs_counter = 0

        # How many steps were made in total in all BFS searches
        # performed by this bot
        self.bfs_step_counter = 0

    def replan(self, action_taken=None):
        """Replan and suggest an action.

        Call this method once per every iteration of the environment.

        Parameters:
        ----------
        action_taken
            The last action that the agent took. Can be `None`,
            in which case the bot assumes that the action it suggested
            was taken (or that it is the first iteration).

        Returns:
        -------
        suggested_action
            The action that the bot suggests. Can be `done` if the
            bot thinks that the mission has been accomplished.

        """
        self._process_obs()

        # Check that no box has been opened
        self._check_erroneous_box_opening(action_taken)

        # TODO: instead of updating all subgoals, just add a couple
        # properties to the `Subgoal` class.
        for subgoal in self.stack:
            subgoal.update_agent_attributes()

        if self.stack:
            self.stack[-1].replan_after_action(action_taken)

        # Clear the stack from the non-essential subgoals
        while self.stack and self.stack[-1].is_exploratory():
            self.stack.pop()

        suggested_action = None
        while self.stack:
            subgoal = self.stack[-1]
            suggested_action = subgoal.replan_before_action()
            # If is not clear what can be done for the current subgoal
            # (because it is completed, because there is blocker,
            # or because exploration is required), keep replanning
            if suggested_action is not None:
                break
        if not self.stack:
            suggested_action = self.mission.actions.done

        self._remember_current_state()

        return suggested_action

    def _find_obj_pos(self, obj_desc, adjacent=False):
        """Find the position of the closest visible object matching a given description."""

        assert len(obj_desc.obj_set) > 0

        best_distance_to_obj = 999
        best_pos = None
        best_obj = None

        for i in range(len(obj_desc.obj_set)):
            try:
                if obj_desc.obj_set[i] == self.mission.carrying:
                    continue
                obj_pos = obj_desc.obj_poss[i]

                if self.vis_mask[obj_pos]:
                    shortest_path_to_obj, _, with_blockers = self._shortest_path(
                        lambda pos, cell: pos == obj_pos,
                        try_with_blockers=True
                    )
                    assert shortest_path_to_obj is not None
                    distance_to_obj = len(shortest_path_to_obj)

                    if with_blockers:
                        # The distance should take into account the steps necessary
                        # to unblock the way. Instead of computing it exactly,
                        # we can use a lower bound on this number of steps
                        # which is 4 when the agent is not holding anything
                        # (pick, turn, drop, turn back
                        # and 7 if the agent is carrying something
                        # (turn, drop, turn back, pick,
                        # turn to other direction, drop, turn back)
                        distance_to_obj = (len(shortest_path_to_obj)
                                           + (7 if self.mission.carrying else 4))

                    # If we looking for a door and we are currently in that cell
                    # that contains the door, it will take us at least 2
                    # (3 if `adjacent == True`) steps to reach the goal.`
                    if distance_to_obj == 0:
                        distance_to_obj = 3 if adjacent else 2

                    # If what we want is to face a location that is adjacent to an object,
                    # and if we are already right next to this object,
                    # then we should not prefer this object to those at distance 2
                    if adjacent and distance_to_obj == 1:
                        distance_to_obj = 3

                    if distance_to_obj < best_distance_to_obj:
                        best_distance_to_obj = distance_to_obj
                        best_pos = obj_pos
                        best_obj = obj_desc.obj_set[i]
            except IndexError:
                # Suppose we are tracking red keys, and we just used a red key to open a door,
                # then for the last i, accessing obj_desc.obj_poss[i] will raise an IndexError
                # -> Solution: Not care about that red key we used to open the door
                pass

        return best_obj, best_pos

    def _process_obs(self):
        """Parse the contents of an observation/image and update our state."""

        grid, vis_mask = self.mission.gen_obs_grid()

        view_size = self.mission.agent_view_size
        pos = self.mission.agent_pos
        f_vec = self.mission.dir_vec
        r_vec = self.mission.right_vec

        # Compute the absolute coordinates of the top-left corner
        # of the agent's view area
        top_left = pos + f_vec * (view_size - 1) - r_vec * (view_size // 2)

        # Mark everything in front of us as visible
        for vis_j in range(0, view_size):
            for vis_i in range(0, view_size):

                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.vis_mask.shape[0]:
                    continue
                if abs_j < 0 or abs_j >= self.vis_mask.shape[1]:
                    continue

                self.vis_mask[abs_i, abs_j] = True

    def _remember_current_state(self):
        self.prev_agent_pos = self.mission.agent_pos
        self.prev_carrying = self.mission.carrying
        fwd_cell = self.mission.grid.get(*self.mission.agent_pos + self.mission.dir_vec)
        if fwd_cell and fwd_cell.type == 'door':
            self.fwd_door_was_open = fwd_cell.is_open
        self.prev_fwd_cell = fwd_cell

    def _closest_wall_or_door_given_dir(self, position, direction):
        distance = 1
        while True:
            position_to_try = position + distance * direction
            # If the current position is outside the field of view,
            # stop everything and return the previous one
            if not self.mission.in_view(*position_to_try):
                return distance - 1
            cell = self.mission.grid.get(*position_to_try)
            if cell and (cell.type.endswith('door') or cell.type == 'wall'):
                return distance
            distance += 1

    def _breadth_first_search(self, initial_states, accept_fn, ignore_blockers):
        """Performs breadth first search.

        This is pretty much your textbook BFS. The state space is agent's locations,
        but the current direction is also added to the queue to slightly prioritize
        going straight over turning.

        """
        self.bfs_counter += 1

        queue = [(state, None) for state in initial_states]
        grid = self.mission.grid
        previous_pos = dict()

        while len(queue) > 0:
            state, prev_pos = queue[0]
            queue = queue[1:]
            i, j, di, dj = state

            if (i, j) in previous_pos:
                continue

            self.bfs_step_counter += 1

            cell = grid.get(i, j)
            previous_pos[(i, j)] = prev_pos

            # If we reached a position satisfying the acceptance condition
            if accept_fn((i, j), cell):
                path = []
                pos = (i, j)
                while pos:
                    path.append(pos)
                    pos = previous_pos[pos]
                return path, (i, j), previous_pos

            # If this cell was not visually observed, don't expand from it
            if not self.vis_mask[i, j]:
                continue

            if cell:
                if cell.type == 'wall':
                    continue
                # If this is a door
                elif cell.type == 'door':
                    # If the door is closed, don't visit neighbors
                    if not cell.is_open:
                        continue
                elif not ignore_blockers:
                    continue

            # Location to which the bot can get without turning
            # are put in the queue first
            for k, l in [(di, dj), (dj, di), (-dj, -di), (-di, -dj)]:
                next_pos = (i + k, j + l)
                next_dir_vec = (k, l)
                next_state = (*next_pos, *next_dir_vec)
                queue.append((next_state, (i, j)))

        # Path not found
        return None, None, previous_pos

    def _shortest_path(self, accept_fn, try_with_blockers=False):
        """
        Finds the path to any of the locations that satisfy `accept_fn`.
        Prefers the paths that avoid blockers for as long as possible.
        """

        # Initial states to visit (BFS)
        initial_states = [(*self.mission.agent_pos, *self.mission.dir_vec)]

        path = finish = None
        with_blockers = False
        path, finish, previous_pos = self._breadth_first_search(
            initial_states, accept_fn, ignore_blockers=False)
        if not path and try_with_blockers:
            with_blockers = True
            path, finish, _ = self._breadth_first_search(
                [(i, j, 1, 0) for i, j in previous_pos],
                accept_fn, ignore_blockers=True)
            if path:
                # `path` now contains the path to a cell that is reachable without
                # blockers. Now let's add the path to this cell
                pos = path[-1]
                extra_path = []
                while pos:
                    extra_path.append(pos)
                    pos = previous_pos[pos]
                path = path + extra_path[1:]

        if path:
            # And the starting position is not required
            path = path[::-1]
            path = path[1:]

        # Note, that with_blockers only makes sense if path is not None
        return path, finish, with_blockers

    def _find_drop_pos(self, except_pos=None):
        """
        Find a position where an object can be dropped, ideally without blocking anything.
        """

        grid = self.mission.grid

        def match_unblock(pos, cell):
            # Consider the region of 8 neighboring cells around the candidate cell.
            # If dropping the object in the candidate makes this region disconnected,
            # then probably it is better to drop elsewhere.

            i, j = pos
            agent_pos = tuple(self.mission.agent_pos)

            if np.array_equal(pos, agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[i, j] or grid.get(i, j):
                return False

            # We distinguish cells of three classes:
            # class 0: the empty ones, including open doors
            # class 1: those that are not interesting (just walls so far)
            # class 2: all the rest, including objects and cells that are current not visible,
            #          and hence may contain objects, and also `except_pos` at it may soon contain
            #          an object
            # We want to ensure that empty cells are connected, and that one can reach
            # any object cell from any other object cell.
            cell_class = []
            for k, l in [(-1, -1), (0, -1), (1, -1), (1, 0),
                         (1, 1), (0, 1), (-1, 1), (-1, 0)]:
                nb_pos = (i + k, j + l)
                cell = grid.get(*nb_pos)
                # compeletely blocked
                if self.vis_mask[nb_pos] and cell and cell.type == 'wall':
                    cell_class.append(1)
                # empty
                elif (self.vis_mask[nb_pos]
                        and (not cell or (cell.type == 'door' and cell.is_open) or nb_pos == agent_pos)
                        and nb_pos != except_pos):
                    cell_class.append(0)
                # an object cell
                else:
                    cell_class.append(2)

            # Now we need to check that empty cells are connected. To do that,
            # let's check how many times empty changes to non-empty
            changes = 0
            for i in range(8):
                if bool(cell_class[(i + 1) % 8]) != bool(cell_class[i]):
                    changes += 1

            # Lastly, we need check that every object has an adjacent empty cell
            for i in range(8):
                next_i = (i + 1) % 8
                prev_i = (i + 7) % 8
                if cell_class[i] == 2 and cell_class[prev_i] != 0 and cell_class[next_i] != 0:
                    return False

            return changes <= 2

        def match_empty(pos, cell):
            i, j = pos

            if np.array_equal(pos, self.mission.agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[pos] or grid.get(*pos):
                return False

            return True

        _, drop_pos, _ = self._shortest_path(match_unblock)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_empty)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_unblock, try_with_blockers=True)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_empty, try_with_blockers=True)

        return drop_pos

    def _process_instr(self, instr):
        """
        Translate instructions into an internal form the agent can execute
        """

        if isinstance(instr, GoToInstr):
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, OpenInstr):
            self.stack.append(OpenSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc, reason='Open'))
            return

        if isinstance(instr, PickupInstr):
            # We pick up and immediately drop so
            # that we may carry other objects
            self.stack.append(DropSubgoal(self))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(DropSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_fixed, reason='PutNext'))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_move))
            return

        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr):
            self._process_instr(instr.instr_b)
            self._process_instr(instr.instr_a)
            return

        if isinstance(instr, AfterInstr):
            self._process_instr(instr.instr_a)
            self._process_instr(instr.instr_b)
            return

        assert False, "unknown instruction type"

    def _check_erroneous_box_opening(self, action):
        """
        When the agent opens a box, we raise an error and mark the task unsolvable.
        This is a tad conservative, because maybe the box is irrelevant to the mission.
        """
        if (action == self.mission.actions.toggle
                and self.prev_fwd_cell is not None
                and self.prev_fwd_cell.type == 'box'):
            raise DisappearedBoxError('A box was opened. I am not sure I can help now.')



================================================
FILE: babyai/efficiency.py
================================================
#!/usr/bin/env python3
"""
Code for launching imitation learning sample efficiency experiments.
"""

import os
import time
import subprocess
import argparse
import math
from babyai.cluster_specific import launch_job

BIG_MODEL_PARAMS = '--memory-dim=2048 --recurrence=80 --batch-size=128 --instr-arch=attgru --instr-dim=256'
SMALL_MODEL_PARAMS = '--batch-size=256'

def main(env, seed, training_time, min_demos, max_demos=None,
         step_size=math.sqrt(2), pretrained_model=None, level_type='small',
         val_episodes=512):
    demos = env

    if not max_demos:
        max_demos = min_demos
        min_demos = max_demos - 1

    demo_counts = []
    demo_count = max_demos
    while demo_count >= min_demos:
        demo_counts.append(demo_count)
        demo_count = math.ceil(demo_count / step_size)

    for demo_count in demo_counts:
        # Decide on the parameters
        epoch_length = 25600 if level_type == 'small' else 51200
        epochs = training_time // epoch_length

        # Print info
        print('{} demos, {} epochs of {} examples'.format(demo_count, epochs, epoch_length))

        # Form the command
        model_name = '{}_seed{}_{}'.format(demos, seed, demo_count)
        if pretrained_model:
            model_name += '_{}'.format(pretrained_model)
        jobname = '{}_efficiency'.format(demos, min_demos, max_demos)
        model_params = BIG_MODEL_PARAMS if level_type == 'big' else SMALL_MODEL_PARAMS
        cmd = ('{model_params} --val-episodes {val_episodes}'
               ' --seed {seed} --env {env} --demos {demos}'
               ' --val-interval 1 --log-interval 1 --epoch-length {epoch_length}'
               ' --model {model_name} --episodes {demo_count} --epochs {epochs} --patience {epochs}'
          .format(**locals()))
        if pretrained_model:
            cmd += ' --pretrained-model {}'.format(pretrained_model)
        launch_job(cmd, jobname)

        seed += 1



================================================
FILE: babyai/evaluate.py
================================================
import numpy as np
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper


# Returns the performance of the agent on the environment for a particular number of episodes.
def evaluate(agent, env, episodes, model_agent=True, offsets=None):
    # Initialize logs
    if model_agent:
        agent.model.eval()
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

    if offsets:
        count = 0

    for i in range(episodes):
        if offsets:
            # Ensuring test on seed offsets that generated successful demonstrations
            while count != offsets[i]:
                obs = env.reset()
                count += 1

        obs = env.reset()
        agent.on_reset()
        done = False

        num_frames = 0
        returnn = 0
        obss = []
        while not done:
            action = agent.act(obs)['action']
            obss.append(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward


        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
    if model_agent:
        agent.model.train()
    return logs


def evaluate_demo_agent(agent, episodes):
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    number_of_demos = len(agent.demos)

    for demo_id in range(min(number_of_demos, episodes)):
        logs["num_frames_per_episode"].append(len(agent.demos[demo_id]))

    return logs


class ManyEnvs(gym.Env):

    def __init__(self, envs):
        self.envs = envs
        self.done = [False] * len(self.envs)

    def seed(self, seeds):
        [env.seed(seed) for seed, env in zip(seeds, self.envs)]

    def reset(self):
        many_obs = [env.reset() for env in self.envs]
        self.done = [False] * len(self.envs)
        return many_obs

    def step(self, actions):
        self.results = [env.step(action) if not done else self.last_results[i]
                        for i, (env, action, done)
                        in enumerate(zip(self.envs, actions, self.done))]
        self.done = [result[2] for result in self.results]
        self.last_results = self.results
        return zip(*self.results)

    def render(self):
        raise NotImplementedError


# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=False, pixel=False):
    num_envs = min(256, episodes)

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        if pixel:
            env = RGBImgPartialObsWrapper(env)
        envs.append(env)
    env = ManyEnvs(envs)

    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": []
    }

    for i in range((episodes + num_envs - 1) // num_envs):
        seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)
        env.seed(seeds)

        many_obs = env.reset()

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')
        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
        while (num_frames == 0).any():
            action = agent.act_batch(many_obs)['action']
            if return_obss_actions:
                for i in range(num_envs):
                    if not already_done[i]:
                        obss[i].append(many_obs[i])
                        actions[i].append(action[i].item())
            many_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)

    return logs



================================================
FILE: babyai/imitation.py
================================================
import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
from babyai.evaluate import batch_evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.model import ACModel
import multiprocessing
import os
import json
import logging

logger = logging.getLogger(__name__)

import numpy


class EpochIndexSampler:
    """
    Generate smart indices for epochs that are smaller than the dataset size.

    The usecase: you have a code that has a strongly baken in notion of an epoch,
    e.g. you can only validate in the end of the epoch. That ties a lot of
    aspects of training to the size of the dataset. You may want to validate
    more often than once per a complete pass over the dataset.

    This class helps you by generating a sequence of smaller epochs that
    use different subsets of the dataset, as long as this is possible.
    This allows you to keep the small advantage that sampling without replacement
    provides, but also enjoy smaller epochs.
    """
    def __init__(self, n_examples, epoch_n_examples):
        self.n_examples = n_examples
        self.epoch_n_examples = epoch_n_examples

        self._last_seed = None

    def _reseed_indices_if_needed(self, seed):
        if seed == self._last_seed:
            return

        rng = numpy.random.RandomState(seed)
        self._indices = list(range(self.n_examples))
        rng.shuffle(self._indices)
        logger.info('reshuffle the dataset')

        self._last_seed = seed

    def get_epoch_indices(self, epoch):
        """Return indices corresponding to a particular epoch.

        Tip: if you call this function with consecutive epoch numbers,
        you will avoid expensive reshuffling of the index list.

        """
        seed = epoch * self.epoch_n_examples // self.n_examples
        offset = epoch * self.epoch_n_examples % self.n_examples

        indices = []
        while len(indices) < self.epoch_n_examples:
            self._reseed_indices_if_needed(seed)
            n_lacking = self.epoch_n_examples - len(indices)
            indices += self._indices[offset:offset + min(n_lacking, self.n_examples - offset)]
            offset = 0
            seed += 1

        return indices


class ImitationLearning(object):
    def __init__(self, args, ):
        self.args = args

        utils.seed(self.args.seed)
        self.val_seed = self.args.val_seed

        # args.env is a list when training on multiple environments
        if getattr(args, 'multi_env', None):
            self.env = [gym.make(item) for item in args.multi_env]

            self.train_demos = []
            for demos, episodes in zip(args.multi_demos, args.multi_episodes):
                demos_path = utils.get_demos_path(demos, None, None, valid=False)
                logger.info('loading {} of {} demos'.format(episodes, demos))
                train_demos = utils.load_demos(demos_path)
                logger.info('loaded demos')
                if episodes > len(train_demos):
                    raise ValueError("there are only {} train demos in {}".format(len(train_demos), demos))
                self.train_demos.extend(train_demos[:episodes])
                logger.info('So far, {} demos loaded'.format(len(self.train_demos)))

            self.val_demos = []
            for demos, episodes in zip(args.multi_demos, [args.val_episodes] * len(args.multi_demos)):
                demos_path_valid = utils.get_demos_path(demos, None, None, valid=True)
                logger.info('loading {} of {} valid demos'.format(episodes, demos))
                valid_demos = utils.load_demos(demos_path_valid)
                logger.info('loaded demos')
                if episodes > len(valid_demos):
                    logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(valid_demos)))
                self.val_demos.extend(valid_demos[:episodes])
                logger.info('So far, {} valid demos loaded'.format(len(self.val_demos)))

            logger.info('Loaded all demos')

            observation_space = self.env[0].observation_space
            action_space = self.env[0].action_space

        else:
            self.env = gym.make(self.args.env)

            demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
            demos_path_valid = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=True)

            logger.info('loading demos')
            self.train_demos = utils.load_demos(demos_path)
            logger.info('loaded demos')
            if args.episodes:
                if args.episodes > len(self.train_demos):
                    raise ValueError("there are only {} train demos".format(len(self.train_demos)))
                self.train_demos = self.train_demos[:args.episodes]

            self.val_demos = utils.load_demos(demos_path_valid)
            if args.val_episodes > len(self.val_demos):
                logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(self.val_demos)))
            self.val_demos = self.val_demos[:self.args.val_episodes]

            observation_space = self.env.observation_space
            action_space = self.env.action_space

        self.obss_preprocessor = utils.ObssPreprocessor(args.model, observation_space,
                                                        getattr(self.args, 'pretrained_model', None))

        # Define actor-critic model
        self.acmodel = utils.load_model(args.model, raise_not_found=False)
        if self.acmodel is None:
            if getattr(self.args, 'pretrained_model', None):
                self.acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
            else:
                logger.info('Creating new model')
                self.acmodel = ACModel(self.obss_preprocessor.obs_space, action_space,
                                       args.image_dim, args.memory_dim, args.instr_dim,
                                       not self.args.no_instr, self.args.instr_arch,
                                       not self.args.no_mem, self.args.arch)
        self.obss_preprocessor.vocab.save()
        utils.save_model(self.acmodel, args.model)

        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def default_model_name(args):
        if getattr(args, 'multi_env', None):
            # It's better to specify one's own model name for this scenario
            named_envs = '-'.join(args.multi_env)
        else:
            named_envs = args.env

        # Define model name
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        instr = args.instr_arch if args.instr_arch else "noinstr"
        model_name_parts = {
            'envs': named_envs,
            'arch': args.arch,
            'instr': instr,
            'seed': args.seed,
            'suffix': suffix}
        default_model_name = "{envs}_IL_{arch}_{instr}_seed{seed}_{suffix}".format(**model_name_parts)
        if getattr(args, 'pretrained_model', None):
            default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
        return default_model_name

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def run_epoch_recurrence(self, demos, is_training=False, indices=None):
        if not indices:
            indices = list(range(len(demos)))
            if is_training:
                np.random.shuffle(indices)
        batch_size = min(self.args.batch_size, len(demos))
        offset = 0

        if not is_training:
            self.acmodel.eval()

        # Log dictionary
        log = {"entropy": [], "policy_loss": [], "accuracy": []}

        start_time = time.time()
        frames = 0
        for batch_index in range(len(indices) // batch_size):
            logger.info("batch {}, FPS so far {}".format(
                batch_index, frames / (time.time() - start_time) if frames else 0))
            batch = [demos[i] for i in indices[offset: offset + batch_size]]
            frames += sum([len(demo[3]) for demo in batch])

            _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training)

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["accuracy"].append(_log["accuracy"])

            offset += batch_size
        log['total_frames'] = frames

        if not is_training:
            self.acmodel.train()

        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False):
        batch = utils.demos.transform_demos(batch)
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]

        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))

        flat_batch = np.array(flat_batch)
        inds = inds[:-1]
        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=self.device)
        episode_ids = np.zeros(len(flat_batch))
        memory = torch.zeros([len(batch), self.acmodel.memory_size], device=self.device)

        preprocessed_first_obs = self.obss_preprocessor(obss[inds], device=self.device)
        instr_embedding = self.acmodel._get_instr_embedding(preprocessed_first_obs.instr)

        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            with torch.no_grad():
                # taking the memory till len(inds), as demos beyond that have already finished
                new_memory = self.acmodel(
                    preprocessed_obs,
                    memory[:len(inds), :], instr_embedding[:len(inds)])['memory']

            memories[inds, :] = memory[:len(inds), :]
            memory[:len(inds), :] = new_memory
            episode_ids[inds] = range(len(inds))

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:len(inds) - sum(done_step)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]

        # Here, actual backprop upto args.recurrence happens
        final_loss = 0
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0

        indexes = self.starting_indexes(num_frames)
        memory = memories[indexes]
        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence
        for _ in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            action_step = action_true[indexes]
            mask_step = mask[indexes]
            model_results = self.acmodel(
                preprocessed_obs, memory * mask_step,
                instr_embedding[episode_ids[indexes]])
            dist = model_results['dist']
            memory = model_results['memory']

            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy += float((action_pred == action_step.unsqueeze(1)).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss
            indexes += 1

        final_loss /= self.args.recurrence

        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        log = {}
        log["entropy"] = float(final_entropy / self.args.recurrence)
        log["policy_loss"] = float(final_policy_loss / self.args.recurrence)
        log["accuracy"] = float(accuracy)

        return log

    def validate(self, episodes, verbose=True):
        if verbose:
            logger.info("Validating the model")
        if getattr(self.args, 'multi_env', None):
            agent = utils.load_agent(self.env[0], model_name=self.args.model, argmax=True)
        else:
            agent = utils.load_agent(self.env, model_name=self.args.model, argmax=True)

        # Setting the agent model to the current model
        agent.model = self.acmodel

        agent.model.eval()
        logs = []

        for env_name in ([self.args.env] if not getattr(self.args, 'multi_env', None)
                         else self.args.multi_env):
            logs += [batch_evaluate(agent, env_name, self.val_seed, episodes)]
            self.val_seed += episodes
        agent.model.train()

        return logs

    def collect_returns(self):
        logs = self.validate(episodes=self.args.eval_episodes, verbose=False)
        mean_return = {tid: np.mean(log["return_per_episode"]) for tid, log in enumerate(logs)}
        return mean_return

    def train(self, train_demos, writer, csv_writer, status_path, header, reset_status=False):
        # Load the status
        def initial_status():
            return {'i': 0,
                    'num_frames': 0,
                    'patience': 0}

        status = initial_status()
        if os.path.exists(status_path) and not reset_status:
            with open(status_path, 'r') as src:
                status = json.load(src)
        elif not os.path.exists(os.path.dirname(status_path)):
            # Ensure that the status directory exists
            os.makedirs(os.path.dirname(status_path))

        # If the batch size is larger than the number of demos, we need to lower the batch size
        if self.args.batch_size > len(train_demos):
            self.args.batch_size = len(train_demos)
            logger.info("Batch size too high. Setting it to the number of train demos ({})".format(len(train_demos)))

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.acmodel, self.args.model)

        # best mean return to keep track of performance on validation set
        best_success_rate, patience, i = 0, 0, 0
        total_start_time = time.time()

        epoch_length = self.args.epoch_length
        if not epoch_length:
            epoch_length = len(train_demos)
        index_sampler = EpochIndexSampler(len(train_demos), epoch_length)

        while status['i'] < getattr(self.args, 'epochs', int(1e9)):
            if 'patience' not in status:  # if for some reason you're finetuining with IL an RL pretrained agent
                status['patience'] = 0
            # Do not learn if using a pre-trained model that already lost patience
            if status['patience'] > self.args.patience:
                break
            if status['num_frames'] > self.args.frames:
                break

            update_start_time = time.time()

            indices = index_sampler.get_epoch_indices(status['i'])
            log = self.run_epoch_recurrence(train_demos, is_training=True, indices=indices)
            
            # Learning rate scheduler
            self.scheduler.step()

            status['num_frames'] += log['total_frames']
            status['i'] += 1

            update_end_time = time.time()

            # Print logs
            if status['i'] % self.args.log_interval == 0:
                total_ellapsed_time = int(time.time() - total_start_time)

                fps = log['total_frames'] / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)

                for key in log:
                    log[key] = np.mean(log[key])

                train_data = [status['i'], status['num_frames'], fps, total_ellapsed_time,
                              log["entropy"], log["policy_loss"], log["accuracy"]]

                logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}".format(*train_data))

                # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
                # afterwards when status['i'] % self.args.val_interval == 0
                if status['i'] % self.args.val_interval != 0:
                    # instantiate a validation_log with empty strings when no validation is done
                    validation_data = [''] * len([key for key in header if 'valid' in key])
                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

            if status['i'] % self.args.val_interval == 0:

                valid_log = self.validate(self.args.val_episodes)
                mean_return = [np.mean(log['return_per_episode']) for log in valid_log]
                success_rate = [np.mean([1 if r > 0 else 0 for r in log['return_per_episode']]) for log in
                                valid_log]

                val_log = self.run_epoch_recurrence(self.val_demos)
                validation_accuracy = np.mean(val_log["accuracy"])

                if status['i'] % self.args.log_interval == 0:
                    validation_data = [validation_accuracy] + mean_return + success_rate
                    logger.info(("Validation: A {: .3f} " + ("| R {: .3f} " * len(mean_return) +
                                                             "| S {: .3f} " * len(success_rate))
                                 ).format(*validation_data))

                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data + validation_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

                # In case of a multi-env, the update condition would be "better mean success rate" !
                if np.mean(success_rate) > best_success_rate:
                    best_success_rate = np.mean(success_rate)
                    status['patience'] = 0
                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)
                    # Saving the model
                    logger.info("Saving best model")

                    if torch.cuda.is_available():
                        self.acmodel.cpu()
                    utils.save_model(self.acmodel, self.args.model + "_best")
                    self.obss_preprocessor.vocab.save(utils.get_vocab_path(self.args.model + "_best"))
                    if torch.cuda.is_available():
                        self.acmodel.cuda()
                else:
                    status['patience'] += 1
                    logger.info(
                        "Losing patience, new value={}, limit={}".format(status['patience'], self.args.patience))

                if torch.cuda.is_available():
                    self.acmodel.cpu()
                utils.save_model(self.acmodel, self.args.model)
                self.obss_preprocessor.vocab.save()
                if torch.cuda.is_available():
                    self.acmodel.cuda()
                with open(status_path, 'w') as dst:
                    json.dump(status, dst)

        return best_success_rate



================================================
FILE: babyai/model.py
================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="bow_endpool_res", aux_info=None):
        super().__init__()

        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        # if not self.use_instr:
        #     raise ValueError("FiLM architecture can be used when instructions are enabled")
        self.image_conv = nn.Sequential(*[
            *([ImageBOWEmbedding(obs_space['image'], 128)] if use_bow else []),
            *([nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=(8, 8),
                stride=8, padding=0)] if pixel else []),
            nn.Conv2d(
                in_channels=128 if use_bow or pixel else 3, out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])
        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            num_module = 2
            self.controllers = []
            for ni in range(num_module):
                mod = FiLM(
                    in_features=self.final_instr_dim,
                    out_features=128 if ni < num_module-1 else self.image_dim,
                    in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_' + str(ni), mod)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, :instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, :mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if 'pixel' in self.arch:
            x /= 256.0
        x = self.image_conv(x)
        if self.use_instr:
            for controller in self.controllers:
                out = controller(x, instr_embedding)
                if self.res:
                    out += x
                x = out
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))



================================================
FILE: babyai/plotting.py
================================================
"""Loading and plotting data from CSV logs.

Schematic example of usage

- load all `log.csv` files that can be found by recursing a root directory:
  `dfs = load_logs($BABYAI_STORAGE)`
- concatenate them in the master dataframe
  `df = pandas.concat(dfs, sort=True)`
- plot average performance for groups of runs using `plot_average(df, ...)`
- plot performance for each run in a group using `plot_all_runs(df, ...)`

Note:
- you can choose what to plot
- groups are defined by regular expressions over full paths to .csv files.
  For example, if your model is called "model1" and you trained it with multiple seeds,
  you can filter all the respective runs with the regular expression ".*model1.*"
- you may want to load your logs from multiple storage directories
  before concatening them into a master dataframe

"""

import os
import re
import numpy as np
from matplotlib import pyplot
import pandas
import scipy
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.linalg import cholesky, cho_solve, solve_triangular


def load_log(dir_):
    """Loads log from a directory and adds it to a list of dataframes."""
    df = pandas.read_csv(os.path.join(dir_, 'log.csv'),
                         error_bad_lines=False,
                         warn_bad_lines=True)
    if not len(df):
        print("empty df at {}".format(dir_))
        return
    df['model'] = dir_
    return df

def load_multiphase_log(dir_):
    df = load_log(dir_)
    phases = []
    cur_phase = 0
    prev_upd = 0
    for i in range(len(df)):
        upd = df.iloc[i]['update']
        if upd < prev_upd:
            cur_phase += 1
        phases.append(cur_phase)
        prev_upd = upd
    df['phase'] = phases
    return df

def load_logs(root, multiphase=False):
    dfs = []
    for root, dirs, files in os.walk(root, followlinks=True):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_multiphase_log(root) if multiphase else load_log(root))
    return dfs


def plot_average_impl(df, regexps, y_value='return_mean', window=1, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])

    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                     for regex in regexps]

    for regex, models in zip(regexps, model_groups):
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                               for _, df_model in df_re.groupby('model')]
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= median_progress]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pandas.concat(parts)

        df_agg = df_re.groupby([x_value]).agg([agg])
        values = df_agg[y_value][agg]
        pyplot.plot(df_agg.index, values, label=regex)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])


def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))
    plot_average_impl(*args, **kwargs)
    pyplot.legend()


def plot_all_runs(df, regex, quantity='return_mean', x_axis='frames', window=1, color=None):
    """Plot a group of runs defined by a regex."""
    pyplot.figure(figsize=(15, 5))

    df = df.dropna(subset=[quantity])

    unique_models = df['model'].unique()
    models = [m for m in unique_models if re.match(regex, m)]
    df_re = df[df['model'].isin(models)]
    for model, df_model in df_re.groupby('model'):
        values = df_model[quantity]
        values = values.rolling(window, center=True).mean()

        kwargs = {}
        if color:
            kwargs['color'] = color(model)
        pyplot.plot(df_model[x_axis],
                    values,
                    label=model,
                    **kwargs)
        print(model, df_model[x_axis].max())

    pyplot.legend()


def model_num_samples(model):
    # the number of samples is mangled in the name
    return int(re.findall('_([0-9]+)', model)[0])


def get_fps(df):
    data = df['FPS']
    data = data.tolist()
    return np.array(data)


def best_within_normal_time(df, regex, patience, limit='epochs', window=1, normal_time=None, summary_path=None):
    """
    Compute the best success rate that is achieved in all runs within the normal time.

    The normal time is defined as `patience * T`, where `T` is the time it takes for the run
    with the most demonstrations to converge. `window` is the size of the sliding window that is
    used for smoothing.

    Returns a dataframe with the best success rate for the runs that match `regex`.

    """
    print()
    print(regex)
    models = [model for model in df['model'].unique() if re.match(regex, model)]
    num_samples = [model_num_samples(model) for model in models]
    # sort models according to the number of samples
    models, num_samples = zip(*sorted(list(zip(models, num_samples)), key=lambda tupl: tupl[1]))

    # choose normal time
    max_samples = max(num_samples)
    limits = []
    for model, num in zip(models, num_samples):
        if num == max_samples:
            df_model = df[df['model'] == model]
            success_rate = df_model['validation_success_rate'].rolling(window, center=True).mean()
            if np.isnan(success_rate.max()) or success_rate.max() < 0.99:
                raise ValueError('{} has not solved the level yet, only at {} so far'.format(
                    model, success_rate.max()))
            first_solved = (success_rate > 0.99).to_numpy().nonzero()[0][0]
            row = df_model.iloc[first_solved]
            print("the model with {} samples first solved after {} epochs ({} seconds, {} frames)".format(
                max_samples, row['update'], row['duration'], row['frames']))
            limits.append(patience * row[limit] + 1)
    if not normal_time:
        normal_time = np.mean(limits)
        print('using {} as normal time'.format(normal_time))

    summary_data = []

    # check how many examples is required to succeed within normal time
    min_samples_required = None
    need_more_time = False
    print("{: <100} {}\t{}\t{}\t{}".format(
        'model_name', 'sr_nt', 'sr', 'dur_nt', 'dur_days'))
    for model, num in zip(models, num_samples):
        df_model = df[df['model'] == model]
        success_rate = df_model['validation_success_rate'].rolling(window, center=True).mean()
        max_within_normal_time = success_rate[df_model[limit] < normal_time].max()
        if max_within_normal_time > 0.99:
            min_samples_required = min(num, min_samples_required
                                       if min_samples_required
                                       else int(1e9))
        if df_model[limit].max() < normal_time:
            need_more_time = True
        print("{: <50} {: <5.4g}\t{: <5.4g}\t{: <5.3g}\t{:.3g}".format(
            model.split('/')[-1],
            max_within_normal_time * 100,
            success_rate.max() * 100,
            df_model[limit].max() / normal_time,
            df_model['duration'].max() / 86400))
        summary_data.append((num, max_within_normal_time))

    summary_df = pandas.DataFrame(summary_data, columns=('num_samples', 'success_rate'))
    if summary_path:
        summary_df.to_csv(summary_path)

    if min(num_samples) == min_samples_required:
        raise ValueError('should be run with less samples!')
    if need_more_time:
        raise ValueError('should be run for more time!')
    return summary_df, normal_time


def estimate_sample_efficiency(df, visualize=False, figure_path=None):
    """
    Estimate sample efficiency and its uncertainty using Gaussian Process.

    This function interpolates between data points given in `df` using a Gaussian Process.
    It returns a 99% interval based on the GP predictions.

    """
    f, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    # preprocess the data
    print("{} datapoints".format(len(df)))
    x = np.log2(df['num_samples'].values)
    y = df['success_rate']
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices].values

    success_threshold = 0.99
    min_datapoints = 5
    almost_threshold = 0.95

    if (y > success_threshold).sum() < min_datapoints:
        raise ValueError(f"You have less than {min_datapoints} datapoints above the threshold.\n"
                         "Consider running experiments with more examples.")
    if ((y > almost_threshold) & (y < success_threshold)).sum() < min_datapoints:
        raise ValueError(f"You have less than {min_datapoints} datapoints"
              " for which the threshold is almost crossed.\n"
              "Consider running experiments with less examples.")
    # try to throw away the extra points with low performance
    # the model is not suitable for handling those
    while True:
        if ((y[1:] > success_threshold).sum() >= min_datapoints
                and ((y[1:] > almost_threshold) & (y[1:] < success_threshold)).sum()
                        >= min_datapoints):
            print('throwing away x={}, y={}'.format(x[0], y[0]))
            x = x[1:]
            y = y[1:]
        else:
            break

    print("min x: {}, max x: {}, min y: {}, max y: {}".format(x.min(), x.max(), y.min(), y.max()))
    y = (y - success_threshold) * 100

    # fit an RBF GP
    kernel = 1.0 * RBF() + WhiteKernel(noise_level_bounds=(1e-10, 10))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False).fit(x[:, None], y)
    print("Kernel:", gp.kernel_)
    print("Marginal likelihood:", gp.log_marginal_likelihood_value_)

    # compute the success rate posterior
    grid_step = 0.02
    grid = np.arange(x[0], x[-1], grid_step)
    y_grid_mean, y_grid_cov = gp.predict(grid[:, None], return_cov=True)
    noise_level = gp.kernel_.k2.noise_level
    f_grid_cov = y_grid_cov - np.diag(np.ones_like(y_grid_cov[0]) * noise_level)

    if visualize:
        axis = axes[0]
        axis.plot(x, y, 'o')
        axis.plot(grid, y_grid_mean)
        axis.set_xlabel('log2(N)')
        axis.set_ylabel('accuracy minus 99%')
        axis.set_title('Data Points & Posterior')
        axis.fill_between(grid, y_grid_mean - np.sqrt(np.diag(y_grid_cov)),
                         y_grid_mean + np.sqrt(np.diag(y_grid_cov)),
                         alpha=0.2, color='k')
        axis.fill_between(grid, y_grid_mean -np.sqrt(np.diag(f_grid_cov)),
                 y_grid_mean + np.sqrt(np.diag(f_grid_cov)),
                 alpha=0.2, color='g')
        axis.hlines(0, x[0], x[-1])

    # compute the N_min posterior
    probs = []
    total_p = 0.
    print("Estimating N_min using a grid of {} points".format(len(grid)))
    for j in range(len(grid)):
        mu = y_grid_mean[:j + 1].copy()
        mu[j] *= -1
        sigma = f_grid_cov[:j + 1, :j + 1].copy()
        sigma[j, :j] *= -1
        sigma[:j, j] *= -1
        sigma[np.diag_indices_from(sigma)] += 1e-6
        # the probability that the first time the success rate crosses the threshold
        # will be between grid[j - 1] and grid[j]
        p = stats.multivariate_normal.cdf(np.zeros_like(mu), mu, sigma, abseps=1e-3, releps=1e-3)
        probs.append(p)
        total_p += p

        can_stop = total_p.sum() > 0.999
        if j and (can_stop or j % 10 == 0):
            print('{} points done'.format(j))
            print(" ".join(["{:.3g}".format(p) for p in probs[-10:]]))
        if can_stop:
            print('the rest is unlikely')
            break
    probs = np.array(probs)
    if (probs.sum() - 1) > 0.01:
        raise ValueError("oops, probabilities don't sum to one")
    else:
        # probs should sum to 1, but there is always a bit of error
        probs = probs / probs.sum()

    first_prob = (probs > 1e-10).nonzero()[0][0]
    subgrid = grid[first_prob:len(probs)]
    subprobs = probs[first_prob:]
    mean_n_min = (subprobs * subgrid).sum()
    mean_n_min_squared = (subprobs * subgrid ** 2).sum()
    std_n_min = (mean_n_min_squared - mean_n_min ** 2) ** 0.5
    if visualize:
        # visualize the N_min posterior density
        # visualize the non-Gaussianity of N_min posterior density
        axis = axes[2]
        axis.plot(subgrid, subprobs)
        axis.plot(subgrid, stats.norm.pdf(subgrid, mean_n_min, std_n_min) * grid_step)

    # compute the credible interval
    cdf = np.cumsum(probs)
    left = grid[(cdf > 0.01).nonzero()[0][0]]
    right = grid[(cdf > 0.99).nonzero()[0][0]]
    print("99% credible interval for N_min:", 2 ** left,  2 ** right)

    if visualize:
        axis = axes[1]
        axis.plot(x, y, 'o')
        axis.set_xlabel('log2(N)')
        axis.set_ylabel('accuracy minus 99%')
        axis.hlines(0, x[0], x[-1])
        axis.vlines(left, min(y), max(y), color='r')
        axis.vlines(mean_n_min, min(y), max(y), color='k')
        axis.vlines(right, min(y), max(y), color='r')
        axis.set_title('Data points & Conf. interval for min. number of samples')

    pyplot.tight_layout()
    if figure_path:
        pyplot.savefig(figure_path)
    return {'min': 2 ** left, 'max': 2 ** right,
            'mean_log2': mean_n_min, 'std_log2': std_n_min}



================================================
FILE: babyai/levels/__init__.py
================================================
from collections import OrderedDict

from . import iclr19_levels
from . import bonus_levels
from . import test_levels

from .levelgen import test, level_dict



================================================
FILE: babyai/levels/bonus_levels.py
================================================
import gym
from gym_minigrid.minigrid import Key, Ball, Box
from .verifier import *
from .levelgen import *


class Level_GoToRedBlueBall(RoomGridLevel):
    """
    Go to the red ball or to the blue ball.
    There is exactly one red or blue ball, and some distractors.
    The distractors are guaranteed not to be red or blue balls.
    Language is not required to solve this level.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()

        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Ensure there is only one red or blue ball
        for dist in dists:
            if dist.type == 'ball' and (dist.color == 'blue' or dist.color == 'red'):
                raise RejectSampling('can only have one blue or red ball')

        color = self._rand_elem(['red', 'blue'])
        obj, _ = self.add_object(0, 0, 'ball', color)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_OpenRedDoor(RoomGridLevel):
    """
    Go to the red door
    (always unlocked, in the current room)
    Note: this level is intentionally meant for debugging and is
    intentionally kept very simple.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=5,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_door(0, 0, 0, 'red', locked=False)
        self.place_agent(0, 0)
        self.instrs = OpenInstr(ObjDesc('door', 'red'))


class Level_OpenDoor(RoomGridLevel):
    """
    Go to the door
    The door to open is given by its color or by its location.
    (always unlocked, in the current room)
    """

    def __init__(
        self,
        debug=False,
        select_by=None,
        seed=None
    ):
        self.select_by = select_by
        self.debug = debug
        super().__init__(seed=seed)

    def gen_mission(self):
        door_colors = self._rand_subset(COLOR_NAMES, 4)
        objs = []

        for i, color in enumerate(door_colors):
            obj, _ = self.add_door(1, 1, door_idx=i, color=color, locked=False)
            objs.append(obj)

        select_by = self.select_by
        if select_by is None:
            select_by = self._rand_elem(["color", "loc"])
        if select_by == "color":
            object = ObjDesc(objs[0].type, color=objs[0].color)
        elif select_by == "loc":
            object = ObjDesc(objs[0].type, loc=self._rand_elem(LOC_NAMES))

        self.place_agent(1, 1)
        self.instrs = OpenInstr(object, strict=self.debug)


class Level_OpenDoorDebug(Level_OpenDoor):
    """
    Same as OpenDoor but the level stops when any door is opened
    """

    def __init__(
        self,
        select_by=None,
        seed=None
    ):
        super().__init__(select_by=select_by, debug=True, seed=seed)


class Level_OpenDoorColor(Level_OpenDoor):
    """
    Go to the door
    The door is selected by color.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="color",
            seed=seed
        )


#class Level_OpenDoorColorDebug(Level_OpenDoorColor, Level_OpenDoorDebug):
    """
    Same as OpenDoorColor but the level stops when any door is opened
    """
#    pass


class Level_OpenDoorLoc(Level_OpenDoor):
    """
    Go to the door
    The door is selected by location.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="loc",
            seed=seed
        )


class Level_GoToDoor(RoomGridLevel):
    """
    Go to a door
    (of a given color, in the current room)
    No distractors, no language variation
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        objs = []
        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)
        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc('door', obj.color))


class Level_GoToObjDoor(RoomGridLevel):
    """
    Go to an object or door
    (of a given type and color, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=8,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(1, 1)
        objs = self.add_distractors(1, 1, num_distractors=8, all_unique=False)

        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)

        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_ActionObjDoor(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(1, 1, num_distractors=5)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append(door)

        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == 'door':
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class Level_UnlockLocal(RoomGridLevel):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors
        super().__init__(seed=seed)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, 'key', door.color)
        if self.distractors:
            self.add_distractors(1, 1, num_distractors=3)
        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class Level_UnlockLocalDist(Level_UnlockLocal):
    """
    Fetch a key and unlock a door
    (in the current room, with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)


class Level_KeyInBox(RoomGridLevel):
    """
    Unlock a door. Key is in a box (in the current room).
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)

        # Put the key in the box, then place the box in the room
        key = Key(door.color)
        box = Box(self._rand_color(), key)
        self.place_in_room(1, 1, box)

        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class Level_UnlockPickup(RoomGridLevel):
    """
    Unlock a door, then pick up a box in another room
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors

        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)
        if self.distractors:
            self.add_distractors(num_distractors=4)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnlockPickupDist(Level_UnlockPickup):
    """
    Unlock a door, then pick up an object in another room
    (with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)


class Level_BlockedUnlockPickup(RoomGridLevel):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0]-1, pos[1], Ball(color))
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_UnlockToUnlock(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=30*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)

        # Add a key of color A in the room on the right
        self.add_object(2, 0, kind="key", color=colors[0])

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)

        # Add a key of color B in the middle room
        self.add_object(1, 0, kind="key", color=colors[1])

        obj, _ = self.add_object(0, 0, kind="ball")

        self.place_agent(1, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_PickupDist(RoomGridLevel):
    """
    Pick up an object
    The object to pick up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)
    """

    def __init__(self, debug=False, seed=None):
        self.debug = debug
        super().__init__(
            num_rows = 1,
            num_cols = 1,
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        # Add 5 random objects in the room
        objs = self.add_distractors(num_distractors=5)
        self.place_agent(0, 0)
        obj = self._rand_elem(objs)
        type = obj.type
        color = obj.color

        select_by = self._rand_elem(["type", "color", "both"])
        if select_by == "color":
            type = None
        elif select_by == "type":
            color = None

        self.instrs = PickupInstr(ObjDesc(type, color), strict=self.debug)


class Level_PickupDistDebug(Level_PickupDist):
    """
    Same as PickupDist but the level stops when any object is picked
    """

    def __init__(self, seed=None):
        super().__init__(
            debug=True,
            seed=seed
        )


class Level_PickupAbove(RoomGridLevel):
    """
    Pick up an object (in the room above)
    This task requires to use the compass to be solved effectively.
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the top-middle room
        obj, pos = self.add_object(1, 0)
        # Make sure the two rooms are directly connected
        self.add_door(1, 1, 3, locked=False)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_OpenTwoDoors(RoomGridLevel):
    """
    Open door X, then open door Y
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self,
        first_color=None,
        second_color=None,
        strict=False,
        seed=None
    ):
        self.first_color = first_color
        self.second_color = second_color
        self.strict = strict

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        first_color = self.first_color
        if first_color is None:
            first_color = colors[0]
        second_color = self.second_color
        if second_color is None:
            second_color = colors[1]

        door1, _ = self.add_door(1, 1, 2, color=first_color, locked=False)
        door2, _ = self.add_door(1, 1, 0, color=second_color, locked=False)

        self.place_agent(1, 1)

        self.instrs = BeforeInstr(
            OpenInstr(ObjDesc(door1.type, door1.color), strict=self.strict),
            OpenInstr(ObjDesc(door2.type, door2.color))
        )


class Level_OpenTwoDoorsDebug(Level_OpenTwoDoors):
    """
    Same as OpenTwoDoors but the level stops when the second door is opened
    """

    def __init__(self,
        first_color=None,
        second_color=None,
        seed=None
    ):
        super().__init__(
            first_color,
            second_color,
            strict=True,
            seed=seed
        )


class Level_OpenRedBlueDoors(Level_OpenTwoDoors):
    """
    Open red door, then open blue door
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )


class Level_OpenRedBlueDoorsDebug(Level_OpenTwoDoorsDebug):
    """
    Same as OpenRedBlueDoors but the level stops when the blue door is opened
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )


class Level_FindObjS5(RoomGridLevel):
    """
    Pick up an object (in a random room)
    Rooms have a size of 5
    This level requires potentially exhaustive exploration
    """

    def __init__(self, room_size=5, seed=None):
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_FindObjS6(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 6
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            seed=seed
        )


class Level_FindObjS7(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 7
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )


class KeyCorridor(RoomGridLevel):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        seed=None
    ):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
            seed=seed,
        )

    def gen_mission(self):
        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_KeyCorridorS3R1(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=1,
            seed=seed
        )

class Level_KeyCorridorS3R2(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=2,
            seed=seed
        )

class Level_KeyCorridorS3R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS4R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS5R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS6R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            num_rows=3,
            seed=seed
        )

class Level_1RoomS8(RoomGridLevel):
    """
    Pick up the ball
    Rooms have a size of 8
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_object(0, 0, kind="ball")
        self.place_agent()
        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_1RoomS12(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 12
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=12,
            seed=seed
        )


class Level_1RoomS16(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 16
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=16,
            seed=seed
        )


class Level_1RoomS20(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 20
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=20,
            seed=seed
        )


class PutNext(RoomGridLevel):
    """
    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.
    """

    def __init__(
        self,
        room_size,
        objs_per_room,
        start_carrying=False,
        seed=None
    ):
        assert room_size >= 4
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room
        self.start_carrying = start_carrying

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        a = self._rand_elem(objs_l)
        b = self._rand_elem(objs_r)

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = a
            a = b
            b = t

        self.obj_a = a

        self.instrs = PutNextInstr(
            ObjDesc(a.type, a.color),
            ObjDesc(b.type, b.color)
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # If the agent starts off carrying the object
        if self.start_carrying:
            self.grid.set(*self.obj_a.init_pos, None)
            self.carrying = self.obj_a

        return obs


class Level_PutNextS4N1(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            objs_per_room=1,
            seed=seed
        )


class Level_PutNextS5N1(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=1,
            seed=seed
        )


class Level_PutNextS5N2(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            seed=seed
        )


class Level_PutNextS6N3(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            objs_per_room=3,
            seed=seed
        )


class Level_PutNextS7N4(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            objs_per_room=4,
            seed=seed
        )


class Level_PutNextS5N2Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            start_carrying=True,
            seed=seed
        )


class Level_PutNextS6N3Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            objs_per_room=3,
            start_carrying=True,
            seed=seed
        )


class Level_PutNextS7N4Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            objs_per_room=4,
            start_carrying=True,
            seed=seed
        )


class MoveTwoAcross(RoomGridLevel):
    """
    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.
    """

    def __init__(
        self,
        room_size,
        objs_per_room,
        seed=None
    ):
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        objs_l = self._rand_subset(objs_l, 2)
        objs_r = self._rand_subset(objs_r, 2)
        a = objs_l[0]
        b = objs_r[0]
        c = objs_r[1]
        d = objs_l[1]

        self.instrs = BeforeInstr(
            PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color)),
            PutNextInstr(ObjDesc(c.type, c.color), ObjDesc(d.type, d.color))
        )


class Level_MoveTwoAcrossS5N2(MoveTwoAcross):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            seed=seed
        )


class Level_MoveTwoAcrossS8N9(MoveTwoAcross):
    def __init__(self, seed=None):
        super().__init__(
            room_size=8,
            objs_per_room=9,
            seed=seed
        )


class OpenDoorsOrder(RoomGridLevel):
    """
    Open one or two doors in the order specified.
    """

    def __init__(
        self,
        num_doors,
        debug=False,
        seed=None
    ):
        assert num_doors >= 2
        self.num_doors = num_doors
        self.debug = debug

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, self.num_doors)
        doors = []
        for i in range(self.num_doors):
            door, _ = self.add_door(1, 1, color=colors[i], locked=False)
            doors.append(door)
        self.place_agent(1, 1)

        door1, door2 = self._rand_subset(doors, 2)
        desc1 = ObjDesc(door1.type, door1.color)
        desc2 = ObjDesc(door2.type, door2.color)

        mode = self._rand_int(0, 3)
        if mode == 0:
            self.instrs = OpenInstr(desc1, strict=self.debug)
        elif mode == 1:
            self.instrs = BeforeInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        elif mode == 2:
            self.instrs = AfterInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        else:
            assert False

class Level_OpenDoorsOrderN2(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=2,
            seed=seed
        )


class Level_OpenDoorsOrderN4(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=4,
            seed=seed
        )


class Level_OpenDoorsOrderN2Debug(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=2,
            debug=True,
            seed=seed
        )


class Level_OpenDoorsOrderN4Debug(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=4,
            debug=True,
            seed=seed
        )

for name, level in list(globals().items()):
    if name.startswith('Level_'):
        level.is_bonus = True

# Register the levels in this file
register_levels(__name__, globals())



================================================
FILE: babyai/levels/iclr19_levels.py
================================================
"""
Levels described in the ICLR 2019 submission.
"""

import gym
from .verifier import *
from .levelgen import *


class Level_GoToRedBallGrey(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        for dist in dists:
            dist.color = 'grey'

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBall(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBallNoDists(Level_GoToRedBall):
    """
    Go to the red ball. No distractors present.
    """

    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=0, seed=seed)


class Level_GoToObj(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=1)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToObjS4(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=4, seed=seed)


class Level_GoToObjS6(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=6, seed=seed)


class Level_GoToLocal(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, num_dists=8, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocalS5N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_dists=2, seed=seed)


class Level_GoToLocalS6N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=2, seed=seed)


class Level_GoToLocalS6N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=3, seed=seed)


class Level_GoToLocalS6N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=4, seed=seed)


class Level_GoToLocalS7N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=4, seed=seed)


class Level_GoToLocalS7N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=5, seed=seed)


class Level_GoToLocalS8N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=2, seed=seed)


class Level_GoToLocalS8N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=3, seed=seed)


class Level_GoToLocalS8N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=4, seed=seed)


class Level_GoToLocalS8N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=5, seed=seed)


class Level_GoToLocalS8N6(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=6, seed=seed)


class Level_GoToLocalS8N7(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=7, seed=seed)


class Level_PutNextLocal(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """

    def __init__(self, room_size=8, num_objs=8, seed=None):
        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PutNextLocalS5N3(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_objs=3, seed=seed)


class Level_PutNextLocalS6N4(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_objs=4, seed=seed)


class Level_GoTo(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=False,
        seed=None
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class Level_GoToOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(doors_open=True, seed=seed)


class Level_GoToObjMaze(Level_GoTo):
    """
    Go to an object, the object may be in another room. No distractors.
    """

    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=False, seed=seed)


class Level_GoToObjMazeOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=True, seed=seed)


class Level_GoToObjMazeS4R2(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, num_rows=2, num_cols=2, seed=seed)


class Level_GoToObjMazeS4(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, seed=seed)


class Level_GoToObjMazeS5(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=5, seed=seed)


class Level_GoToObjMazeS6(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=6, seed=seed)


class Level_GoToObjMazeS7(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=7, seed=seed)


class Level_GoToImpUnlock(RoomGridLevel):
    """
    Go to an object, which may be in a locked room.
    Competencies: Maze, GoTo, ImpUnlock
    No unblocking.
    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=2,
                        all_unique=False
                    )

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        # Add a single object to the locked room
        # The instruction requires going to an object matching that description
        obj, = self.add_distractors(id, jd, num_distractors=1, all_unique=False)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_Pickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnblockPickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room. The path may
    be blocked by one or more obstructors.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=20, all_unique=False)

        # Ensure that at least one object is not reachable without unblocking
        # Note: the selected object will still be reachable most of the time
        if self.check_objs_reachable(raise_exc=False):
            raise RejectSampling('all objects reachable')

        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_Open(RoomGridLevel):
    """
    Open a door, which may be in another room
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        # Collect a list of all the doors in the environment
        doors = []
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)

        door = self._rand_elem(doors)
        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_Unlock(RoomGridLevel):
    """
    Unlock a door.

    Competencies: Maze, Open, Unlock. No unblocking.
    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        # With 50% probability, ensure that the locked door is the only
        # door of that color
        if self._rand_bool():
            colors = list(filter(lambda c: c is not door.color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=3,
                        all_unique=False
                    )

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_PutNext(RoomGridLevel):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)
        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PickupLoc(LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False
        )


class Level_GoToSeq(LevelGen):
    """
    Sequencing of go-to-object commands.

    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None
    ):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto'],
            locked_room_prob=0,
            locations=False,
            unblocking=False
        )


class Level_GoToSeqS5R2(Level_GoToSeq):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_rows=2, num_cols=2, num_dists=4, seed=seed)


class Level_Synth(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            instr_kinds=['action'],
            locations=False,
            unblocking=True,
            implicit_unlock=False
        )


class Level_SynthS5R2(Level_Synth):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=2,
            num_cols=2,
            num_dists=7,
            seed=seed
        )


class Level_SynthLoc(LevelGen):
    """
    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            instr_kinds=['action'],
            locations=True,
            unblocking=True,
            implicit_unlock=False
        )


class Level_SynthSeq(LevelGen):
    """
    Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.
    No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            locations=True,
            unblocking=True,
            implicit_unlock=False
        )


class Level_MiniBossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25
        )


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )


class Level_BossLevelNoUnlock(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            locked_room_prob=0,
            implicit_unlock=False
        )


# Register the levels in this file
register_levels(__name__, globals())



================================================
FILE: babyai/levels/levelgen.py
================================================
import random
from collections import OrderedDict
from copy import deepcopy
import gym
from gym_minigrid.roomgrid import RoomGrid
from .verifier import *


class RejectSampling(Exception):
    """
    Exception used for rejection sampling
    """

    pass


class RoomGridLevel(RoomGrid):
    """
    Base for levels based on RoomGrid
    A level, given a random seed, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """

    def __init__(
        self,
        room_size=8,
        **kwargs
    ):
        super().__init__(
            room_size=room_size,
            **kwargs
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.instrs.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = self.room_size ** 2
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        num_navs = self.num_navs_needed(self.instrs)
        self.max_steps = num_navs * nav_time_maze

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we drop an object, we need to update its position in the environment
        if action == self.actions.drop:
            self.update_objs_poss()

        # If we've successfully completed the mission
        status = self.instrs.verify(action)

        if status == 'success':
            done = True
            reward = self._reward()
        elif status == 'failure':
            done = True
            reward = 0

        return obs, reward, done, info

    def update_objs_poss(self, instr=None):
        if instr is None:
            instr = self.instrs
        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr) or isinstance(instr, AfterInstr):
            self.update_objs_poss(instr.instr_a)
            self.update_objs_poss(instr.instr_b)
        else:
            instr.update_objs_poss()

    def _gen_grid(self, width, height):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                super()._gen_grid(width, height)

                # Generate the mission
                self.gen_mission()

                # Validate the instructions
                self.validate_instrs(self.instrs)

            except RecursionError as error:
                print('Timeout during mission generation:', error)
                continue

            except RejectSampling as error:
                #print('Sampling rejected:', error)
                continue

            break

        # Generate the surface form for the instructions
        self.surface = self.instrs.surface(self)
        self.mission = self.surface

    def validate_instrs(self, instr):
        """
        Perform some validation on the generated instructions
        """
        # Gather the colors of locked doors
        if hasattr(self, 'unblocking') and self.unblocking:
            colors_of_locked_doors = []
            for i in range(self.num_cols):
                for j in range(self.num_rows):
                    room = self.get_room(i, j)
                    for door in room.doors:
                        if door and door.is_locked:
                            colors_of_locked_doors.append(door.color)

        if isinstance(instr, PutNextInstr):
            # Resolve the objects referenced by the instruction
            instr.reset_verifier(self)

            # Check that the objects are not already next to each other
            if set(instr.desc_move.obj_set).intersection(
                    set(instr.desc_fixed.obj_set)):
                raise RejectSampling(
                    "there are objects that match both lhs and rhs of PutNext")
            if instr.objs_next():
                raise RejectSampling('objs already next to each other')

            # Check that we are not asking to move an object next to itself
            move = instr.desc_move
            fixed = instr.desc_fixed
            if len(move.obj_set) == 1 and len(fixed.obj_set) == 1:
                if move.obj_set[0] is fixed.obj_set[0]:
                    raise RejectSampling('cannot move an object next to itself')

        if isinstance(instr, ActionInstr):
            if not hasattr(self, 'unblocking') or not self.unblocking:
                return
            # TODO: either relax this a bit or make the bot handle this super corner-y scenarios
            # Check that the instruction doesn't involve a key that matches the color of a locked door
            potential_objects = ('desc', 'desc_move', 'desc_fixed')
            for attr in potential_objects:
                if hasattr(instr, attr):
                    obj = getattr(instr, attr)
                    if obj.type == 'key' and obj.color in colors_of_locked_doors:
                        raise RejectSampling('cannot do anything with/to a key that can be used to open a door')
            return

        if isinstance(instr, SeqInstr):
            self.validate_instrs(instr.instr_a)
            self.validate_instrs(instr.instr_b)
            return

        assert False, "unhandled instruction type"

    def gen_mission(self):
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError

    @property
    def level_name(self):
        return self.__class__.level_name

    @property
    def gym_id(self):
        return self.__class__.gym_id

    def num_navs_needed(self, instr):
        """
        Compute the maximum number of navigations needed to perform
        a simple or complex instruction
        """

        if isinstance(instr, PutNextInstr):
            return 2

        if isinstance(instr, ActionInstr):
            return 1

        if isinstance(instr, SeqInstr):
            na = self.num_navs_needed(instr.instr_a)
            nb = self.num_navs_needed(instr.instr_b)
            return na + nb

    def open_all_doors(self):
        """
        Open all the doors in the maze
        """

        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        door.is_open = True

    def check_objs_reachable(self, raise_exc=True):
        """
        Check that all objects are reachable from the agent's starting
        position without requiring any other object to be moved
        (without unblocking)
        """

        # Reachable positions
        reachable = set()

        # Work list
        stack = [self.agent_pos]

        while len(stack) > 0:
            i, j = stack.pop()

            if i < 0 or i >= self.grid.width or j < 0 or j >= self.grid.height:
                continue

            if (i, j) in reachable:
                continue

            # This position is reachable
            reachable.add((i, j))

            cell = self.grid.get(i, j)

            # If there is something other than a door in this cell, it
            # blocks reachability
            if cell and cell.type != 'door':
                continue

            # Visit the horizontal and vertical neighbors
            stack.append((i+1, j))
            stack.append((i-1, j))
            stack.append((i, j+1))
            stack.append((i, j-1))

        # Check that all objects are reachable
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)

                if not cell or cell.type == 'wall':
                    continue

                if (i, j) not in reachable:
                    if not raise_exc:
                        return False
                    raise RejectSampling('unreachable object at ' + str((i, j)))

        # All objects reachable
        return True


class LevelGen(RoomGridLevel):
    """
    Level generator which attempts to produce every possible sentence in
    the baby language as an instruction.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        locked_room_prob=0.5,
        locations=True,
        unblocking=True,
        implicit_unlock=True,
        action_kinds=['goto', 'pickup', 'open', 'putnext'],
        instr_kinds=['action', 'and', 'seq'],
        seed=None
    ):
        self.num_dists = num_dists
        self.locked_room_prob = locked_room_prob
        self.locations = locations
        self.unblocking = unblocking
        self.implicit_unlock = implicit_unlock
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds

        self.locked_room = None

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            seed=seed
        )

    def gen_mission(self):
        if self._rand_float(0, 1) < self.locked_room_prob:
            self.add_locked_room()

        self.connect_all()

        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break

        # If no unblocking required, make sure all objects are
        # reachable without unblocking
        if not self.unblocking:
            self.check_objs_reachable()

        # Generate random instructions
        self.instrs = self.rand_instr(
            action_kinds=self.action_kinds,
            instr_kinds=self.instr_kinds
        )

    def add_locked_room(self):
        # Until we've successfully added a locked room
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            door_idx = self._rand_int(0, 4)
            self.locked_room = self.get_room(i, j)

            # Don't add a locked door in an external wall
            if self.locked_room.neighbors[door_idx] is None:
                continue

            door, _ = self.add_door(
                i, j,
                door_idx,
                locked=True
            )

            # Done adding locked room
            break

        # Until we find a room to put the key
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            key_room = self.get_room(i, j)

            if key_room is self.locked_room:
                continue

            self.add_object(i, j, 'key', door.color)
            break

    def rand_obj(self, types=OBJ_TYPES, colors=COLOR_NAMES, max_tries=100):
        """
        Generate a random object descriptor
        """

        num_tries = 0

        # Keep trying until we find a matching object
        while True:
            if num_tries > max_tries:
                raise RecursionError('failed to find suitable object')
            num_tries += 1

            color = self._rand_elem([None, *colors])
            type = self._rand_elem(types)

            loc = None
            if self.locations and self._rand_bool():
                loc = self._rand_elem(LOC_NAMES)

            desc = ObjDesc(type, color, loc)

            # Find all objects matching the descriptor
            objs, poss = desc.find_matching_objs(self)

            # The description must match at least one object
            if len(objs) == 0:
                continue

            # If no implicit unlocking is required
            if not self.implicit_unlock and self.locked_room:
                # Check that at least one object is not in the locked room
                pos_not_locked = list(filter(
                    lambda p: not self.locked_room.pos_inside(*p),
                    poss
                ))

                if len(pos_not_locked) == 0:
                    continue

            # Found a valid object description
            return desc

    def rand_instr(
        self,
        action_kinds,
        instr_kinds,
        depth=0
    ):
        """
        Generate random instructions
        """

        kind = self._rand_elem(instr_kinds)

        if kind == 'action':
            action = self._rand_elem(action_kinds)

            if action == 'goto':
                return GoToInstr(self.rand_obj())
            elif action == 'pickup':
                return PickupInstr(self.rand_obj(types=OBJ_TYPES_NOT_DOOR))
            elif action == 'open':
                return OpenInstr(self.rand_obj(types=['door']))
            elif action == 'putnext':
                return PutNextInstr(
                    self.rand_obj(types=OBJ_TYPES_NOT_DOOR),
                    self.rand_obj()
                )

            assert False

        elif kind == 'and':
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth+1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth+1
            )
            return AndInstr(instr_a, instr_b)

        elif kind == 'seq':
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth+1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth+1
            )

            kind = self._rand_elem(['before', 'after'])

            if kind == 'before':
                return BeforeInstr(instr_a, instr_b)
            elif kind == 'after':
                return AfterInstr(instr_a, instr_b)

            assert False

        assert False


# Dictionary of levels, indexed by name, lexically sorted
level_dict = OrderedDict()


def register_levels(module_name, globals):
    """
    Register OpenAI gym environments for all levels in a file
    """

    # Iterate through global names
    for global_name in sorted(list(globals.keys())):
        if not global_name.startswith('Level_'):
            continue

        level_name = global_name.split('Level_')[-1]
        level_class = globals[global_name]

        # Register the levels with OpenAI Gym
        gym_id = 'BabyAI-%s-v0' % (level_name)
        entry_point = '%s:%s' % (module_name, global_name)
        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        # Add the level to the dictionary
        level_dict[level_name] = level_class

        # Store the name and gym id on the level class
        level_class.level_name = level_name
        level_class.gym_id = gym_id


def test():
    for idx, level_name in enumerate(level_dict.keys()):
        print('Level %s (%d/%d)' % (level_name, idx+1, len(level_dict)))

        level = level_dict[level_name]

        # Run the mission for a few episodes
        rng = random.Random(0)
        num_episodes = 0
        for i in range(0, 15):
            mission = level(seed=i)

            # Check that the surface form was generated
            assert isinstance(mission.surface, str)
            assert len(mission.surface) > 0
            obs = mission.reset()
            assert obs['mission'] == mission.surface

            # Reduce max_steps because otherwise tests take too long
            mission.max_steps = min(mission.max_steps, 200)

            # Check for some known invalid patterns in the surface form
            import re
            surface = mission.surface
            assert not re.match(r".*pick up the [^ ]*door.*", surface), surface

            while True:
                action = rng.randint(0, mission.action_space.n - 1)
                obs, reward, done, info = mission.step(action)
                if done:
                    obs = mission.reset()
                    break

            num_episodes += 1

        # The same seed should always yield the same mission
        m0 = level(seed=0)
        m1 = level(seed=0)
        grid1 = m0.unwrapped.grid
        grid2 = m1.unwrapped.grid
        assert grid1 == grid2
        assert m0.surface == m1.surface

    # Check that gym environment names were registered correctly
    gym.make('BabyAI-1RoomS8-v0')
    gym.make('BabyAI-BossLevel-v0')



================================================
FILE: babyai/levels/test_levels.py
================================================
"""
Regression tests.
"""

import numpy as np

import gym
from .verifier import *
from .levelgen import *
from gym_minigrid.minigrid import *


class Level_TestGoToBlocked(RoomGridLevel):
    """
    Go to a yellow ball that is blocked with a lot of red balls.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        obj = Ball('yellow')
        self.grid.set(1, 1, obj)
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if (i, j) not in [(1 ,1), (3, 3)]:
                    self.place_obj(Ball('red'), (i, j), (1, 1))
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))



class Level_TestPutNextToBlocked(RoomGridLevel):
    """
    Pick up a yellow ball and put it next to a blocked blue ball.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        obj1 = Ball('yellow')
        obj2 = Ball('blue')
        self.place_obj(obj1, (4, 4), (1, 1))
        self.place_obj(obj2, (1, 1), (1, 1))
        self.grid.set(1, 2, Ball('red'))
        self.grid.set(2, 1, Ball('red'))
        self.instrs = PutNextInstr(ObjDesc(obj1.type, obj1.color),
                                   ObjDesc(obj2.type, obj2.color))


class Level_TestPutNextToCloseToDoor1(RoomGridLevel):
    """
    The yellow ball must be put near the blue ball.
    But blue ball is right next to a door.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        door, pos = self.add_door(0, 0, None, 'red', False)
        self.obj1 = Ball('yellow')
        self.obj2 = Ball('blue')
        self.place_obj(self.obj1, (4, 4), (1, 1))
        self.place_obj(self.obj2, (pos[0], pos[1] + 1), (1, 1))
        self.instrs = BeforeInstr(
            OpenInstr(ObjDesc('door', door.color)),
            PutNextInstr(ObjDesc(self.obj1.type, self.obj1.color),
                         ObjDesc(self.obj2.type, self.obj2.color)))


class Level_TestPutNextToCloseToDoor2(Level_TestPutNextToCloseToDoor1):
    """
    The yellow ball must be put near the blue ball.
    But blue ball is right next to a door.
    """

    def gen_mission(self):
        super().gen_mission()
        self.instrs = PutNextInstr(ObjDesc(self.obj1.type, self.obj1.color),
                                   ObjDesc(self.obj2.type, self.obj2.color))



class Level_TestPutNextToIdentical(RoomGridLevel):
    """
    Test that the agent does not endlessly hesitate between
    two identical objects.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        self.place_obj(Box('yellow'), (1, 1), (1, 1))
        self.place_obj(Ball('blue'), (4, 4), (1, 1))
        self.place_obj(Ball('red'), (2, 2), (1, 1))
        instr1 = PutNextInstr(ObjDesc('ball', 'blue'),
                              ObjDesc('box', 'yellow'))
        instr2 = PutNextInstr(ObjDesc('box', 'yellow'),
                              ObjDesc('ball', None))
        self.instrs = BeforeInstr(instr1, instr2)


class Level_TestUnblockingLoop(RoomGridLevel):
    """Test that unblocking does not results into an infinite loop."""

    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([15, 4])
        self.agent_dir = 2
        door, pos = self.add_door(0, 0, 1, 'red', False)
        door, pos = self.add_door(0, 1, 0, 'red', False)
        door, pos = self.add_door(1, 1, 3, 'blue', False)
        self.place_obj(Box('yellow'), (9, 1), (1, 1))
        self.place_obj(Ball('blue'), (5, 3), (1, 1))
        self.place_obj(Ball('yellow'), (6, 2), (1, 1))
        self.place_obj(Key('blue'), (15, 15), (1, 1))
        put = PutNextInstr(ObjDesc('key', 'blue'), ObjDesc('door', 'blue'))
        goto1 = GoToInstr(ObjDesc('ball', 'yellow'))
        goto2 = GoToInstr(ObjDesc('box', 'yellow'))
        self.instrs = BeforeInstr(put, AndInstr(goto1, goto2))


class Level_TestPutNextCloseToDoor(RoomGridLevel):
    """Test putting next when there is door where the object should be put."""

    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([5, 10])
        self.agent_dir = 2
        door, pos1 = self.add_door(0, 0, 1, 'red', False)
        door, pos2 = self.add_door(0, 1, 0, 'red', False)
        door, pos3 = self.add_door(1, 1, 3, 'blue', False)
        self.place_obj(Ball('blue'), (pos1[0], pos1[1] - 1), (1, 1))
        self.place_obj(Ball('blue'), (pos1[0], pos1[1] - 2), (1, 1))
        if pos1[0] - 1 >= 1:
            self.place_obj(Box('green'), (pos1[0] - 1, pos1[1] - 1), (1, 1))
        if pos1[0] + 1 < 8:
            self.place_obj(Box('green'), (pos1[0] + 1, pos1[1] - 1), (1, 1))
        self.place_obj(Box('yellow'), (3, 15), (1, 1))
        self.instrs = PutNextInstr(ObjDesc('box', 'yellow'), ObjDesc('ball', 'blue'))


class Level_TestLotsOfBlockers(RoomGridLevel):
    """
    Test that the agent does not endlessly hesitate between
    two identical objects.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=8,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([5, 5])
        self.agent_dir = 0
        self.place_obj(Box('yellow'), (2, 1), (1, 1))
        self.place_obj(Box('yellow'), (2, 2), (1, 1))
        self.place_obj(Box('yellow'), (2, 3), (1, 1))
        self.place_obj(Box('yellow'), (3, 4), (1, 1))
        self.place_obj(Box('yellow'), (2, 6), (1, 1))
        self.place_obj(Box('yellow'), (1, 3), (1, 1))
        self.place_obj(Ball('blue'), (1, 2), (1, 1))
        self.place_obj(Ball('red'), (3, 6), (1, 1))
        self.instrs = PutNextInstr(ObjDesc('ball', 'red'),
                                   ObjDesc('ball', 'blue'))


register_levels(__name__, globals())



================================================
FILE: babyai/levels/verifier.py
================================================
import os
import numpy as np
from enum import Enum
from gym_minigrid.minigrid import COLOR_NAMES, DIR_TO_VEC

# Object types we are allowed to describe in language
OBJ_TYPES = ['box', 'ball', 'key', 'door']

# Object types we are allowed to describe in language
OBJ_TYPES_NOT_DOOR = list(filter(lambda t: t != 'door', OBJ_TYPES))

# Locations are all relative to the agent's starting position
LOC_NAMES = ['left', 'right', 'front', 'behind']

# Environment flag to indicate that done actions should be
# used by the verifier
use_done_actions = os.environ.get('BABYAI_DONE_ACTIONS', False)


def dot_product(v1, v2):
    """
    Compute the dot product of the vectors v1 and v2.
    """

    return sum([i * j for i, j in zip(v1, v2)])


def pos_next_to(pos_a, pos_b):
    """
    Test if two positions are next to each other.
    The positions have to line up either horizontally or vertically,
    but positions that are diagonally adjacent are not counted.
    """

    xa, ya = pos_a
    xb, yb = pos_b
    d = abs(xa - xb) + abs(ya - yb)
    return d == 1


class ObjDesc:
    """
    Description of a set of objects in an environment
    """

    def __init__(self, type, color=None, loc=None):
        assert type in [None, *OBJ_TYPES], type
        assert color in [None, *COLOR_NAMES], color
        assert loc in [None, *LOC_NAMES], loc

        self.color = color
        self.type = type
        self.loc = loc

        # Set of objects possibly matching the description
        self.obj_set = []

        # Set of initial object positions
        self.obj_poss = []

    def __repr__(self):
        return "{} {} {}".format(self.color, self.type, self.loc)

    def surface(self, env):
        """
        Generate a natural language representation of the object description
        """

        self.find_matching_objs(env)
        assert len(self.obj_set) > 0, "no object matching description"

        if self.type:
            s = str(self.type)
        else:
            s = 'object'

        if self.color:
            s = self.color + ' ' + s

        if self.loc:
            if self.loc == 'front':
                s = s + ' in front of you'
            elif self.loc == 'behind':
                s = s + ' behind you'
            else:
                s = s + ' on your ' + self.loc

        # Singular vs plural
        if len(self.obj_set) > 1:
            s = 'a ' + s
        else:
            s = 'the ' + s

        return s

    def find_matching_objs(self, env, use_location=True):
        """
        Find the set of objects matching the description and their positions.
        When use_location is False, we only update the positions of already tracked objects, without taking into account
        the location of the object. e.g. A ball that was on "your right" initially will still be tracked as being "on
        your right" when you move.
        """

        if use_location:
            self.obj_set = []
            # otherwise we keep the same obj_set

        self.obj_poss = []

        agent_room = env.room_from_pos(*env.agent_pos)

        for i in range(env.grid.width):
            for j in range(env.grid.height):
                cell = env.grid.get(i, j)
                if cell is None:
                    continue

                if not use_location:
                    # we should keep tracking the same objects initially tracked only
                    already_tracked = any([cell is obj for obj in self.obj_set])
                    if not already_tracked:
                        continue

                # Check if object's type matches description
                if self.type is not None and cell.type != self.type:
                    continue

                # Check if object's color matches description
                if self.color is not None and cell.color != self.color:
                    continue

                # Check if object's position matches description
                if use_location and self.loc in ["left", "right", "front", "behind"]:
                    # Locations apply only to objects in the same room
                    # the agent starts in
                    if not agent_room.pos_inside(i, j):
                        continue

                    # Direction from the agent to the object
                    v = (i - env.agent_pos[0], j - env.agent_pos[1])

                    # (d1, d2) is an oriented orthonormal basis
                    d1 = DIR_TO_VEC[env.agent_dir]
                    d2 = (-d1[1], d1[0])

                    # Check if object's position matches with location
                    pos_matches = {
                        "left": dot_product(v, d2) < 0,
                        "right": dot_product(v, d2) > 0,
                        "front": dot_product(v, d1) > 0,
                        "behind": dot_product(v, d1) < 0
                    }

                    if not (pos_matches[self.loc]):
                        continue

                if use_location:
                    self.obj_set.append(cell)
                self.obj_poss.append((i, j))

        return self.obj_set, self.obj_poss


class Instr:
    """
    Base class for all instructions in the baby language
    """

    def __init__(self):
        self.env = None

    def surface(self, env):
        """
        Produce a natural language representation of the instruction
        """

        raise NotImplementedError

    def reset_verifier(self, env):
        """
        Must be called at the beginning of the episode
        """

        self.env = env

    def verify(self, action):
        """
        Verify if the task described by the instruction is incomplete,
        complete with success or failed. The return value is a string,
        one of: 'success', 'failure' or 'continue'.
        """

        raise NotImplementedError

    def update_objs_poss(self):
        """
        Update the position of objects present in the instruction if needed
        """
        potential_objects = ('desc', 'desc_move', 'desc_fixed')
        for attr in potential_objects:
            if hasattr(self, attr):
                getattr(self, attr).find_matching_objs(self.env, use_location=False)


class ActionInstr(Instr):
    """
    Base class for all action instructions (clauses)
    """

    def __init__(self):
        super().__init__()

        # Indicates that the action was completed on the last step
        self.lastStepMatch = False

    def verify(self, action):
        """
        Verifies actions, with and without the done action.
        """

        if not use_done_actions:
            return self.verify_action(action)

        if action == self.env.actions.done:
            if self.lastStepMatch:
                return 'success'
            return 'failure'

        res = self.verify_action(action)
        self.lastStepMatch = (res == 'success')

    def verify_action(self):
        """
        Each action instruction class should implement this method
        to verify the action.
        """

        raise NotImplementedError


class OpenInstr(ActionInstr):
    def __init__(self, obj_desc, strict=False):
        super().__init__()
        assert obj_desc.type == 'door'
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return 'open ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # Only verify when the toggle action is performed
        if action != self.env.actions.toggle:
            return 'continue'

        # Get the contents of the cell in front of the agent
        front_cell = self.env.grid.get(*self.env.front_pos)

        for door in self.desc.obj_set:
            if front_cell and front_cell is door and door.is_open:
                return 'success'

        # If in strict mode and the wrong door is opened, failure
        if self.strict:
            if front_cell and front_cell.type == 'door':
                return 'failure'

        return 'continue'


class GoToInstr(ActionInstr):
    """
    Go next to (and look towards) an object matching a given description
    eg: go to the door
    """

    def __init__(self, obj_desc):
        super().__init__()
        self.desc = obj_desc

    def surface(self, env):
        return 'go to ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # For each object position
        for pos in self.desc.obj_poss:
            # If the agent is next to (and facing) the object
            if np.array_equal(pos, self.env.front_pos):
                return 'success'

        return 'continue'


class PickupInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, obj_desc, strict=False):
        super().__init__()
        assert obj_desc.type != 'door'
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return 'pick up ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Object previously being carried
        self.preCarrying = None

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # To keep track of what was carried at the last time step
        preCarrying = self.preCarrying
        self.preCarrying = self.env.carrying

        # Only verify when the pickup action is performed
        if action != self.env.actions.pickup:
            return 'continue'

        for obj in self.desc.obj_set:
            if preCarrying is None and self.env.carrying is obj:
                return 'success'

        # If in strict mode and the wrong door object is picked up, failure
        if self.strict:
            if self.env.carrying:
                return 'failure'

        self.preCarrying = self.env.carrying

        return 'continue'


class PutNextInstr(ActionInstr):
    """
    Put an object next to another object
    eg: put the red ball next to the blue key
    """

    def __init__(self, obj_move, obj_fixed, strict=False):
        super().__init__()
        assert obj_move.type != 'door'
        self.desc_move = obj_move
        self.desc_fixed = obj_fixed
        self.strict = strict

    def surface(self, env):
        return 'put ' + self.desc_move.surface(env) + ' next to ' + self.desc_fixed.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Object previously being carried
        self.preCarrying = None

        # Identify set of possible matching objects in the environment
        self.desc_move.find_matching_objs(env)
        self.desc_fixed.find_matching_objs(env)

    def objs_next(self):
        """
        Check if the objects are next to each other
        This is used for rejection sampling
        """

        for obj_a in self.desc_move.obj_set:
            pos_a = obj_a.cur_pos

            for pos_b in self.desc_fixed.obj_poss:
                if pos_next_to(pos_a, pos_b):
                    return True
        return False

    def verify_action(self, action):
        # To keep track of what was carried at the last time step
        preCarrying = self.preCarrying
        self.preCarrying = self.env.carrying

        # In strict mode, picking up the wrong object fails
        if self.strict:
            if action == self.env.actions.pickup and self.env.carrying:
                return 'failure'

        # Only verify when the drop action is performed
        if action != self.env.actions.drop:
            return 'continue'

        for obj_a in self.desc_move.obj_set:
            if preCarrying is not obj_a:
                continue

            pos_a = obj_a.cur_pos

            for pos_b in self.desc_fixed.obj_poss:
                if pos_next_to(pos_a, pos_b):
                    return 'success'

        return 'continue'


class SeqInstr(Instr):
    """
    Base class for sequencing instructions (before, after, and)
    """

    def __init__(self, instr_a, instr_b, strict=False):
        assert isinstance(instr_a, ActionInstr) or isinstance(instr_a, AndInstr)
        assert isinstance(instr_b, ActionInstr) or isinstance(instr_b, AndInstr)
        self.instr_a = instr_a
        self.instr_b = instr_b
        self.strict = strict


class BeforeInstr(SeqInstr):
    """
    Sequence two instructions in order:
    eg: go to the red door then pick up the blue ball
    """

    def surface(self, env):
        return self.instr_a.surface(env) + ', then ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.a_done == 'success':
            self.b_done = self.instr_b.verify(action)

            if self.b_done == 'failure':
                return 'failure'

            if self.b_done == 'success':
                return 'success'
        else:
            self.a_done = self.instr_a.verify(action)
            if self.a_done == 'failure':
                return 'failure'

            if self.a_done == 'success':
                return self.verify(action)

            # In strict mode, completing b first means failure
            if self.strict:
                if self.instr_b.verify(action) == 'success':
                    return 'failure'

        return 'continue'


class AfterInstr(SeqInstr):
    """
    Sequence two instructions in reverse order:
    eg: go to the red door after you pick up the blue ball
    """

    def surface(self, env):
        return self.instr_a.surface(env) + ' after you ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.b_done == 'success':
            self.a_done = self.instr_a.verify(action)

            if self.a_done == 'success':
                return 'success'

            if self.a_done == 'failure':
                return 'failure'
        else:
            self.b_done = self.instr_b.verify(action)
            if self.b_done == 'failure':
                return 'failure'

            if self.b_done == 'success':
                return self.verify(action)

            # In strict mode, completing a first means failure
            if self.strict:
                if self.instr_a.verify(action) == 'success':
                    return 'failure'

        return 'continue'


class AndInstr(SeqInstr):
    """
    Conjunction of two actions, both can be completed in any other
    eg: go to the red door and pick up the blue ball
    """

    def __init__(self, instr_a, instr_b, strict=False):
        assert isinstance(instr_a, ActionInstr)
        assert isinstance(instr_b, ActionInstr)
        super().__init__(instr_a, instr_b, strict)

    def surface(self, env):
        return self.instr_a.surface(env) + ' and ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.a_done != 'success':
            self.a_done = self.instr_a.verify(action)

        if self.b_done != 'success':
            self.b_done = self.instr_b.verify(action)

        if use_done_actions and action is self.env.actions.done:
            if self.a_done == 'failure' and self.b_done == 'failure':
                return 'failure'

        if self.a_done == 'success' and self.b_done == 'success':
            return 'success'

        return 'continue'



================================================
FILE: babyai/rl/__init__.py
================================================
from babyai.rl.algos import PPOAlgo
from babyai.rl.utils import DictList
from babyai.rl.model import ACModel, RecurrentACModel



================================================
FILE: babyai/rl/format.py
================================================
import torch

def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)


================================================
FILE: babyai/rl/LICENSE
================================================
MIT License

Copyright (c) 2018 Lucas Willems

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



================================================
FILE: babyai/rl/model.py
================================================
from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass


================================================
FILE: babyai/rl/algos/__init__.py
================================================
from babyai.rl.algos.ppo import PPOAlgo



================================================
FILE: babyai/rl/algos/base.py
================================================
from abc import ABC, abstractmethod
import torch
import numpy

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass



================================================
FILE: babyai/rl/algos/ppo.py
================================================
import numpy
import torch
import torch.nn.functional as F


from babyai.rl.algos.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss

                    model_results = self.acmodel(sb.obs, memory * sb.mask)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes



================================================
FILE: babyai/rl/utils/__init__.py
================================================
from babyai.rl.utils.dictlist import DictList
from babyai.rl.utils.penv import ParallelEnv



================================================
FILE: babyai/rl/utils/dictlist.py
================================================
class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


================================================
FILE: babyai/rl/utils/penv.py
================================================
from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()


================================================
FILE: babyai/rl/utils/supervised_losses.py
================================================
import torch

import torch.nn.functional as F
import numpy
from babyai.rl.utils import DictList

# dictionary that defines what head is required for each extra info used for auxiliary supervision
required_heads = {'seen_state': 'binary',
                  'see_door': 'binary',
                  'see_obj': 'binary',
                  'obj_in_instr': 'binary',
                  'in_front_of_what': 'multiclass9',  # multi class classifier with 9 possible classes
                  'visit_proportion': 'continuous01',  # continous regressor with outputs in [0, 1]
                  'bot_action': 'binary'
                  }

class ExtraInfoCollector:
    '''
    This class, used in rl.algos.base, allows connecting the extra information from the environment, and the
    corresponding predictions using the specific heads in the model. It transforms them so that they are easy to use
    to evaluate losses
    '''
    def __init__(self, aux_info, shape, device):
        self.aux_info = aux_info
        self.shape = shape
        self.device = device

        self.collected_info = dict()
        self.extra_predictions = dict()
        for info in self.aux_info:
            self.collected_info[info] = torch.zeros(*shape, device=self.device)
            if required_heads[info] == 'binary' or required_heads[info].startswith('continuous'):
                # we predict one number only
                self.extra_predictions[info] = torch.zeros(*shape, 1, device=self.device)
            elif required_heads[info].startswith('multiclass'):
                # means that this is a multi-class classification and we need to predict the whole proba distr
                n_classes = int(required_heads[info].replace('multiclass', ''))
                self.extra_predictions[info] = torch.zeros(*shape, n_classes, device=self.device)
            else:
                raise ValueError("{} not supported".format(required_heads[info]))

    def process(self, env_info):
        # env_info is now a tuple of dicts
        env_info = [{k: v for k, v in dic.items() if k in self.aux_info} for dic in env_info]
        env_info = {k: [env_info[_][k] for _ in range(len(env_info))] for k in env_info[0].keys()}
        # env_info is now a dict of lists
        return env_info

    def fill_dictionaries(self, index, env_info, extra_predictions):
        for info in self.aux_info:
            dtype = torch.long if required_heads[info].startswith('multiclass') else torch.float
            self.collected_info[info][index] = torch.tensor(env_info[info], dtype=dtype, device=self.device)
            self.extra_predictions[info][index] = extra_predictions[info]

    def end_collection(self, exps):
        collected_info = dict()
        extra_predictions = dict()
        for info in self.aux_info:
            # T x P -> P x T -> P * T
            collected_info[info] = self.collected_info[info].transpose(0, 1).reshape(-1)
            if required_heads[info] == 'binary' or required_heads[info].startswith('continuous'):
                # T x P x 1 -> P x T x 1 -> P * T
                extra_predictions[info] = self.extra_predictions[info].transpose(0, 1).reshape(-1)
            elif type(required_heads[info]) == int:
                # T x P x k -> P x T x k -> (P * T) x k
                k = required_heads[info]  # number of classes
                extra_predictions[info] = self.extra_predictions[info].transpose(0, 1).reshape(-1, k)
        # convert the dicts to DictLists, and add them to the exps DictList.
        exps.collected_info = DictList(collected_info)
        exps.extra_predictions = DictList(extra_predictions)

        return exps


class SupervisedLossUpdater:
    '''
    This class, used by PPO, allows the evaluation of the supervised loss when using extra information from the
    environment. It also handles logging accuracies/L2 distances/etc...
    '''
    def __init__(self, aux_info, supervised_loss_coef, recurrence, device):
        self.aux_info = aux_info
        self.supervised_loss_coef = supervised_loss_coef
        self.recurrence = recurrence
        self.device = device

        self.log_supervised_losses = []
        self.log_supervised_accuracies = []
        self.log_supervised_L2_losses = []
        self.log_supervised_prevalences = []

        self.batch_supervised_loss = 0
        self.batch_supervised_accuracy = 0
        self.batch_supervised_L2_loss = 0
        self.batch_supervised_prevalence = 0

    def init_epoch(self):
        self.log_supervised_losses = []
        self.log_supervised_accuracies = []
        self.log_supervised_L2_losses = []
        self.log_supervised_prevalences = []

    def init_batch(self):
        self.batch_supervised_loss = 0
        self.batch_supervised_accuracy = 0
        self.batch_supervised_L2_loss = 0
        self.batch_supervised_prevalence = 0

    def eval_subbatch(self, extra_predictions, sb):
        supervised_loss = torch.tensor(0., device=self.device)
        supervised_accuracy = torch.tensor(0., device=self.device)
        supervised_L2_loss = torch.tensor(0., device=self.device)
        supervised_prevalence = torch.tensor(0., device=self.device)

        binary_classification_tasks = 0
        classification_tasks = 0
        regression_tasks = 0

        for pos, info in enumerate(self.aux_info):
            coef = self.supervised_loss_coef[pos]
            pred = extra_predictions[info]
            target = dict.__getitem__(sb.collected_info, info)
            if required_heads[info] == 'binary':
                binary_classification_tasks += 1
                classification_tasks += 1
                supervised_loss += coef * F.binary_cross_entropy_with_logits(pred.reshape(-1), target)
                supervised_accuracy += ((pred.reshape(-1) > 0).float() == target).float().mean()
                supervised_prevalence += target.mean()
            elif required_heads[info].startswith('continuous'):
                regression_tasks += 1
                mse = F.mse_loss(pred.reshape(-1), target)
                supervised_loss += coef * mse
                supervised_L2_loss += mse
            elif required_heads[info].startswith('multiclass'):
                classification_tasks += 1
                supervised_accuracy += (pred.argmax(1).float() == target).float().mean()
                supervised_loss += coef * F.cross_entropy(pred, target.long())
            else:
                raise ValueError("{} not supported".format(required_heads[info]))
        if binary_classification_tasks > 0:
            supervised_prevalence /= binary_classification_tasks
        else:
            supervised_prevalence = torch.tensor(-1)
        if classification_tasks > 0:
            supervised_accuracy /= classification_tasks
        else:
            supervised_accuracy = torch.tensor(-1)
        if regression_tasks > 0:
            supervised_L2_loss /= regression_tasks
        else:
            supervised_L2_loss = torch.tensor(-1)

        self.batch_supervised_loss += supervised_loss.item()
        self.batch_supervised_accuracy += supervised_accuracy.item()
        self.batch_supervised_L2_loss += supervised_L2_loss.item()
        self.batch_supervised_prevalence += supervised_prevalence.item()

        return supervised_loss

    def update_batch_values(self):
        self.batch_supervised_loss /= self.recurrence
        self.batch_supervised_accuracy /= self.recurrence
        self.batch_supervised_L2_loss /= self.recurrence
        self.batch_supervised_prevalence /= self.recurrence

    def update_epoch_logs(self):
        self.log_supervised_losses.append(self.batch_supervised_loss)
        self.log_supervised_accuracies.append(self.batch_supervised_accuracy)
        self.log_supervised_L2_losses.append(self.batch_supervised_L2_loss)
        self.log_supervised_prevalences.append(self.batch_supervised_prevalence)

    def end_training(self, logs):
        logs["supervised_loss"] = numpy.mean(self.log_supervised_losses)
        logs["supervised_accuracy"] = numpy.mean(self.log_supervised_accuracies)
        logs["supervised_L2_loss"] = numpy.mean(self.log_supervised_L2_losses)
        logs["supervised_prevalence"] = numpy.mean(self.log_supervised_prevalences)

        return logs



================================================
FILE: babyai/utils/__init__.py
================================================
import os
import random
import numpy
import torch
from babyai.utils.agent import load_agent, ModelAgent, DemoAgent, BotAgent
from babyai.utils.demos import (
    load_demos, save_demos, synthesize_demos, get_demos_path)
from babyai.utils.format import ObssPreprocessor, IntObssPreprocessor, get_vocab_path
from babyai.utils.log import (
    get_log_path, get_log_dir, synthesize, configure_logging)
from babyai.utils.model import get_model_dir, load_model, save_model


def storage_dir():
    # defines the storage directory to be in the root (Same level as babyai folder)
    return os.environ.get("BABYAI_STORAGE", '.')


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



================================================
FILE: babyai/utils/agent.py
================================================
from abc import ABC, abstractmethod
import torch
from .. import utils
from babyai.bot import Bot
from babyai.model import ACModel
from random import Random


class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_or_name, obss_preprocessor, argmax):
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = utils.ObssPreprocessor(model_or_name)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            self.model = utils.load_model(model_or_name)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

    def act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        with torch.no_grad():
            model_results = self.model(preprocessed_obs, self.memory)
            dist = model_results['dist']
            value = model_results['value']
            self.memory = model_results['memory']

        if self.argmax:
            action = dist.probs.argmax(1)
        else:
            action = dist.sample()

        return {'action': action,
                'dist': dist,
                'value': value}

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0.
        else:
            self.memory *= (1 - done)


class RandomAgent:
    """A newly initialized model-based agent."""

    def __init__(self, seed=0, number_of_actions=7):
        self.rng = Random(seed)
        self.number_of_actions = number_of_actions

    def act(self, obs):
        action = self.rng.randint(0, self.number_of_actions - 1)
        # To be consistent with how a ModelAgent's output of `act`:
        return {'action': torch.tensor(action),
                'dist': None,
                'value': None}


class DemoAgent(Agent):
    """A demonstration-based agent. This agent behaves using demonstrations."""

    def __init__(self, demos_name, env_name, origin):
        self.demos_path = utils.get_demos_path(demos_name, env_name, origin, valid=False)
        self.demos = utils.load_demos(self.demos_path)
        self.demos = utils.demos.transform_demos(self.demos)
        self.demo_id = 0
        self.step_id = 0

    @staticmethod
    def check_obss_equality(obs1, obs2):
        if not(obs1.keys() == obs2.keys()):
            return False
        for key in obs1.keys():
            if type(obs1[key]) in (str, int):
                if not(obs1[key] == obs2[key]):
                    return False
            else:
                if not (obs1[key] == obs2[key]).all():
                    return False
        return True

    def act(self, obs):
        if self.demo_id >= len(self.demos):
            raise ValueError("No demonstration remaining")
        expected_obs = self.demos[self.demo_id][self.step_id][0]
        assert DemoAgent.check_obss_equality(obs, expected_obs), "The observations do not match"

        return {'action': self.demos[self.demo_id][self.step_id][1]}

    def analyze_feedback(self, reward, done):
        self.step_id += 1

        if done:
            self.demo_id += 1
            self.step_id = 0


class BotAgent:
    def __init__(self, env):
        """An agent based on a GOFAI bot."""
        self.env = env
        self.on_reset()

    def on_reset(self):
        self.bot = Bot(self.env)

    def act(self, obs=None, update_internal_state=True, *args, **kwargs):
        action = self.bot.replan()
        return {'action': action}

    def analyze_feedback(self, reward, done):
        pass


def load_agent(env, model_name, demos_name=None, demos_origin=None, argmax=True, env_name=None):
    # env_name needs to be specified for demo agents
    if model_name == 'BOT':
        return BotAgent(env)
    elif model_name is not None:
        obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)
        return ModelAgent(model_name, obss_preprocessor, argmax)
    elif demos_origin is not None or demos_name is not None:
        return DemoAgent(demos_name=demos_name, env_name=env_name, origin=demos_origin)



================================================
FILE: babyai/utils/demos.py
================================================
import os
import pickle

from .. import utils
import blosc


def get_demos_path(demos=None, env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)


def load_demos(path, raise_not_found=True):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            return []


def save_demos(demos, path):
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))


def transform_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission}
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos



================================================
FILE: babyai/utils/format.py
================================================
import os
import json
import numpy
import re
import torch
import babyai.rl

from .. import utils


def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_



================================================
FILE: babyai/utils/log.py
================================================
import os
import sys
import numpy
import logging

from .. import utils


def get_log_dir(log_name):
    return os.path.join(utils.storage_dir(), "logs", log_name)


def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.log")


def synthesize(array):
    import collections
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def configure_logging(log_name):
    path = get_log_path(log_name)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )



================================================
FILE: babyai/utils/model.py
================================================
import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")


def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    try:
        if torch.cuda.is_available():
            model = torch.load(path)
        else:
            model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def save_model(model, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)



================================================
FILE: docs/bonus_levels.md
================================================
# Bonus Levels

The levels described in this file were created prior to the ICLR19 publication.
We've chosen to keep these because they may be useful for curriculum learning
or for specific research projects.

Please note that these levels are not as widely tested as the ICLR19 levels.
If you run into problems, please open an issue on this repository.

In naming the levels we adhere to the following convention:
- `N2`, `N3`, `N4` refers to the number of objects in the room/environment
- `S2`, `S3`, `S4` refers to the size of the room/environment
- in `Debug` levels the episode is terminated once the agent does something unnecessary or fatally bad, for example
    - picks up an object which it is not supposed to pick up (unnecessary)
    - open the door that it is supposed to open _after_ another one (fatal)
- in `Carrying` levels the agent starts carrying the object of interest
- in `Dist` levels distractor objects are placed to confuse the agent

## OpenRedDoor

- Environment: The agent is placed in a room with a door.
- instruction: open the red door
- Evaluate: image understanding
- Level id: `BabyAI-OpenRedDoor-v0`

<p align="center"><img src="/media/OpenRedDoor.png" width="250"></p>

## OpenDoor

- Environment: The agent is placed in a room with 4 different doors. The environment is done when the instruction is executed in the regular mode or when a door is opened in the `debug` mode.
- instruction: open a door of:
    - a given color or location in `OpenDoor`
    - a given color in `OpenDoorColor`
    - a given location in `OpenDoorLoc`
- Evaluate: image & text understanding, memory in `OpenDoor` and `OpenDoorLoc`
- Level id:
    - `BabyAI-OpenDoor-v0`
    - `BabyAI-OpenDoorDebug-v0`
    - `BabyAI-OpenDoorColor-v0`
    - `BabyAI-OpenDoorColorDebug-v0`
    - `BabyAI-OpenDoorLoc-v0`
    - `BabyAI-OpenDoorLocDebug-v0`

<p align="center"><img src="/media/OpenDoor.png" width="250"></p>

## GoToDoor

- Environment: The agent is placed in a room with 4 different doors.
- Instruction: Go to a door of a given of a given color.
- Evaluate: image & text understanding
- Level id: `BabyAI-GoToDoor-v0`

## GoToObjDoor

- Environment: The agent is placed in a room with 4 different doors and 5 different objects.
- Instruction: Go to an object or a door of a given type and color
- Evaluate: image & text understanding
- Level id: `BabyAI-GoToObjDoor-v0`

<p align="center"><img src="/media/GoToObjDoor.png" width="250"></p>

## ActionObjDoor

- Environment: The agent is placed in a room with 4 different doors and 5 different objects.
- Instruction: [Pick up an object] or [go to an object or door] or [open a door]
- Evaluate: image & text understanding
- Level id: `BabyAI-ActionObjDoor-v0`

<p align="center"><img src="/media/ActionObjDoor.png" width="250"></p>

## UnlockPickup

- Environment: The agent is placed in a room with a key and a locked door. The door opens onto a room with a box. Rooms have either no distractors in `UnlockPickup` or 4 distractors in `UnlockPickupDist`.
- instruction: pick up an object of a given type and color
- Evaluate: image understanding, memory in `UnlockPickupDist`
- Level id: `BabyAI-UnlockPickup-v0`, `BabyAI-UnlockPickupDist-v0`

<p align="center">
    <img src="/media/UnlockPickup.png" width="250">
    <img src="/media/UnlockPickupDist.png" width="250">
</p>

## BlockedUnlockPickup

- Environment: The agent is placed in a room with a key and a locked door. The door is blocked by a ball. The door opens onto a room with a box.
- instruction: pick up the box
- Evaluate: image understanding
- Level id: `BabyAI-BlockedUnlockPickup-v0`

<p align="center"><img src="/media/BlockedUnlockPickup.png" width="250"></p>

## UnlockToUnlock

- Environment: The agent is placed in a room with a key of color A and two doors of color A and B. The door of color A opens onto a room with a key of color B. The door of color B opens onto a room with a ball.
- instruction: pick up the ball
- Evaluate: image understanding
- Level id: `BabyAI-UnlockToUnlock-v0`

<p align="center"><img src="/media/UnlockToUnlock.png" width="250"></p>

## KeyInBox

- Environment: The agent is placed in a room with a box containing a key and a locked door.
- instruction: open the door
- Evaluate: image understanding
- Level id: `BabyAI-KeyInBox-v0`

<p align="center"><img src="/media/KeyInBox.png" width="250"></p>

## PickupDist

- Environment: The agent is placed in a room with 5 objects. The environment is done when the instruction is executed in the regular mode or when any object is picked in the `debug` mode.
- instruction: pick up an object of a given type and color
- Evaluate: image & text understanding
- Level id:
    - `BabyAI-PickupDist-v0`
    - `BabyAI-PickupDistDebug-v0`

<p align="center"><img src="/media/PickupDist.png" width="250"></p>

## PickupAbove

- Environment: The agent is placed in the middle room. An object is placed in the top-middle room.
- instruction: pick up an object of a given type and color
- Evaluate: image & text understanding, memory
- Level id: `BabyAI-PickupAbove-v0`

<p align="center"><img src="/media/PickupAbove.png" width="250"></p>

## OpenRedBlueDoors

- Environment: The agent is placed in a room with a red door and a blue door facing each other. The environment is done when the instruction is executed in the regular mode or when the blue door is opened in the `debug` mode.
- instruction: open the red door then open the blue door
- Evaluate: image understanding, memory
- Level id:
    - `BabyAI-OpenRedBlueDoors-v0`
    - `BabyAI-OpenRedBlueDoorsDebug-v0`

<p align="center"><img src="/media/OpenRedBlueDoors.png" width="250"></p>

## OpenTwoDoors

- Environment: The agent is placed in a room with a red door and a blue door facing each other. The environment is done when the instruction is executed in the regular mode or when the second door is opened in the `debug` mode.
- instruction: open the door of color X then open the door of color Y
- Evaluate: image & text understanding, memory
- Level id:
    - `BabyAI-OpenTwoDoors-v0`
    - `BabyAI-OpenTwoDoorsDebug-v0`

<p align="center"><img src="/media/OpenTwoDoors.png" width="250"></p>

## FindObj

- Environment: The agent is placed in the middle room. An object is placed in one of the rooms. Rooms have a size of 5 in `FindObjS5`, 6 in `FindObjS6` or 7 in `FindObjS7`.
- instruction: pick up an object of a given type and color
- Evaluate: image understanding, memory
- Level id:
    - `BabyAI-FindObjS5-v0`
    - `BabyAI-FindObjS6-v0`
    - `BabyAI-FindObjS7-v0`

<p align="center">
    <img src="/media/FindObjS5.png" width="250">
    <img src="/media/FindObjS6.png" width="250">
    <img src="/media/FindObjS7.png" width="250">
</p>

## FourObjs

- Environment: The agent is placed in the middle room. 4 different objects are placed in the adjacent rooms. Rooms have a size of 5 in `FourObjsS5`, 6 in `FourObjsS6` or 7 in `FourObjsS7`.
- instruction: pick up an object of a given type and location
- Evaluate: image understanding, memory
- Level id:
    - `BabyAI-FourObjsS5-v0`
    - `BabyAI-FourObjsS6-v0`
    - `BabyAI-FourObjsS7-v0`

<p align="center">
    <img src="/media/FourObjsS5.png" width="250">
    <img src="/media/FourObjsS6.png" width="250">
    <img src="/media/FourObjsS7.png" width="250">
</p>

## KeyCorridor

- Environment: The agent is placed in the middle of the corridor. One of the rooms is locked and contains a ball. Another room contains a key for opening the previous one. The level is split into a curriculum starting with one row of 3x3 rooms, going up to 3 rows of 6x6 rooms.
- instruction: pick up an object of a given type
- Evaluate: image understanding, memory
- Level ids:
  - `BabyAI-KeyCorridorS3R1-v0`
  - `BabyAI-KeyCorridorS3R2-v0`
  - `BabyAI-KeyCorridorS3R3-v0`
  - `BabyAI-KeyCorridorS4R3-v0`
  - `BabyAI-KeyCorridorS5R3-v0`
  - `BabyAI-KeyCorridorS6R3-v0`

<p align="center">
    <img src="/media/KeyCorridorS3R1.png" width="250">
    <img src="/media/KeyCorridorS3R2.png" width="250">
    <img src="/media/KeyCorridorS3R3.png" width="250">
    <img src="/media/KeyCorridorS4R3.png" width="250">
    <img src="/media/KeyCorridorS5R3.png" width="250">
    <img src="/media/KeyCorridorS6R3.png" width="250">
</p>

## 1Room

- Environment: The agent is placed in a room with a ball. The level is split into a curriculum with rooms of size 8, 12, 16 or 20.
- instruction: pick up the ball
- Evaluate: image understanding, memory
- Level ids:
  - `BabyAI-1RoomS8-v0`
  - `BabyAI-1RoomS12-v0`
  - `BabyAI-1RoomS16-v0`
  - `BabyAI-1RoomS20-v0`

<p align="center">
    <img src="/media/1RoomS8.png" width="250">
    <img src="/media/1RoomS12.png" width="250">
    <img src="/media/1RoomS16.png" width="250">
    <img src="/media/1RoomS20.png" width="250">
</p>

## OpenDoorsOrder

- Environment: There are two or four doors in a room. The agent has to open
 one or two of the doors in a given order.
- Instruction:
  - open the X door
  - open the X door and then open the Y door
  - open the X door after you open the Y door
- Level ids:
  - `BabyAI-OpenDoorsOrderN2-v0`
  - `BabyAI-OpenDoorsOrderN4-v0`
  - `BabyAI-OpenDoorsOrderN2Debug-v0`
  - `BabyAI-OpenDoorsOrderN4Debug-v0`

## PutNext

- Environment: Single room with multiple objects. One of the objects must be moved next to another specific object.
- instruction: put the X next to the Y
- Level ids:
  - `BabyAI-PutNextS4N1-v0`
  - `BabyAI-PutNextS5N1-v0`
  - `BabyAI-PutNextS6N2-v0`
  - `BabyAI-PutNextS6N3-v0`
  - `BabyAI-PutNextS7N4-v0`
  - `BabyAI-PutNextS6N2Carrying-v0`
  - `BabyAI-PutNextS6N3Carrying-v0`
  - `BabyAI-PutNextS7N4Carrying-v0`

## MoveTwoAcross

- Environment: Two objects must be moved so that they are next to two other objects. This task is structured to have a very large number of possible instructions.
- instruction: put the A next to the B and the C next to the D
- Level ids:
  - `BabyAI-MoveTwoAcrossS5N2-v0`
  - `BabyAI-MoveTwoAcrossS8N9-v0`



================================================
FILE: docs/iclr19_levels.md
================================================
# ICLR19 Levels

The levels described in this file were created for the ICLR19 submission.
These form a curriculum that is subdivided according to specific competencies.

## GoToObj

Go to an object, inside a single room with no doors, no distractors.

<p align="center"><img src="/media/GoToObj.png" width="180"></p>

## GoToRedBall

Go to the red ball, single room, with obstacles.
The obstacles/distractors are all the same, to eliminate
perceptual complexity.

<p align="center"><img src="/media/GoToRedBall.png" width="180"></p>

## GoToRedBallGrey

Go to the red ball, single room, with obstacles.
The obstacles/distractors are all grey boxes, to eliminate
perceptual complexity. No unblocking required.

<p align="center"><img src="/media/GoToRedBallGrey.png" width="180"></p>

## GoToLocal

Go to an object, inside a single room with no doors, no distractors.

<p align="center"><img src="/media/GoToLocal.png" width="180"></p>

## PutNextLocal

Put an object next to another object, inside a single room
with no doors, no distractors.

<p align="center"><img src="/media/PutNextLocal.png" width="180"></p>

## PickUpLoc

Pick up an object which may be described using its location. This is a
single room environment.

Competencies: PickUp, Loc. No unblocking.

<p align="center"><img src="/media/PickupLoc.png" width="180"></p>

## GoToObjMaze

Go to an object, the object may be in another room. No distractors.

<p align="center"><img src="/media/GoToObjMaze.png" width="400"></p>

## GoTo

Go to an object, the object may be in another room. Many distractors.

<p align="center"><img src="/media/GoTo.png" width="400"></p>

## Pickup

Pick up an object, the object may be in another room.

<p align="center"><img src="/media/Pickup.png" width="400"></p>

## UnblockPickup

Pick up an object, the object may be in another room. The path may
be blocked by one or more obstructors.

<p align="center"><img src="/media/UnblockPickup.png" width="400"></p>

## Open

Open a door, which may be in another room.

<p align="center"><img src="/media/Open.png" width="400"></p>

## Unlock

Maze environment where the agent has to retrieve a key to open a locked door.

Competencies: Maze, Open, Unlock. No unblocking.

<p align="center"><img src="/media/Unlock.png" width="400"></p>

## PutNext

Put an object next to another object. Either of these may be in another room.

<p align="center"><img src="/media/PutNext.png" width="400"></p>

## Synth

Union of all instructions from PutNext, Open, Goto and PickUp. The agent
may need to move objects around. The agent may have to unlock the door,
but only if it is explicitly referred by the instruction.

Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open

<p align="center"><img src="/media/Synth.png" width="400"></p>

## SynthLoc

Like Synth, but a significant share of object descriptions involves
location language like in PickUpLoc. No implicit unlocking.
Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc

<p align="center"><img src="/media/SynthLoc.png" width="400"></p>

## GoToSeq

Sequencing of go-to-object commands.

Competencies: Maze, GoTo, Seq. No locked room. No locations. No unblocking.

<p align="center"><img src="/media/GoToSeq.png" width="400"></p>

## SynthSeq

Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.

Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq. No implicit unlocking.

<p align="center"><img src="/media/SynthSeq.png" width="400"></p>

## GoToImpUnlock

Go to an object, which may be in a locked room. No unblocking.

Competencies: Maze, GoTo, ImpUnlock

<p align="center"><img src="/media/GoToImpUnlock.png" width="400"></p>

## BossLevel

Command can be any sentence drawn from the Baby Language grammar. Union of
all competencies. This level is a superset of all other levels.

<p align="center"><img src="/media/BossLevel.png" width="400"></p>




================================================
FILE: scripts/README.md
================================================
# Scripts

There are sixteen scripts, split in five categories.

Reinforcement Learning:
- `train_rl.py`
- `rl_dataeff.py`

Imitation Learning:
- `make_agent_demos.py`
- `train_il.py`
- `il_dataeff.py`
- `il_perf`
- `compare_dataeff.py`

Visualisation:
- `manual_control.py`
- `compute_possible_instructions.py`
- `show_level_instructions.py`

Evaluating the Agent
- `enjoy.py`
- `evaluate.py`
- `evaluate_all_models.py`

Others:
- `train_intelligent_expert.py`
- `evaluate_all_demos.py`
- `eval_bot.py`

A common argument in this script is  `--env`.  Possible values are `BabyAI-<LEVEL_NAME>-v0`, where `LEVEL_NAME` is one of 19 levels presented in our paper.

## Reinforcement Learning

To train an RL agent run e.g.
```
scripts/train_rl.py --env BabyAI-GoToLocal-v0
```
Folders `logs/` and `models/` will be created in the current directory. The default name
for the model is chosen based on the level name, the current time and the other settings (e.g. `BabyAI-GoToLocal-v0_ppo_expert_filmcnn_gru_mem_seed1_18-10-12-12-45-02`). You can also choose the model name by setting `--model`. After 5 hours of training you should be getting a success rate of 97-99\%.

A machine readable log can be found in `logs/<MODEL>/log.csv`, a human readable in `logs/<MODEL>/log.log`.

To reproduce our results, use `scripts/train_rl.py` to run several jobs for each level (and don't forget to vary `--seed`).  The jobs don't stop by themselves, cancel them when you feel like.


### Reinforcement Learning Sample Efficiency

To measure how many episodes is required to get 100% performance, do:
```
scripts/rl_dataeff.py --path <PATH/TO/LOGS> --regex <REGEX>
```
If you want to perform a two-tailed T-test with unequal variance, add the `--ttest <PATH/TO/LOGS>` and `--ttest_regex`.

The regex arguments are optional.

For most levels the default value `--window=100` makes sense, but for `GoToRedBallGrey` we used `--window=10`.


## Imitation learning

To generate demos, run:
```
scripts/make_agent_demos.py --episodes <NUM_OF_EPISODES> --env <ENV_NAME> --demos <PATH/TO/FILENAME>
```
To train an agent with IL (imitation learning) first make sure that you have your demonstrations in `demos/<DEMOS>`. Then run e.g.
```
scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos <DEMOS>
```
For simple levels (`GoToRedBallGrey`, `GoToRedBall`, `GoToLocal`, `PickupLoc`, `PutNextLocal`), we used the **small** architectural configuration:
```
--batch-size=256 --val-episodes 512 --val-interval 1 --log-interval 1 --epoch-length 25600
```

For all other levels, we use the **big** architectural configuration:
```
--memory-dim=2048 --recurrence=80 --batch-size=128 --instr-arch=attgru --instr-dim=256 --val-interval 1 --log-interval 1  --epoch-length 51200 --lr 5e-5
```

Optional arguments for this script are
```
--episodes <NUMBER_OF_DEMOS> --arch <ARCH> --seed <SEED>
```
If `<SEED> = 0`, a random seed is automatically generated.  Otherwise, manually set a seed.

`<ARCH>` is one of `original`, `original_endpool_res`, `bow_endpool_res`.  **Using the `pixels` architecture does not work with imitation learning**, because the demonstrations were not generated to use pixels.


### Imitation Learning Performance

To measure the success rate of an agent trained by imitation learning, do
```
scripts/il_perf.py --path <PATH/TO/LOGS> --regex <REGEX>
```
If you want to perform a two-tailed T-test with unequal variance, add the `--ttest <PATH/TO/LOGS>` and `--ttest_regex`.

The regex arguments are optional.

For most levels the default value `--window=100` makes sense, but for `GoToRedBallGrey` we used `--window=10`.


### Sample efficiency

In [BabyAI 1.1](http://arxiv.org/abs/2007.12770), we do not evaluate using this process.  See [Imitation Learning Performance](###-imitation-learning-performance) instead.

To measure sample efficiency of imitation learning you have to train the model using different numbers of samples.  The `main` function from `babyai/efficiency.py` can help with you this. In order to use `main`, you have to create a file `babyai/cluster_specific.py` and implement a `launch_job` function in it that launches the job at the cluster that you have at your disposal.

Below is an example launch script for the `GoToRedBallGrey` level. Before running the script, make sure the [official demonstration files](https://drive.google.com/drive/folders/124DhBJ5BdiLyRowkYnVtfcYHKre9ouSp) are located in `./demos`.
``` python
from babyai.efficiency import main
total_time = int(1e6)
for i in [1, 2, 3]:
    # i is the random seed
    main('BabyAI-GoToRedBallGrey-v0', i, total_time, 1000000)
# 'main' will use a different seed for each of the runs in this series
main('BabyAI-GoToRedBallGrey-v0', 1, total_time, int(2 ** 12), int(2 ** 15), step_size=2 ** 0.2)
```
`total_time` is the total number of examples in all the batches that the model is trained on. This is not to be confused with the number of invidiual examples. The above code will run
-  3 jobs with 1M demonstrations (these are used to compute the ``normal'' time it takes to train the model on a given level, see the paper for more details)
- 16 jobs with the number of demonstrations varied from `2 ** 12` to `2 ** 15` using the log-scale step of ``2 ** 0.2``

When all the jobs finish, use `scripts/il_dataeff.py` to estimate the minimum number of demonstrations that are required to achieve the 99% success rate:
```
scripts/il_dataeff.py --regex '.*-GoToRedBallGrey-.*' --window 10 gotoredballgrey
```
`--window 10` means that results of 10 subsequent validations are averaged to make sure that the 99% threshold is crossed robustly. When you have many models in one directory, use `--regex` to select the models that were trained on a specific level, in this case GoToRedBallGrey. `gotoredballgrey` directory will contain 3 files:
- `summary.csv` summarizes the results of all runs that were taken into consideration
- `visualization.png` illustrates the GP-based interpolation and the estimated credible interval
- `result.json` describes the resulting sample efficiency estimate. `min` and `max` are the boundaries of the 99% credible interval. The estimatation is done by using Gaussian Process interpolation, see the paper for more details.

If you wish to compare sample efficiencies of two models `M1` and `M2`, use `scripts/compare_dataeff.py`:
```
scripts/compare_dataeff.py M1 M2
```
Here, `M1` and `M2` are report directories created by `scripts/il_dataeff.py`.

Note: use `level_type='big'` in your `main` call to train big models of the kind that we use for big 3x3 maze levels.

### Curriculum learning sample efficiency.
Use the `pretrained_model` argument of `main` from `babyai/efficiency.py`.

### Big baselines for all Levels
Just like above, but always use a big model.

To reproduce results in the paper, trained for 20 passes over 1M examples for big levels and 40 passes for small levels.

### Imitation learning from an RL expert
Generate 1M demos from the agents that were trained for ~24 hours. Do same as above.


## Evaluating the Agent

In the same directory where you trained your model run e.g.
```
scripts/evaluate.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```
to evaluate the performance of your model named `<MODEL>` on 1000 episodes. If you want to see your agent performing, run
```
scripts/enjoy.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```
 `evaluate_all_models.py` evaluates the performance of all models in a storage directory.

## Visualisation

To run the interactive GUI platform:
```
scripts/manual_control.py -- env <LEVEL>
```
To see what instructions a `LEVEL` generates, run:
```
scripts/show_level_instructions.py -- env <LEVEL>
```
`compute_possible_instructions.py` returns the number of different possible instructions in BossLevel.  It accepts no arguments.

## Others

- `train_intelligent_expert.py` trains an agent with an interactive imitation learning algorithm that incrementally grows the training set by adding demonstrations for the missions that the agent currently fails.
- `eval_bot.py` is used to debug the bot and ensure that it works on all levels.
- `evaluate_all_demos.py` ensures that all demos complete the instruction.



================================================
FILE: scripts/compare_dataeff.py
================================================
#!/usr/bin/env python3

import argparse
import json
import os
from scipy import stats

from babyai import plotting


parser = argparse.ArgumentParser("Compare data efficiency of two approaches")
parser.add_argument("report1", default=None)
parser.add_argument("report2", default=None)
args = parser.parse_args()

r1 = json.load(open(os.path.join(args.report1, 'result.json')))
r2 = json.load(open(os.path.join(args.report2, 'result.json')))
diff_std = (r1['std_log2'] ** 2 + r2['std_log2'] ** 2) ** 0.5
p_less = stats.norm.cdf(0, r2['mean_log2'] - r1['mean_log2'], diff_std)
print('less samples required with {} probability'.format(p_less))
print('more samples required with {} probability'.format(1 - p_less))



================================================
FILE: scripts/compute_possible_instructions.py
================================================
#!/usr/bin/env python3

"""
Compute the number of possible instructions in the BabyAI grammar.
"""

from gym_minigrid.minigrid import COLOR_NAMES

def count_Sent():
    return (
        count_Sent1() +
        # Sent1, then Sent1
        count_Sent1() * count_Sent1() +
        # Sent1 after you Sent1
        count_Sent1() * count_Sent1()
    )

def count_Sent1():
    return (
        count_Clause() +
        # Clause and Clause
        count_Clause() * count_Clause()
    )

def count_Clause():
    return (
        # go to
        count_Descr() +
        # pick up
        count_DescrNotDoor() +
        # open
        count_DescrDoor() +
        # put next
        count_DescrNotDoor() * count_Descr()
    )

def count_DescrDoor():
    # (the|a) Color door Location
    return 2 * count_Color() * count_LocSpec()
def count_DescrBall():
    return count_DescrDoor()
def count_DescrBox():
    return count_DescrDoor()
def count_DescrKey():
    return count_DescrDoor()
def count_Descr():
    return count_DescrDoor() + count_DescrBall() + count_DescrBox() + count_DescrKey()
def count_DescrNotDoor():
    return count_DescrBall() + count_DescrBox() + count_DescrKey()

def count_Color():
    # Empty string or color
    return len([None] + COLOR_NAMES)

def count_LocSpec():
    # Empty string or location
    return len([None, 'left', 'right', 'front', 'behind'])

print('DescrKey: ', count_DescrKey())
print('Descr: ', count_Descr())
print('DescrNotDoor: ', count_DescrNotDoor())
print('Clause: ', count_Clause())
print('Sent1: ', count_Sent1())
print('Sent: ', count_Sent())
print('Sent: {:.3g}'.format(count_Sent()))



================================================
FILE: scripts/enjoy.py
================================================
#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""

import argparse
import gym
import time

import babyai.utils as utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")

args = parser.parse_args()

action_map = {
    "left"      : "left",
    "right"     : "right",
    "up"        : "forward",
    "p"         : "pickup",
    "pageup"    : "pickup",
    "d"         : "drop",
    "pagedown"  : "drop",
    " "         : "toggle"
}

assert args.model is not None or args.demos is not None, "--model or --demos must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))

# Define agent
agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)

# Run the agent

done = True

action = None

def keyDownCb(event):
    global obs

    keyName = event.key
    print(keyName)

    # Avoiding processing of observation by agent for wrong key clicks
    if keyName not in action_map and keyName != "enter":
        return

    agent_action = agent.act(obs)['action']

    # Map the key to an action
    if keyName in action_map:
        action = env.actions[action_map[keyName]]

    # Enter executes the agent's action
    elif keyName == "enter":
        action = agent_action

    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)
    if done:
        print("Reward:", reward)
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))


if args.manual_mode:
    env.render('human')
    env.window.reg_key_handler(keyDownCb)

step = 0
episode_num = 0
while True:
    time.sleep(args.pause)
    env.render("human")
    if not args.manual_mode:
        result = agent.act(obs)
        obs, reward, done, _ = env.step(result['action'])
        agent.analyze_feedback(reward, done)
        if 'dist' in result and 'value' in result:
            dist, value = result['dist'], result['value']
            dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
            print("step: {}, mission: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
                step, obs["mission"], dist_str, float(dist.entropy()), float(value)))
        else:
            print("step: {}, mission: {}".format(step, obs['mission']))
        if done:
            print("Reward:", reward)
            episode_num += 1
            env.seed(args.seed + episode_num)
            obs = env.reset()
            agent.on_reset()
            step = 0
        else:
            step += 1

    if env.window.closed:
        break



================================================
FILE: scripts/eval_bot.py
================================================
#!/usr/bin/env python3

"""
Evaluate the success rate of the bot
This script is used for testing/debugging purposes

Examples of usage:
- Run the bot on the GoTo level 10 times (seeds 9 to 18)
eval_bot.py --level GoTo --num_runs 10 --seed 9
- for all levels, 100 times, run a Random(seed 0) agent for len(episode)/3 steps before running the bot:
eval_bot.py --advise_mode --num_runs 100
- for all levels, 500 times, during the first 10 steps, choose action form a Random(seed 9) agent with proba .9 or
 optimal (from bot) with proba .1, then continue with optimal bot actions:
eval_boy.py --advise_mode --bad_action_proba .8 --non_optimal_steps 10 --random_agent_seed 9

"""

import random
import time
import traceback
from optparse import OptionParser
from babyai.levels import level_dict
from babyai.bot import Bot
from babyai.utils.agent import ModelAgent, RandomAgent
from random import Random


# MissBossLevel is the only level the bot currently can't always handle
level_list = [name for name, level in level_dict.items()
              if (not getattr(level, 'is_bonus', False) and not name == 'MiniBossLevel')]


parser = OptionParser()
parser.add_option(
    "--level",
    default=None
)
parser.add_option(
    "--advise_mode",
    action='store_true',
    default=False,
    help='If specified, a RandomAgent or ModelAgent will act first, then the bot will take over')
parser.add_option(
    "--non_optimal_steps",
    type=int,
    default=None,
    help='Number of non bot steps ModelAgent or RandomAgent takes before letting the bot take over'
)
parser.add_option(
    "--model",
    default=None,
    help='Model to use to act for a few steps before letting the bot take over'
)
parser.add_option(
    "--random_agent_seed",
    type="int",
    default=1,
    help='Seed of the random agent that acts a few steps before letting the bot take over'
)
parser.add_option(
    "--bad_action_proba",
    type="float",
    default=1.,
    help='Probability of performing the non-optimal action when the random/model agent is performing'
)
parser.add_option(
    "--seed",
    type="int",
    default=1
)
parser.add_option(
    "--num_runs",
    type="int",
    default=500
)
parser.add_option(
    "--verbose",
    action='store_true'
)
(options, args) = parser.parse_args()

if options.level:
    level_list = [options.level]

bad_agent = None
if options.advise_mode:
    if options.model:
        bad_agent = ModelAgent(options.model, obss_preprocessor=None,
                               argmax=True)
    else:
        bad_agent = RandomAgent(seed=options.random_agent_seed)

start_time = time.time()

all_good = True

for level_name in level_list:

    num_success = 0
    total_reward = 0
    total_steps = []
    total_bfs = 0
    total_episode_steps = 0
    total_bfs_steps = 0

    for run_no in range(options.num_runs):
        level = level_dict[level_name]

        mission_seed = options.seed + run_no
        mission = level(seed=mission_seed)
        expert = Bot(mission)

        if options.verbose:
            print('%s/%s: %s, seed=%d' % (run_no+1, options.num_runs, mission.surface, mission_seed))

        optimal_actions = []
        before_optimal_actions = []
        non_optimal_steps = options.non_optimal_steps or int(mission.max_steps // 3)
        rng = Random(mission_seed)

        try:
            episode_steps = 0
            last_action = None
            while True:
                action = expert.replan(last_action)
                if options.advise_mode and episode_steps < non_optimal_steps:
                    if rng.random() < options.bad_action_proba:
                        while True:
                            action = bad_agent.act(mission.gen_obs())['action'].item()
                            fwd_pos = mission.agent_pos + mission.dir_vec
                            fwd_cell = mission.grid.get(*fwd_pos)
                            # The current bot can't recover from two kinds of behaviour:
                            # - opening a box (cause it just disappears)
                            # - closing a door (cause its path finding mechanism get confused)
                            opening_box = (action == mission.actions.toggle
                                and fwd_cell and fwd_cell.type == 'box')
                            closing_door = (action == mission.actions.toggle
                                and fwd_cell and fwd_cell.type == 'door' and fwd_cell.is_open)
                            if not opening_box and not closing_door:
                                break
                    before_optimal_actions.append(action)
                else:
                    optimal_actions.append(action)

                obs, reward, done, info = mission.step(action)
                last_action = action

                total_reward += reward
                episode_steps += 1

                if done:
                    total_episode_steps += episode_steps
                    total_bfs_steps += expert.bfs_step_counter
                    total_bfs += expert.bfs_counter
                    if reward > 0:
                        num_success += 1
                        total_steps.append(episode_steps)
                        if options.verbose:
                            print('SUCCESS on seed {}, reward {:.2f}'.format(mission_seed, reward))
                    if reward <= 0:
                        assert episode_steps == mission.max_steps  # Is there another reason for this to happen ?
                        if options.verbose:
                            print('FAILURE on %s, seed %d, reward %.2f' % (level_name, mission_seed, reward))
                    break
        except Exception as e:
            print('FAILURE on %s, seed %d' % (level_name, mission_seed))
            traceback.print_exc()
            # Playing these 2 sets of actions should get you to the mission snapshot above
            print(before_optimal_actions)
            print(optimal_actions)
            print(expert.stack)
            break

    all_good = all_good and (num_success == options.num_runs)

    success_rate = 100 * num_success / options.num_runs
    mean_reward = total_reward / options.num_runs
    mean_steps = sum(total_steps) / options.num_runs

    print('%16s: %.1f%%, r=%.3f, s=%.2f' % (level_name, success_rate, mean_reward, mean_steps))
    # Uncomment the following line to print the number of steps per episode (useful to look for episodes to debug)
    # print({options.seed + num_run: total_steps[num_run] for num_run in range(options.num_runs)})
end_time = time.time()
total_time = end_time - start_time
print('total time: %.1fs' % total_time)
if not all_good:
    raise Exception("some tests failed")
print('total episode_steps:', total_episode_steps)
print('total bfs:', total_bfs)
print('total bfs steps:', total_bfs_steps)



================================================
FILE: scripts/evaluate.py
================================================
#!/usr/bin/env python3

"""
Evaluate a trained model or bot
"""

import argparse
import gym
import time
import datetime

import babyai.utils as utils
from babyai.evaluate import evaluate_demo_agent, batch_evaluate, evaluate
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="name of the demos file (REQUIRED or --demos-origin or --model REQUIRED)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes of evaluation (default: 1000)")
parser.add_argument("--seed", type=int, default=int(1e9),
                    help="random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--contiguous-episodes", action="store_true", default=False,
                    help="Make sure episodes on which evaluation is done are contiguous")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")


def main(args, seed, episodes):
    # Set seed for all randomness sources
    utils.seed(seed)

    # Define agent

    env = gym.make(args.env)
    env.seed(seed)
    agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)
    if args.model is None and args.episodes > len(agent.demos):
        # Set the number of episodes to be the number of demos
        episodes = len(agent.demos)

    # Evaluate
    if isinstance(agent, utils.DemoAgent):
        logs = evaluate_demo_agent(agent, episodes)
    elif isinstance(agent, utils.BotAgent) or args.contiguous_episodes:
        logs = evaluate(agent, env, episodes, False)
    else:
        logs = batch_evaluate(agent, args.env, seed, episodes)


    return logs


if __name__ == "__main__":
    args = parser.parse_args()
    assert_text = "ONE of --model or --demos-origin or --demos must be specified."
    assert int(args.model is None) + int(args.demos_origin is None) + int(args.demos is None) == 2, assert_text

    start_time = time.time()
    logs = main(args, args.seed, args.episodes)
    end_time = time.time()

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    ellapsed_time = int(end_time - start_time)
    duration = datetime.timedelta(seconds=ellapsed_time)

    if args.model is not None:
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])

    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    if args.model is not None:
        print("F {} | FPS {:.0f} | D {} | R:xsmM {:.3f} {:.3f} {:.3f} {:.3f} | S {:.3f} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration,
                      *return_per_episode.values(),
                      success_per_episode['mean'],
                      *num_frames_per_episode.values()))
    else:
        print("F {} | FPS {:.0f} | D {} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration, *num_frames_per_episode.values()))

    indexes = sorted(range(len(logs["num_frames_per_episode"])), key=lambda k: - logs["num_frames_per_episode"][k])

    n = args.worst_episodes_to_show
    if n > 0:
        print("{} worst episodes:".format(n))
        for i in indexes[:n]:
            if 'seed_per_episode' in logs:
                print(logs['seed_per_episode'][i])
            if args.model is not None:
                print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
            else:
                print("- episode {}: F={}".format(i, logs["num_frames_per_episode"][i]))



================================================
FILE: scripts/evaluate_all_demos.py
================================================
"""
Script to evaluate all available demos.

Assumes all demos (human and agent, except the "valid" ones)
are generated with seed 1
"""

import os
from subprocess import call
import sys

import babyai.utils as utils

folder = os.path.join(utils.storage_dir(), "demos")
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".pkl") and 'valid' in filename:
        env = 'BabyAI-BossLevel-v0'  # It doesn't really matter. The evaluation only considers the lengths of demos.
        demo = filename[:-4]  # Remove the .pkl part of the name

        print("> Demos: {}".format(demo))

        command = ["python evaluate.py --env {} --demos {} --worst-episodes-to-show 0".format(env, demo)] + sys.argv[1:]
        call(" ".join(command), shell=True)



================================================
FILE: scripts/evaluate_all_models.py
================================================
"""
Evaluate all models in a storage directory.

In order to use this script make sure to add baby-ai-game/scripts to the $PATH
environment variable.

Sample usage:
evaluate_all_models.py --episodes 200 --argmax
"""

import os
from subprocess import call
import sys

import babyai.utils as utils
from babyai.levels import level_dict
import re

# List of all levels ordered by length of the level name from longest to shortest
LEVELS = sorted(list(level_dict.keys()), key=len)[::-1]


def get_levels_from_model_name(model):
    levels = []
    # Assume that our model names are separated with _ or -
    model_name_parts = re.split('_|-', model)
    for part in model_name_parts:
        # Assume that each part contains at most one level name.
        # Sorting LEVELS using length of level name is to avoid scenarios like
        # extracting 'GoTo' from the model name 'GoToLocal-model'
        for level in LEVELS:
            if level in part:
                levels.append('BabyAI-{}-v0'.format(level))
                break
    return list(set(levels))


folder = os.path.join(utils.storage_dir(), "models")

for model in sorted(os.listdir(folder)):
    if model.startswith('.'):
        continue
    envs = get_levels_from_model_name(model)
    print("> Envs: {} > Model: {}".format(envs, model))
    for env in envs:
        command = ["evaluate.py --env {} --model {}".format(env, model)] + sys.argv[1:]
        print("Command: {}".format(" ".join(command)))
        call(" ".join(command), shell=True)



================================================
FILE: scripts/il_dataeff.py
================================================
#!/usr/bin/env python3

import argparse
import pandas
import os
import json

from babyai import plotting


parser = argparse.ArgumentParser("Analyze data efficiency of imitation learning")
parser.add_argument('--path', default='.')
parser.add_argument("--regex", default='.*')
parser.add_argument("--patience", default=2, type=int)
parser.add_argument("--window", default=1, type=int)
parser.add_argument("--limit", default="frames")
parser.add_argument("report")
args = parser.parse_args()

if os.path.exists(args.report):
    raise ValueError("report directory already exists")
os.mkdir(args.report)

summary_path = os.path.join(args.report, 'summary.csv')
figure_path = os.path.join(args.report, 'visualization.png')
result_path = os.path.join(args.report, 'result.json')

df_logs = pandas.concat(plotting.load_logs(args.path), sort=True)
df_success_rate, normal_time = plotting.best_within_normal_time(
    df_logs, args.regex,
    patience=args.patience, window=args.window, limit=args.limit,
    summary_path=summary_path)
result = plotting.estimate_sample_efficiency(
    df_success_rate, visualize=True, figure_path=figure_path)
result['normal_time'] = normal_time

with open(result_path, 'w') as dst:
    json.dump(result, dst)



================================================
FILE: scripts/il_perf.py
================================================
#!/usr/bin/env python3
import argparse
import pandas
import os
import json
import re
import numpy as np
from scipy import stats

from babyai import plotting as bp


parser = argparse.ArgumentParser("Analyze performance of imitation learning")
parser.add_argument("--path", default='.',
    help="path to model logs")
parser.add_argument("--regex", default='.*',
    help="filter out some logs")
parser.add_argument("--other", default=None,
    help="path to model logs for ttest comparison")
parser.add_argument("--other_regex", default='.*',
    help="filter out some logs from comparison")
parser.add_argument("--window", type=int, default=100,
    help="size of sliding window average, 10 for GoToRedBallGrey, 100 otherwise")
args = parser.parse_args()


def get_data(path, regex):
    df = pandas.concat(bp.load_logs(path), sort=True)
    fps = bp.get_fps(df)
    models = df['model'].unique()
    models = [model for model in df['model'].unique() if re.match(regex, model)]

    maxes = []
    for model in models:
        df_model = df[df['model'] == model]
        success_rate = df_model['validation_success_rate']
        success_rate = success_rate.rolling(args.window, center=True).mean()
        success_rate = max(success_rate[np.logical_not(np.isnan(success_rate))])
        print(model, success_rate)
        maxes.append(success_rate)
    return np.array(maxes), fps



if args.other is not None:
    print("is this architecture better")
print(args.regex)
maxes, fps = get_data(args.path, args.regex)
result = {'samples': len(maxes), 'mean': maxes.mean(), 'std': maxes.std(),
          'fps_mean': fps.mean(), 'fps_std': fps.std()}
print(result)

if args.other is not None:
    print("\nthan this one")
    maxes_ttest, fps = get_data(args.other, args.other_regex)
    result = {'samples': len(maxes_ttest),
        'mean': maxes_ttest.mean(), 'std': maxes_ttest.std(),
        'fps_mean': fps.mean(), 'fps_std': fps.std()}
    print(result)
    ttest = stats.ttest_ind(maxes, maxes_ttest, equal_var=False)
    print(f"\n{ttest}")



================================================
FILE: scripts/make_agent_demos.py
================================================
#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import logging
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch

import babyai.utils as utils

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='BOT',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to generate demonstrations for")
parser.add_argument("--valid-episodes", type=int, default=512,
                    help="number of validation episodes to generate demonstrations for")
parser.add_argument("--seed", type=int, default=0,
                    help="start random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--log-interval", type=int, default=100,
                    help="interval between progress reports")
parser.add_argument("--save-interval", type=int, default=10000,
                    help="interval between demonstrations saving")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps")
parser.add_argument("--on-exception", type=str, default='warn', choices=('warn', 'crash'),
                    help="How to handle exceptions during demo generation")

parser.add_argument("--job-script", type=str, default=None,
                    help="The script that launches make_agent_demos.py at a cluster.")
parser.add_argument("--jobs", type=int, default=0,
                    help="Split generation in that many jobs")

args = parser.parse_args()
logger = logging.getLogger(__name__)

# Set seed for all randomness sources


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info('Demo length: {:.3f}+-{:.3f}'.format(
        np.mean(num_frames_per_episode), np.std(num_frames_per_episode)))


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)

    # Generate environment
    env = gym.make(args.env)

    agent = utils.load_agent(env, args.model, args.demos, 'agent', args.argmax, args.env)
    demos_path = utils.get_demos_path(args.demos, args.env, 'agent', valid)
    demos = []

    checkpoint_time = time.time()

    just_crashed = False
    while True:
        if len(demos) == n_episodes:
            break

        done = False
        if just_crashed:
            logger.info("reset the environment to find a mission that the bot can solve")
            env.reset()
        else:
            env.seed(seed + len(demos))
        obs = env.reset()
        agent.on_reset()

        actions = []
        mission = obs["mission"]
        images = []
        directions = []

        try:
            while not done:
                action = agent.act(obs)['action']
                if isinstance(action, torch.Tensor):
                    action = action.item()
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                actions.append(action)
                images.append(obs['image'])
                directions.append(obs['direction'])

                obs = new_obs
            if reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
                demos.app