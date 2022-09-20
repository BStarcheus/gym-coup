# Coup RL Gym Environment

gym-coup contains the Coup game logic and the [OpenAI Gym](https://gym.openai.com/) environment to simulate a complete 2-player game.

## Usage
In a python file:
```python
import gym
import gym_coup
env = gym.make('coup-v0')
env.reset()
```
Then you can take `.step()`'s with each player's actions in the game.
Actions are [defined here](https://github.com/BStarcheus/coup-rl/blob/main/gym-coup/gym_coup/envs/coup_env.py#L18).
Make sure that on any turn, you are only taking valid actions. Check with `.get_valid_actions()`.

To see how the game progresses set the log level and call `.render()`:
```python
import logging
logging.basicConfig()
logger = logging.getLogger('gym_coup')
logger.setLevel(logging.INFO)
```

For example:
```python
>>> env.render()
INFO:gym_coup:Turn 0
INFO:gym_coup:Player: Cards | IsCardFaceUp | Coins | LastAction
INFO:gym_coup:P1: Captain Contessa | False False | 1 | _
INFO:gym_coup:P2: Assassin Ambassador | False False | 2 | _
>>> env.get_valid_actions(text=True)
['income', 'foreign_aid', 'tax', 'exchange', 'steal']
>>> env.step('income')  # P1 takes 1 coin of income
>>> env.render()
INFO:gym_coup:Turn 1
INFO:gym_coup:Player: Cards | IsCardFaceUp | Coins | LastAction
INFO:gym_coup:P1: Captain Contessa | False False | 2 | income
INFO:gym_coup:P2: Assassin Ambassador | False False | 2 | _
```