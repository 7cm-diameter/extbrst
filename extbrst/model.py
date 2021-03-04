from abc import ABC

import numpy as np
from scipy.special import betaln, digamma

Prediction = float
Probability = float
Action = int
Reward = int


class Agent(ABC):
    def update(self, reward: Reward, action: Action):
        pass

    def predict(self) -> Prediction:
        return 0.

    def calculate_response_prob(self, pred: Prediction) -> Probability:
        return 0.

    def emit_action(self, prob: Probability) -> Action:
        return 0


class GAIAgent(Agent):
    def __init__(self, lamb: float, bias: float = 5.):
        self.__lambda = lamb
        self.__alpha = 1.
        self.__beta = 1.
        self.__bias = bias

    def update(self, reward: Reward, action: Action):
        self.__alpha += reward * action
        self.__beta += (1 - reward) * action

    def predict(self) -> Prediction:
        nu = self.__alpha + self.__beta
        mu = self.__alpha / nu
        alpha = np.exp(2 * self.__lambda)
        kl_div_a = -betaln(self.__alpha, self.__beta) + \
            (self.__alpha - alpha) * digamma(self.__alpha) + \
            (self.__beta - 1) * digamma(self.__beta) + \
            (alpha + 1 - nu) * digamma(nu)
        ent_a = - mu * digamma(self.__alpha + 1) - \
            (1 - mu) * digamma(self.__beta + 1) + digamma(nu + 1)
        return kl_div_a + ent_a

    def calculate_response_prob(self, pred: Prediction) -> Probability:
        return 1 / (1 + np.exp(-pred - self.__bias))

    def emit_action(self, prob: Probability) -> Action:
        return int(np.random.uniform() < prob)
