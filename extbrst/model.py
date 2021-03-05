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
        self.__alpha_t = 1.
        self.__beta_t = 1.
        self.__bias = bias

    def update(self, reward: Reward, action: Action):
        self.__alpha_t += reward * action
        self.__beta_t += (1 - reward) * action

    def predict(self) -> Prediction:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t
        alpha = np.exp(2 * self.__lambda)
        kl_div_a = -betaln(self.__alpha_t, self.__beta_t) \
            + (self.__alpha_t - alpha) * digamma(self.__alpha_t) \
            + (self.__beta_t - 1) * digamma(self.__beta_t) \
            + (alpha + 1 - nu_t) * digamma(nu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        return kl_div_a + h_a

    def calculate_response_prob(self, pred: Prediction) -> Probability:
        return 1 / (1 + np.exp(-pred - self.__bias))

    def emit_action(self, prob: Probability) -> Action:
        return int(np.random.uniform() < prob)


class SAIAgent(Agent):
    def __init__(self, lamb: float, bias: float):
        self.__lambda = lamb
        self.__alpha_t = 1.
        self.__beta_t = 1.
        self.__bias = bias

    def update(self, reward: Reward, action: Action):
        self.__alpha_t += reward * action
        self.__beta_t += (1 - reward) * action

    def predict(self) -> Prediction:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t

        kl_div_a = - self.__lambda * (2 * mu_t - 1) \
            + mu_t * np.log(mu_t) \
            + (1 - mu_t) * np.log(1 - mu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        return kl_div_a + h_a

    def calculate_response_prob(self, pred: Prediction) -> Probability:
        return 1 / (1 + np.exp(-pred - self.__bias))

    def emit_action(self, prob: Probability) -> Action:
        return int(np.random.uniform() < prob)
