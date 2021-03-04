from typing import List, Tuple

import numpy as np

from extbrst.model import Action, Agent, Prediction, Probability, Reward

TrialResult = Tuple[Action, Reward, Prediction, Probability]
Result = List[TrialResult]


def reward_function(prob: Probability) -> Reward:
    return int(np.random.uniform() < prob)


def trial_process(agent: Agent, reward_prob: Probability) -> TrialResult:
    pred = agent.predict()
    prob = agent.calculate_response_prob(-pred)
    action = agent.emit_action(prob)
    reward = reward_function(reward_prob)
    agent.update(reward, action)
    return action, reward, pred, prob


def run(agent: Agent, trial: int, reward_prob: Probability) -> Result:
    result: Result = []
    for _ in range(trial):
        ret = trial_process(agent, reward_prob)
        result.append(ret)
    return result


if __name__ == '__main__':
    from pandas import DataFrame

    from extbrst.model import GAIAgent

    agent = GAIAgent(1.)
    crf = 1.
    extinction = 0.

    crf_result = run(agent, trial=200, reward_prob=crf)
    ext_result = run(agent, trial=200, reward_prob=extinction)

    merged_result = crf_result + ext_result
    df = DataFrame(merged_result, columns=["action", "reward", "G", "p"])
    df.to_csv("result.csv")
