import os, redis, json, ray
import gymnasium as gym
from ray import tune
from redis import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = Redis.from_url(REDIS_URL)
ray.init(address="auto")

def train(config, reporter):
    import gymnasium as gym
    env = gym.make(config["env"])
    from ray.rllib.algorithms.dqn import DQN
    algo = DQN(config={"env": config["env"], "framework": "torch",
                      "num_workers": 1, "train_batch_size": 32})
    total = 0
    for i in range(config["max_iter"]):
        res = algo.train()
        reward = res["episode_reward_mean"]
        total = reward
        reporter(timesteps_total=i+1, episode_reward_mean=reward)
        if i % 10 == 0:
            r.publish(config["run_id"], json.dumps({"type": "log", "reward": reward}))
    r.publish(config["run_id"], json.dumps({"type": "done", "reward": total}))

@ray.remote
class JobRunner:
    def run(self, payload):
        tune.run(train, config=payload, verbose=0)

runner = JobRunner.remote()

while True:
    _, payload = r.brpop("jobs")
    pl = json.loads(payload)
    runner.run.remote(pl)