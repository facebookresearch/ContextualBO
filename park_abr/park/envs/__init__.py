# Format follows OpenAI gym https://gym.openai.com
# Folk of the adaptive video streaming environment in https://github.com/park-project/park

from park.envs.registration import make, register


register(env_id="abr_sim_fb", entry_point="park.envs.abr_sim:ABRSimFBEnv")
