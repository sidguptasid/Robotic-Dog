import time
from collections import deque
import copy
import os

from numpy import dtype

import torch
from ml_logger import logger
from params_proto import PrefixProto

#from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 200  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 10  # check for potential saves every this many iterations
    save_video_interval = 2
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True


class Runner:

    def __init__(self, env, device='cpu', isTorque=True):

        self.device = device
        self.env = env
        self.isTorque = isTorque

        if self.isTorque:
            from .ppo import PPO
            from .actor_critic import ActorCritic

            actor_critic = ActorCritic(self.env.num_obs,
                                        self.env.num_privileged_obs,
                                        self.env.num_obs_history,
                                        self.env.num_actions,
                                        ).to(self.device)
            self.alg = PPO(actor_critic, device=self.device)
        else:
            from .ppo_navigate import PPO
            from .actor_critic_navigate import ActorCritic

            actor_critic = ActorCritic(self.env.num_obs_vel,
                                        self.env.num_obs_history_vel,
                                        self.env.num_actions_vel,
                                        ).to(self.device)
            self.alg = PPO(actor_critic, device=self.device)

        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            from ml_logger import ML_Logger
            loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
                               prefix=RunnerArgs.resume_path)
            weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                # load curriculum state
                distributions = loader.load_pkl("curriculum/distribution.pkl")
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        if self.isTorque:
            self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])
        else:
            self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs_vel],
                              [self.env.num_obs_history_vel], [self.env.num_actions_vel])
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = None

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, obs_history = obs_dict["obs"], obs_dict["obs_history"]
        obs, obs_history = obs.to(self.device), obs_history.to(self.device)

        if "privileged_obs" in obs_dict:
            privileged_obs = obs_dict["privileged_obs"]
            privileged_obs = privileged_obs.to(self.device)

        if "obs_vel" in obs_dict:
            obs_vel = obs_dict["obs_vel"]
            obs_vel = obs_vel.to(self.device)

        if 'obs_history_vel' in obs_dict:
            obs_history_vel = obs_dict["obs_history_vel"]
            obs_history_vel = obs_history_vel.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        total_dones = []
        total_rews_buf = []
        y_dist_rews_buf = []
        wall_dist_rews_buf = []
        obstacle_rews_buf = []
        ep_goal_rews_buf = []
        num_traj_buf = []
        max_success_rate = 0
        last_max_success_rate_it = 0

        model_root_path = os.path.join(logger.root, logger.prefix)
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            
            num_success = 0
            tot_rew = 0
            tot_y_dist_rew = 0
            tot_wall_dist_rew = 0
            tot_obstacle_rew = 0
            tot_goal_rew = 0
            num_traj = self.env.num_envs
            num_new_traj = 0
            frames = None
            with torch.inference_mode():
                dones = None
                for i in range(self.num_steps_per_env):
                    num_traj += num_new_traj
                    if self.isTorque:
                        actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                    obs_history[:num_train_envs])
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])
                    else:
                        vel_actions_train = self.alg.act(obs_vel[:num_train_envs], obs_history_vel[:num_train_envs])
                        actions_train = self.env.convert_vel_to_action(vel_actions_train, obs_history[:num_train_envs], env_ids=list(range(num_train_envs)))

                        vel_actions_eval = self.alg.actor_critic.act_student(obs_history_vel[num_train_envs:])
                        actions_eval = self.env.convert_vel_to_action(vel_actions_eval, obs_history[num_train_envs:], env_ids = list(range(num_train_envs, self.env.num_envs)))

                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret

                    failures = failures | dones
                    if it == self.last_recording_it and dones[0] and frames is None:
                        frames = self.env.get_complete_frames() or self.env.video_frames

                    new_success = torch.sum(dones.to(dtype=torch.float))
                    num_success += new_success
                    num_new_traj = new_success

                    tot_rew += infos['rew_buf'].sum()
                    tot_y_dist_rew += infos['y_dist_rew'].sum()
                    tot_wall_dist_rew += infos['closest_wall_dist_rew'].sum()
                    tot_obstacle_rew += infos['obstacle_rew'].sum()
                    tot_goal_rew += infos['goal_rew'].sum()

                    obs, obs_history = obs_dict["obs"], obs_dict["obs_history"]
                    obs, obs_history = obs.to(self.device), obs_history.to(self.device)

                    if "privileged_obs" in obs_dict:
                        privileged_obs = obs_dict["privileged_obs"]
                        privileged_obs = privileged_obs.to(self.device)

                    if "obs_vel" in obs_dict:
                        obs_vel = obs_dict["obs_vel"]
                        obs_vel = obs_vel.to(self.device)

                    obs, obs_history = obs_dict["obs"], obs_dict["obs_history"]

                    obs, obs_history, rewards, dones = obs.to(self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                num_failures = torch.logical_not(failures).to(dtype=torch.float).sum()
                success_rate = 100 * num_success / (num_success + num_failures)
                ep_rewards_mean = tot_rew / num_traj
                ep_y_dist_rews_mean = tot_y_dist_rew / num_traj
                ep_wall_dist_rews_mean = tot_wall_dist_rew / num_traj
                ep_obstacle_rew_mean = tot_obstacle_rew / num_traj
                ep_goal_rews_mean = tot_goal_rew / num_traj
                total_dones.append(success_rate)
                total_rews_buf.append(ep_rewards_mean)
                y_dist_rews_buf.append(ep_y_dist_rews_mean)
                wall_dist_rews_buf.append(ep_wall_dist_rews_mean)
                obstacle_rews_buf.append(ep_obstacle_rew_mean)
                ep_goal_rews_buf.append(ep_goal_rews_mean)
                num_traj_buf.append(num_traj)

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                # self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])
                self.alg.compute_returns(obs_history_vel[:num_train_envs])

                if it == self.last_recording_it and frames is None:
                    frames = self.env.get_complete_frames() or self.env.video_frames

                self.env.reset()

            print(f'\n\nSuccess Rates: {success_rate}%')
            print(f'Reward Means: {ep_rewards_mean}')
            print(f'Y Dist Rews: {ep_y_dist_rews_mean}')
            print(f'Wall Dist Rews: {ep_wall_dist_rews_mean}')
            print(f'Obstacle Rews: {ep_obstacle_rew_mean}')
            print(f'Goal Rews: {ep_goal_rews_mean}')
            print(f'Num Trajectories: {num_traj}')

            if it == 0:
                with open(os.path.join(model_root_path, 'success_rate.txt'), 'a') as f:
                    f.write(f'Success Rate,\t\tEpisode Reward,\tGoal Distance Rew,\t\tWall Dist Rew,\t\tObstacle Rwe,\t\tGoal Rew\n')

            if (it + 1) % 30:
                with open(os.path.join(model_root_path, 'success_rate.txt'), 'a') as f:
                    for success_rate, total_rew, y_dist_rew, wall_dist_rew, obstacle_rew, ep_goal_rew, num_traj in zip(total_dones, total_rews_buf, y_dist_rews_buf, wall_dist_rews_buf, obstacle_rews_buf, ep_goal_rews_buf, num_traj_buf):
                        f.write(f'{success_rate},\t\t{total_rew},\t\t{y_dist_rew},\t\t{wall_dist_rew},\t\t{obstacle_rew}, \t\t{ep_goal_rew}, \t\t{num_traj}\n')
                total_dones = []
                total_rews_buf = []
                y_dist_rews_buf = []
                wall_dist_rews_buf = []
                obstacle_rews_buf = []
                ep_goal_rews_buf = []

            results = self.alg.update()

            if self.isTorque:
                mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = results

            else:
                mean_value_loss, mean_surrogate_loss, mean_decoder_loss, mean_decoder_loss_student, mean_decoder_test_loss, mean_decoder_test_loss_student = results
                mean_adaptation_module_loss = 0
                mean_adaptation_module_test_loss = 0
            stop = time.time()
            learn_time = stop - start

            logger.store_metrics(
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            )


            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()

            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    if self.isTorque:
                        adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                        adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                        traced_script_adaptation_module = torch.jit.script(adaptation_module)
                        traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    #logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            self.current_learning_iteration += num_learning_iterations

            if RunnerArgs.save_video_interval:
                if (it + 1) % RunnerArgs.save_video_interval == 0:
                    self.start_video_recording(it + 1)

                if it == self.last_recording_it:
                    self.log_video(it, frames)

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = './tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            if self.isTorque:
                adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                traced_script_adaptation_module = torch.jit.script(adaptation_module)
                traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)


    def start_video_recording(self, it):
        self.env.start_recording()
        self.env.complete_video_frames = []
        if self.env.num_eval_envs > 0:
            self.env.start_recording_eval()
        print("START RECORDING")
        self.last_recording_it = it

    def log_video(self, it, frames):
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
