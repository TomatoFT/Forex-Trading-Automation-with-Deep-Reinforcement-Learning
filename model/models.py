# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
from stable_baselines import PPO2
from stable_baselines import ACKTR
from stable_baselines import TD3
from stable_baselines import DDPG

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleForex_train import ForexEnvTrain
from env.EnvMultipleForex_validation import ForexEnvValidation
from env.EnvMultipleForex_trade import ForexEnvTrade


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def train_TD3(env_train, model_name, timesteps=1000):
    """TD3 Model"""
    start = time.time()
    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3('MlpPolicy',env_train, action_noise=action_noise, verbose=1)
    #model = TD3('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (TD3): ', (end - start) / 60, ' minutes')
    return model
def train_ACKTR(env_train, model_name, timesteps=1000):
    """ACKTR Model"""
    start = time.time()
    model = ACKTR('MlpPolicy', env_train, verbose=1)
    #model = ACKTR('MlpPolicy', env_train, ent_coef = 0.005)
    
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (ACKTR): ', (end - start) / 60, ' minutes')
    return model

def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: ForexEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/{}/last_state_{}_{}.csv'.format(name, name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, TD3 and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    td3_sharpe_list = []
    acktr_sharpe_list = []
    model_use = []

    compare_model=pd.DataFrame({
     'start': [],'end' : [],'TD3 sharpe': [],
     'PPO sharpe': [], 'DDPG sharpe': [], 
     'ACKTR sharpe': [], 'Used Model': []})

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate < 20210701) & (df.datadate>=20180102)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
    turbu=pd.DataFrame()
    turbu['turbulence']=insample_turbulence_threshold
    turbu.to_csv('turbulence.csv')
    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20180102, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: ForexEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: ForexEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20180102, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")

        print("======TD3 Training========")
        model_td3 = train_TD3(env_train, model_name="TD3_30k_result_{}".format(i), timesteps=2000)
        print("======TD3 Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_td3, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_td3 = get_validation_sharpe(i)
        print("TD3 Sharpe Ratio: ", sharpe_td3)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_result_{}".format(i), timesteps=2000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_result_{}".format(i), timesteps=2000)
        #model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)


        print("======ACKTR Training========")
        model_acktr = train_ACKTR(env_train, model_name="ACKTR_10k_result_{}".format(i), timesteps=2000)
        #model_acktr = train_TD3(env_train, model_name="ACKTR_10k_dow_{}".format(i), timesteps=20000)
        print("======ACKTR Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_acktr, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_acktr = get_validation_sharpe(i)

        ppo_sharpe_list.append(sharpe_ppo)
        td3_sharpe_list.append(sharpe_td3)
        ddpg_sharpe_list.append(sharpe_ddpg)
        acktr_sharpe_list.append(sharpe_acktr)

        start_day=unique_trade_date[i - rebalance_window - validation_window]
        end_day=unique_trade_date[i - rebalance_window]

        # Model Selection based on sharpe ratio
        sharpe_list=[sharpe_ppo,sharpe_td3,sharpe_ddpg,sharpe_acktr]
        if sharpe_ppo>=max(sharpe_list):
            model_ensemble = model_ppo
            model_use.append('PPO')
            compare_model.loc[len(compare_model.index)] =[start_day,end_day,sharpe_td3,sharpe_ppo,sharpe_ddpg,sharpe_acktr,'PPO']
        elif sharpe_td3>=max(sharpe_list): 
            model_ensemble = model_td3
            model_use.append('TD3')
            compare_model.loc[len(compare_model.index)] =[start_day,end_day,sharpe_td3,sharpe_ppo,sharpe_ddpg,sharpe_acktr,'TD3']            
        elif sharpe_ddpg>=max(sharpe_list): 
            model_ensemble = model_ddpg
            model_use.append('DDPG')
            compare_model.loc[len(compare_model.index)] =[start_day,end_day,sharpe_td3,sharpe_ppo,sharpe_ddpg,sharpe_acktr,'DDPG']
        else : 
            model_ensemble = model_acktr
            model_use.append('ACKTR')
            compare_model.loc[len(compare_model.index)] =[start_day,end_day,sharpe_td3,sharpe_ppo,sharpe_ddpg,sharpe_acktr,'ACKTR']

        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        #Ensemble method
        last_state_TD3 = DRL_prediction(df=df, model=model_td3, name="TD3",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        last_state_DDPG = DRL_prediction(df=df, model=model_ddpg, name="DDPG",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        last_state_PPO = DRL_prediction(df=df, model=model_ppo, name="PPO",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        last_state_ACKTR = DRL_prediction(df=df, model=model_acktr, name="ACKTR",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)                                                                                                                                                                                    
        # print("============Trading Done============")
        ############## Trading ends ##############
        compare_model.to_csv('compare_model.csv')
    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
