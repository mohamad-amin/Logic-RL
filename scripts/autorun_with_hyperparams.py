import argparse
from copy import deepcopy
from datetime import date
import math
from pathlib import Path
import subprocess
import yaml


BASE_CONFIG = {
    'allocation_code': 'zhiyuanli-gpu',
    'ngpus': 8,
    'model_path': 'Qwen/Qwen2.5-7B-Instruct',
    'data_train_name': 'kk_34/instruct/train.parquet',
    'data_test_name': 'kk_34/instruct/test.parquet',
    'train_batch_size': 8,
    'val_batch_size': 32,
    'max_response_length': 2048,
    'actor_optim_lr': 4e-7,
    'ppo_mini_batch_size': 8,
    'ppo_micro_batch_size': 8,
    'kl_loss_coef': 0.001,
    'rollout_n': 8,
    'total_epochs': 2,
    'name': 'Qwen-7B_logic_KK',
    'run_dir': 'out',
    'job_index': 0,
    'save_freq': 100,
    'test_freq': 9999999,
    'format_reward': 0.5,
    'answer_reward': 1.0,
    'hesitation_reward': 0.0,
    'partial_answer_reward': 0.25,
    'wrong_answer_reward': 0.0,
}


TTIC_SUBMIT_SCRIPT = '''#!/bin/sh

#SBATCH --nodes=1
#SBATCH -p {allocation_code}
#SBATCH -G{ngpus}
#SBATCH -C48g,ada
#SBATCH --job-name={job_index}.{name}
#SBATCH --output={run_dir}/slurm_output_{job_index}.txt

zsh
wandb login a2ec141c7427b0c7830f3493d822be104c0cd6e3

set -x
MODEL_PATH={model_path}

export HF_HOME=/tmp

ray start --head --port 6397 --temp-dir /tmp/ray_job_{job_index}_{name}
export RAY_ADDRESS="127.0.0.1:6397"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/{data_train_name} \
    data.val_files=data/{data_test_name} \
    data.train_batch_size={train_batch_size} \
    data.val_batch_size={val_batch_size} \
    data.max_prompt_length=600 \
    data.max_response_length={max_response_length} \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr={actor_optim_lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size={ppo_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n={rollout_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef={kl_loss_coef} \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name={name} \
    +trainer.specific_tag={specific_tag} \
    trainer.n_gpus_per_node={ngpus} \
    trainer.nnodes=1 \
    trainer.default_local_dir={run_dir} \
    trainer.default_hdfs_dir=null \
    trainer.save_freq={save_freq} \
    trainer.test_freq={test_freq} \
    custom_reward_function.path=verl/utils/reward_score/kk.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.format_reward={format_reward} \
    +custom_reward_function.reward_kwargs.answer_reward={answer_reward} \
    +custom_reward_function.reward_kwargs.hesitation_reward={hesitation_reward} \
    +custom_reward_function.reward_kwargs.partial_answer_reward={partial_answer_reward} \
    +custom_reward_function.reward_kwargs.wrong_answer_reward={wrong_answer_reward} \
    trainer.total_epochs={total_epochs} $@ 2>&1

'''


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_config(config, config_path):
    """Save configuration to a YAML file.
    
    Args:
        config (dict): Configuration dictionary to save
        config_path (str): Path where to save the YAML file
    """
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=None)


def add_explanation(explanation, key, value):
    return explanation + ('' if explanation == '' else '-__-') + '{}={}'.format(key, value)


def make_and_get(parent, child):
    if child not in parent:
        parent[child] = {}
    return parent[child]


def update_attribute_in_config(base_config, attribute, new_value, inplace=False):
    config = deepcopy(base_config)
    if attribute in config:
        config[attribute] = new_value
    else: 
        raise ValueError(f'Attribute {attribute} not found in config!')
    return config


def generate_configs(base_config, hyperparam_config):
    def extract_configs_from_attribute_tree(tree, key, base_config, explanation):
        for branch in tree.keys():
            yield from extract_configs_from_branch(tree[branch], key, base_config, explanation=explanation)

    def extract_configs_from_branch(tree, parent_key, base_config, explanation):

        for i, value in enumerate(tree['__value']):

            new_config = update_attribute_in_config(base_config, parent_key, value)
            new_explanation = add_explanation(explanation, parent_key, value)
            correspondings = tree.get('__correspondings', None)
            if correspondings is not None:
                corr_keys = list(correspondings.keys())
                for attr in corr_keys:
                    val = correspondings[attr]['__value'][i]
                    new_config = update_attribute_in_config(new_config, attr, val)
                    new_explanation = add_explanation(new_explanation, attr, val)

            has_children = False
            for key in filter(lambda k: not k.startswith('__'), tree.keys()):
                has_children = True
                yield from extract_configs_from_attribute_tree(
                    tree[key], key, new_config, explanation=new_explanation)

            if not has_children:
                yield new_config, new_explanation
        

    if hyperparam_config is not None:
        for key in hyperparam_config.keys():
            yield from extract_configs_from_attribute_tree(hyperparam_config[key], key, base_config, explanation='')
    else:
        yield base_config, 'single_run'


def construct_task_name(explanation):
    name = ''

    def cond_query(label, test_query, true_val, false_val):
        if (label + "=" + test_query) in explanation:
            return true_val
        else:
            return false_val

    def extract_query(label, addition='', postval='_'):
        l = label + '='
        if l in explanation:
            index = explanation.index(l) + len(l)
            exp_begin = explanation[index:]
            if '-__-' in exp_begin:
                exp_end = exp_begin.index('-__-')
                val = exp_begin[:exp_end]
            else:
                val = exp_begin
            return addition + val + postval
        return addition + '???' + postval
    
    if 'data_train_name' in explanation:
        name += extract_query('data_train_name', addition='trainf')

    if 'data_test_name' in explanation:
        name += extract_query('data_teest_name', addition='testf')
        
    if 'ngpus' in explanation:
        name += extract_query('ngpus', addition='ngpus')  # seed

    if 'train_batch_size' in explanation:
        name += extract_query('train_batch_size', 'tbs')

    if 'val_batch_size' in explanation:
        name += extract_query('train_--_lr_--_max', addition='maxlr')

    if 'max_response_length' in explanation:
        name += extract_query('max_response_length', addition='maxseq')

    if 'actor_optim_lr' in explanation:
        name += extract_query('actor_optim_lr', addition='aolr')

    if 'ppo_mini_batch_size' in explanation:
        name += extract_query('ppo_mini_batch_size', addition='pminbs')

    if 'ppo_micro_batch_size' in explanation:
        name += extract_query('ppo_micro_batch_size', addition='pmicbs')

    if 'kl_loss_coef' in explanation:
        name += extract_query('kl_loss_coef', addition='klc')

    if 'rollout_n' in explanation:
        name += extract_query('rollout_n', addition='rolln')

    if 'total_epochs' in explanation:
        name += extract_query('total_epochs', addition='te')

    if 'job_index' in explanation:
        name += extract_query('job_index', addition='ind')

    if 'format_reward' in explanation:
        name += extract_query('format_reward', addition='fr')

    if 'partial_answer_reward' in explanation:
        name += extract_query('partial_answer_reward', addition='par')
    
    if 'answer_reward' in explanation:
        name += extract_query('answer_reward', addition='ar')
    
    if 'hesitation_reward' in explanation:
        name += extract_query('hesitation_reward', addition='hr')

    if 'wrong_answer_reward' in explanation:
        name += extract_query('wrong_answer_reward', addition='war')

    return name


if __name__ == '__main__':

    print('Warning! Should be run inside *root* directory of the project.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparams_config_path', default='', help='Hyperparameters to choose from')
    parser.add_argument('--allocation_code', default='', help='Script allocation code')
    parser.add_argument('--user_mail', default='', help='Script user mail')
    parser.add_argument('--repetition_num', default=1, type=int, help='Repeation Count')
    parser.add_argument('--time_hrs', default=2, type=int, help='Time in hours')
    parser.add_argument('--ttic', dest='ttic', default=False, action='store_true')
    parser.add_argument('--narval', dest='narval', action='store_true')
    parser.add_argument('--directory_modifier', default='', help='Storage directory of results modification')
    parser.add_argument('--ngpus', default=4, type=int, help='Number of GPUs (nodes as well)')
    parser.add_argument('--ncpus', default=6, type=int, help='Number of CPUs per task')
    parser.add_argument('--mem', default='80G', help='Memory per node')
    parser.set_defaults(pbs=False, ttic=False, narval=True, force_random=False, sbatch=True, eval=False)
    args = parser.parse_args()

    base_config = BASE_CONFIG
    hyperparams_config = load_config(args.hyperparams_config_path)
    new_configs = list(generate_configs(base_config, hyperparams_config))

    today = date.today().strftime('%Y_%m_%d')
    root_path = 'jobs/' + today + args.directory_modifier
    path = Path(root_path)
    path.mkdir(parents=True)

    for config, explanation in new_configs:

        name = construct_task_name(explanation)
        job_path = path.joinpath(name)
        job_path.mkdir()
        config_path = str(job_path.joinpath('config.yml').absolute())

        save_config(config, config_path)

        for i in range(args.repetition_num):
            script_path = str(job_path.joinpath('script_{}.sh'.format(i)).absolute())
            if args.ttic:
                script = TTIC_SUBMIT_SCRIPT
            else:
                # script = NARVAL_SUBMIT_SCRIPT
                raise NotImplementedError('Narval is not implemented yet!')
            time_hrs = args.time_hrs
            # time_hrs = 8
            script = script.format(
                walltime=time_hrs,
                walldays=int(math.floor(time_hrs / 24)),
                wallhours=int(time_hrs % 24),
                ncpus=args.ncpus,
                ngpus=args.ngpus,
                mem=args.mem,
                allocation_code=args.allocation_code,
                user_mail=args.user_mail,
                config_path=config_path,
                load_dir=(lambda path: str(path.absolute()))(job_path.parent.parent.parent if args.eval else job_path),
                run_dir=str(job_path.absolute()),
                name=name,
                job_index=i,
                specific_tag=args.directory_modifier[1:],
                model_path=config['model_path'],
                data_train_name=config['data_train_name'],
                data_test_name=config['data_test_name'],
                train_batch_size=config['train_batch_size'],
                val_batch_size=config['val_batch_size'],
                max_response_length=config['max_response_length'],
                actor_optim_lr=config['actor_optim_lr'],
                ppo_mini_batch_size=config['ppo_mini_batch_size'],
                ppo_micro_batch_size=config['ppo_micro_batch_size'],
                kl_loss_coef=config['kl_loss_coef'],
                rollout_n=config['rollout_n'],
                total_epochs=config['total_epochs'],
                save_freq=config['save_freq'],
                test_freq=config['test_freq'],
                format_reward=config['format_reward'],
                answer_reward=config['answer_reward'],
                hesitation_reward=config['hesitation_reward'],
                partial_answer_reward=config['partial_answer_reward'],
                wrong_answer_reward=config['wrong_answer_reward'],  
            )
            with open(script_path, 'w') as f:
                f.write(script)

    inp = input('Created all the files! Press any key to submit them...\n')

    for _, explanation in new_configs:
        name = construct_task_name(explanation)
        job_path = path.joinpath(name)
        if args.repetition_num > 1:
            for i in range(args.repetition_num):
                script_path = str(job_path.joinpath('script_{}.sh'.format(i)).absolute())
                subprocess.check_call(['sbatch', '-J', name, '-d', 'singleton', script_path])
        else:
            script_path = str(job_path.joinpath('script_{}.sh'.format(0)).absolute())
            subprocess.check_call(['sbatch', script_path])