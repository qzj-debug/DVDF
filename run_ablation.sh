#!/bin/bash

newTmuxSession(){ #new tmux session
    session=$1
    tmux has-session -t $session 2>/dev/null
    if [ $? == 0 ]; then
        echo "Session $session already exists"
        tmux kill-session -t $session
        tmux new-session -d -s $session
        echo "Session $session created done"
    else
        tmux new-session -d -s $session
        echo "Session $session created done"
    fi
}

#support "halfcheetah-kinematic-footjnt" "hopper-kinematic-footjnt" "walker2d-kinematic-footjnt"
#"halfcheetah-morph-thigh" "hopper-morph-torso" "walker2d-morph-leg"

envs=("hopper-kinematic")
#tasks=("halfcheetah-random-v2" "halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-random-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-random-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
#("antmaze-umaze-v0","antmaze-umaze-diverse-v0","antmaze-medium-diverse-v0","antmaze-medium-play-v0","antmaze-large-diverse-v0","antmaze-large-play-v0")
#("pen-human-v0","pen-cloned-v0","pen-expert-v0","hammer-human-v0","hammer-cloned-v0","hammer-expert-v0","door-human-v0","door-cloned-v0","door-expert-v0","relocate-human-v0","relocate-cloned-v0","relocate-expert-v0")

# support "random" "expert" "medium" "medium-replay" "random-expert" "medium-expert"
srctypes=("expert" "medium")

seeds=("100" "200" "300")
device="cuda:0"
algo="DV_IGDF"
save_model="False"
filter_alpha="0.8"
filter_beta="0.8"
dir="./ablationlogs/alpha08beta1"

for env in "${envs[@]}"
do  
    for srctype in "${srctypes[@]}"
    do
        for seed in "${seeds[@]}"
        do  
            tmux_name="${algo}_${env}_${srctype}_${seed}"
            newTmuxSession ${tmux_name}
            tmux send -t ${tmux_name} "cd /data/qzj/DVDF" C-m
            tmux send -t ${tmux_name} "conda activate o2o" C-m
            tmux send -t ${tmux_name} "python my_train.py --algo=${algo} --env=${env} --srctype=${srctype} --seed=${seed} --device=${device} --save-model=${save_model} --filter_alpha=${filter_alpha} --filter_beta=${filter_beta} --dir=${dir}" C-m
        done
    done
done
