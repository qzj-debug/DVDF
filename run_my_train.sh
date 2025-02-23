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
envs=("halfcheetah-kinematic-footjnt")

#tasks=("halfcheetah-random-v2" "halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-random-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-random-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
#("antmaze-umaze-v0","antmaze-umaze-diverse-v0","antmaze-medium-diverse-v0","antmaze-medium-play-v0","antmaze-large-diverse-v0","antmaze-large-play-v0")
#("pen-human-v0","pen-cloned-v0","pen-expert-v0","hammer-human-v0","hammer-cloned-v0","hammer-expert-v0","door-human-v0","door-cloned-v0","door-expert-v0","relocate-human-v0","relocate-cloned-v0","relocate-expert-v0")

#support easy medium hard
shift_levels=("hard")

# support random expert, medium
srctypes=("random" "medium" "expert")

seeds=("100" "200" "300")
device="cuda:7"
algo="IQL"
save_model="True"
mode="3"

for env in "${envs[@]}"
do  
    for shift_level in "${shift_levels[@]}"
    do
        for srctype in "${srctypes[@]}"
        do
            for seed in "${seeds[@]}"
            do  
                tmux_name="baseline_${algo}_${env}_${shift_level}_${srctype}_${seed}"
                newTmuxSession ${tmux_name}
                tmux send -t ${tmux_name} "source ~/.bashrc" C-m
                tmux send -t ${tmux_name} "cd /data/qzj/DVDF" C-m
                tmux send -t ${tmux_name} "conda activate o2o" C-m
                tmux send -t ${tmux_name} "python my_train.py --mode=${mode} --policy=${algo} --env=${env} --shift_level=${shift_level} --srctype=${srctype} --tartype=${srctype} --seed=${seed} --device=${device} --save-model=${save_model}" C-m
            done
        done
    done
done
