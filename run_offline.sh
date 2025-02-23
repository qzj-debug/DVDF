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

envs=("walker2d")
#tasks=("halfcheetah-random-v2" "halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-random-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-random-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
#("antmaze-umaze-v0","antmaze-umaze-diverse-v0","antmaze-medium-diverse-v0","antmaze-medium-play-v0","antmaze-large-diverse-v0","antmaze-large-play-v0")
#("pen-human-v0","pen-cloned-v0","pen-expert-v0","hammer-human-v0","hammer-cloned-v0","hammer-expert-v0","door-human-v0","door-cloned-v0","door-expert-v0","relocate-human-v0","relocate-cloned-v0","relocate-expert-v0")

srctypes=("expert")

seeds=("100" "200" "300")
device="cuda:0"
algo="IQL"
save_model="True"

for env in "${envs[@]}"
do  
    for srctype in "${srctypes[@]}"
    do
        for seed in "${seeds[@]}"
        do  
            tmux_name="offline_${algo}_${env}_${srctype}_${seed}"
            newTmuxSession ${tmux_name}
            tmux send -t ${tmux_name} "source ~/.bashrc" C-m
            tmux send -t ${tmux_name} "cd /data/qzj/DVDF" C-m
            tmux send -t ${tmux_name} "conda activate o2o" C-m
            tmux send -t ${tmux_name} "python train_offline.py --policy=${algo} --env=${env} --srctype=${srctype} --seed=${seed} --device=${device} --save_model=${save_model}" C-m
        done
    done
done
