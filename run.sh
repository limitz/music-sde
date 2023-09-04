ps -aux | grep 'python' | grep 'multiprocessing' | awk '{ print $2 }' | xargs -n1 kill -9 2> /dev/null
#export NCCL_DEBUG=INFO
#export NCCL_ASYNC_ERROR_HANDLING=1

# FOR CLUSTERS, synchronize via Git

#if [[ -n $1 && $1 != 0 ]]; then
#    git pull origin `git branch --show-current`
#    export DIST_OFFSETRANK=$1
#    export DIST_HOSTNAME=`cat .dist/hostname`
#else
#    export DIST_OFFSETRANK=0
#    export DIST_HOSTNAME=`hostname`
#    mkdir -p .dist
#    hostname > .dist/hostname
#fi
#if [[ -n $2 ]]; then
#    export DIST_WORLDSIZE=$2
#    git add .dist/hostname
#    git commit -a -m "Synchronize cluster"
#    git push origin `git branch --show-current`
#else
#    export DIST_WORLDSIZE=${DIST_NPROCS}
#fi
#export DIST_PORT=23456
#env | grep DIST

python main.py 
