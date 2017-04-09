pwd
ls -la


#export PATH=$PATH:/root/cling/bin
NCPUS=`python -c "import multiprocessing as mp; print(mp.cpu_count())"`
echo "Detected $NCPUS cpus"

#python -c "import sys; sys.path.append('/root/inst/bin/')"
export PATH=/root/inst/bin/:$PATH

echo $PATH


#dask-scheduler --host localhost &
#dask-worker localhost:8786 $* &
jupyter notebook --allow-root "$@" &

# run postgress
su postgres -c "/usr/lib/postgresql/9.3/bin/postgres -D /var/lib/postgresql/9.3/main -c config_file=/etc/postgresql/9.3/main/postgresql.conf" &
