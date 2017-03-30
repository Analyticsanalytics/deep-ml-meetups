#!/usr/bin/env bash
# Copyright 2015 deep-ml.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
jupyter notebook "$@" &

# run postgress
su postgres -c "/usr/lib/postgresql/9.3/bin/postgres -D /var/lib/postgresql/9.3/main -c config_file=/etc/postgresql/9.3/main/postgresql.conf" &
