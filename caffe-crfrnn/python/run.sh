$#!/usr/bin/env sh
# This script converts the net data to mat file
# if permission deny: chmod +x file_name
DBTYPE=lmdb
MODEL2=fcn32
MODEL1=fcn8
IMAGE1=child
IMAGE2=flower
IMAGE3=dog

echo "RUNNING $MODEL..."

#python net2mat_sege.py  $MODEL1  $IMAGE
python net2mat_sege.py  $MODEL1 $IMAGE3
python net2mat_sege.py  $MODEL1 $IMAGE2
python net2mat_sege.py  $MODEL1 $IMAGE1

 echo "Done."
