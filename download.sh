"""
Implementation of MAML, Prototypical Network with Pytorch. Neptune logging supported
"""

FILE=$1

if [ $FILE == "miniimagenet" ]; then
    URL = https://www.dropbox.com/s/vlnjhlm0bn8zqk1/miniimagenet.zip?dl=0
    ZIP_FILE=./data/miniimagenet.zip 
    mkdir -p ./data 
    wget -N $URL -0 $ZIP_FILE 
    unzip $ZIP_FILE -d ./data 
    rm $ZIP_FILE

else
    echo "Available arguments are miniimagenet for now"
    exit 1
fi