if [ -z $FUEL_DATA_PATH ]; then 
    echo "Please set FUEL_DATA_PATH"
    exit 1
fi

cd $FUEL_DATA_PATH
fuel-download caltech101_silhouettes 28
#fuel-download binarized_mnist

fuel-convert caltech101_silhouettes 28
#fuel-convert binarized_mnist

