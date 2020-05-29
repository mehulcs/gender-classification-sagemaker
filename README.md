# Gender Classification

## Training in local
1. Create a directory at root level of this project named dataset.
2. Add 2 sub-directories named `men` and `women` containing respective images for men and women.
3. Run `train.py` for training.
4. Run `test.py` file for output.
```bash
# Trainig
python train.py

# Prediction
python test.py -i <path to input image>
```

## Deploying to Sagemaker
1. Login using AWS SDK in CLI.
2. Run `buildpush.sh` file located inside `container` directory.
```bash
cd container
./buildpush.sh <image name>
```
This script accepts one argument for image name. Image is generated using `Dockerfile` located at `/container`. After building image, Image will be pushed to `Amazon ECR`.

Head over to `Amazon Sagemaker` , create instance and specify to path to image pushed in above script. This image can be used for training and creating endpoints.