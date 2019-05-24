# Convolutional Autoencoder

This is a stand-alone program that trains a convolutional autoencoder on pre-processed and post-processed MRI scans. The scans are from the ADNI dataset, and the post-processed scans are processed with FreeSurfer. The goal of the autoencoder is to approximate the post-processing done with FreeSurfer, which eliminates unnecessary elements from the MRI, such as the neck, the skull, and other undesirable features.

## Notes

1. At least 16 images per batch can be run on 1 GTX1080 TI GPU with the VanillaCAE model,

2. A mapping file is needed to run. The mapping file contains path mapping between preprocess and postprocessed images, and can be generated with the script in `utils/mapping.py`. Be warned that running this script will take a look time, as it also opens each file and checks for error, so that corrupted files are excluded, and it does so single-threaded. Alternatively, you can find the `mapping_manifest.pickle` in `/mnt/nfs/work1/mfiterau/zguan/work1/disease_forecasting/src/cae/outputs/`.

### Current Setup on Gypsum

All numbers depend on the number of GPUs available, therefore adjust accordingly.

1. 2 GPUs,

2. 16 images per GPU = 32 images,

3. 4 DataLoader workers per 1 GPU = 8 workers,

4. 8 workers + 2 extra * 2 GPU = 12 CPUs, (48 CPUs / 8 GPUs = 6 CPUs/GPU)

5. 45GB per 1 GPU = 90,000 (if you use more than 45GB per GPU, someone might be angry)

### Procedures for Adding Model

1. Create a file under `models/`,

2. Import the class into `engine.py`,

3. In the `_setup_model()` method of the `Engine` class, add another `elif` statement. For example,

```python
if model_class == "vanilla_cae":
    ...
# START OF YOUR CODE
elif model_class = "custom_model":
    print("Using custom model.")
    self._model = CustomModel() # Instantiate your model
# END OF YOUR CODE
else:
            raise Exception("Unrecognized model: {}".format(model_class))
```

4. In `config_default.yaml`, update the model/class parameter. For example,

```yaml
model:
    class: custom_class
```

### Tensorboard

To start a tensorboard session, run

`tensorboard --logdir=DIR_TO_LOG --port==SERVER_SIDE_PORT`

To tunnel to the server, run

`ssh -L LOCAL_PORT:127.0.0.1:SERVER_SIDE_PORT -N YOUR_ID@SERVER_ADDRESS`
