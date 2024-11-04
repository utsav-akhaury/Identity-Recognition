# Identity Recognition using Images and Audio Clips

For the general concept, please see `docs/presentation.pdf`. 

## Training notebooks

Our networks were trained on Google Colab, see the directory `training_notebooks`.  Their weights are stored in the `weights` directory.

## Networks

The following files contain the codes of our implemented networks in the Image classification pipeline. We basically obatined the source code from the Keras GitHub repo, replaced all Conv2D layers by SeparableConv2D layers, and slightly modified the original networks to reduce the number of parameters.

- `ResNet.py`
- `InceptionNet.py`

## Preprocessing

`preprocessing.ipynb` runs the benchmark code to calculate the equivalent number of parameters involved in the audio and image preprocessing stages.

## Pieces of the processing pipeline

>  <strong>If you want to, you can skip immediately to `3_combine.py` as all the embeddings are already included in this directory</strong>

The different python files at the root of the directory implement the general pipeline. After placing the file `audVisIdn.npz` in the directory `datadir`, run the files in order:

- `11_preprocessing_audio.py` computes the spectral features we fit on. Here, we use `librosa`.
- `12_embedding_audio.ipynb` runs the clustering network whose weights are in `weights/audio_clusterer_trimmed.h5` on the spectral features extracted before.
- `21_preprocessing_image.py` detects faces and crops them out of the original picture.
- `22_embedding_image.ipynb`  runs the InceptionResnetV2 with weights `weights/28epochscrosscat_30epochstriplet_inceptionresnet_smaller_sepconv.h5` to create embeddings of the pictures.
- `3_combine.py` trains a dense network on the audio and image embeddings created in the previous steps, and outputs the test accuracy. 

The final test accuracy we obtain is 95.4%.
