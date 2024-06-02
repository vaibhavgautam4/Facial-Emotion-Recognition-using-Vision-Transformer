# Facial Emotion Recognition using Vision Transformer
 
Spearheaded the creation of a facial emotion detection model using VIT and MobileNet architectures.
we have tried and tested VIT transfer model to learn features and able to recognize them with more accuracy and precision.
A Model that can Understand and Extract Human emotions from a given image efficiently.

- Demonstrated proficiency in computer vision, leading the development of the emotion detection model.

- Looking forward to Translate model capabilities into a practical web application,enabling real-time emotion analysis for users.

### Installation
- Create a Conda environment or a Virtual Environment based on your convinience.

- Open terminal in the directory where the environment is created or can use CLI from any Code Editor terminal.

- make sure you have ``` Python>=3.10 ``` installed on your machine.

- install other dependencies and libraries using the command:

```
pip install requirement.txt
```

- Once all the requirements are installed run the **vision_transformer_and_other_cnn's.ipynb** by connecting to the kernel you made and running cells according to requirements.

- As the structure may be not in sequence, you are advised to check your requirements and run the files accordingly.

## Vision Transformer

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.

**Facial emotion recognition (FER)** is a significant task in the field of computer vision, where the objective is to detect and classify human emotions based on facial expressions. Traditionally, convolutional neural networks (CNNs) have been the predominant architecture for this task due to their ability to capture spatial hierarchies in images. However, with the advent of Vision Transformers (ViTs), there's a new paradigm that leverages the self-attention mechanism, which has shown promising results in various vision tasks.

Transformers, initially designed for natural language processing (NLP), have been adapted for vision tasks. The Vision Transformer (ViT) is a type of transformer architecture specifically tailored for image recognition tasks. Instead of using convolutions, ViTs split an image into patches, linearly embed these patches, and then process the resulting sequence of embeddings using transformer layers.

The key components of ViTs include:

- Patch Embedding: The input image is divided into fixed-size patches, which are then flattened and linearly embedded.
- Positional Encoding: Since transformers are permutation-invariant, positional encodings are added to the patch embeddings to retain spatial information.
- Transformer Encoder: The embedded patches, along with their positional encodings, are passed through multiple layers of transformer encoders that utilize multi-head self-attention and feed-forward neural networks.
- Classification Head: The output from the transformer encoder is typically passed through a classification head to produce the final prediction.
Vision Transformers for Facial Emotion Recognition

Using Vision Transformers for FER involves adapting the ViT architecture to effectively handle the nuances of facial expressions. Here are the steps typically involved in employing ViTs for FER:

- Data Preparation: Collect and preprocess a dataset of facial images labeled with corresponding emotions. Popular datasets include FER-2013, AffectNet, and CK+.
- Patch Extraction: Divide each facial image into fixed-size patches (e.g., 16x16 pixels). Each patch is then flattened into a vector.
Embedding and Positional Encoding: Embed the patches into a higher-dimensional space and add positional encodings to each patch embedding to retain the spatial arrangement.
- Transformer Encoding: Pass the sequence of patch embeddings through transformer layers. The self-attention mechanism allows the model to learn global relationships between patches, which is particularly useful for capturing complex facial expressions.
- Emotion Classification: Use a classification head to predict the emotion from the encoded patches. This head can be a simple fully connected layer or a more complex structure tailored for FER.

Can read more about Vision Transformers in [A survey on vision transformer(Han, K., Wang, Y., Chen, H., Chen, X., Guo, J., Liu, Z., ... & Tao, D. (2022). A survey on vision transformer. IEEE transactions on pattern analysis and machine intelligence, 45(1), 87-110.)](https://bibbase.org/service/mendeley/bfbbf840-4c42-3914-a463-19024f50b30c/file/09259b3f-fe6b-20cd-52c1-690f40c13ce6/full_text.pdf.pdf).

![Figure 1 from paper](vit_figure.png)

### Available ViT models

A variety of ViT models in different GCS buckets can be used and The models can be
downloaded with e.g.:

```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

The model filenames (without the `.npz` extension) correspond to the
`config.model_name` in [`vit_jax/configs/models.py`]

- [`gs://vit_models/imagenet21k`] - Models pre-trained on ImageNet-21k.
- [`gs://vit_models/imagenet21k+imagenet2012`] - Models pre-trained on
  ImageNet-21k and fine-tuned on ImageNet.
- [`gs://vit_models/augreg`] - Models pre-trained on ImageNet-21k,
  applying varying amounts of [AugReg]. Improved performance.
- [`gs://vit_models/sam`] - Models pre-trained on ImageNet with [SAM].
- [`gs://vit_models/gsam`] - Models pre-trained on ImageNet with [GSAM].

### Datasets Used

- [FER 2013](https://www.kaggle.com/datasets/msambare/fer2013) Dataset is used for the training of model.
