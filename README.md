# Image Analogy

## Project Intro/Objective
This project implemented image analogies described in the paper "Image Analogies" [[1]](#1). Image analogies is a framework that can learn a filter from a pair of training images and use this filter to synthesize new images. It has many applications like artistic filter, improved texture synthesis, or super-resolution.

### Methods Used
* Texture Synthesys
* Texture Transfer
* Ball Tree Nearest Neighbor Search
* Example-based Rendering

### Technologies
* Python
* numpy
* scikit-learn
* opencv
* scipy

## Project Description
To create an image analogy, three input images A, A’, and B are needed. A and A’ are a pair of training images for learning a filter. B is the target image on which the learnt filter will be applied. They together synthesize a new image B’:
<div align="center" style="font-size:15px"><b> A : A’ :: B : B’ </b></div>

### Procedure
Images A, A’, B, and B’ will go through a procedure as follows:
<p align="center">
 <img src="https://drive.google.com/uc?export=view&id=1-3tTcAaKUthUUgxG4A38eGFefqcbw2oF" width="600" height="150">
</p>

### Showcases
#### Super-resolution
<img src="https://drive.google.com/uc?export=view&id=1fy8Alc-8Gs7zzeVwFSREZOFQvwNcbglq" width="600" height="150">

#### Texture Transfer
<img src="https://drive.google.com/uc?export=view&id=1pq-9bv8lUH5bJ6WSKxJz9gMjAKTDFztM" width="600" height="150">

#### Artistic Filter
<img src="https://drive.google.com/uc?export=view&id=1crixK--TBxXqSWizf4G9UvVKDsKfXZ7q" width="600" height="150"><br/>

## Running Project
```bash
	$ python main.py
```

## References
<a id="1">[1]</a>
Hertzmann,A., Jacobs,C.E., Oliver,N., Curless,B., & Salesin,D.H.(2001).<br/>
Image Analogies.<br/>
<em>SIGGRAPH '01 Proceedings of the 28th annual conference on Computer graphics and interactive techniques</em>, 327-340.