# Style Transfer using [GoMLX](github.com/gomlx/gomlx), a machine learning framework for Go

> Note: see demo in the [notebook](https://github.com/janpfeifer/styletransfer/blob/main/demo.ipynb)

Based on [1], the seminal paper on the topic. With some help of the blog post [2]. But it uses the InceptionV3 image model instead of the VGG-19 for style transfer.

All the code is in the repository [github.com/janpfeifer/styletransfer/styletransfer](https://github.com/janpfeifer/styletransfer/styletransfer),
it's only about 400 lines of code, including I/O code (loading and displaying the images).

* [1] "A Neural Algorithm of Artistic Style", 2015, Gatys, Ecker & Bethge -- https://arxiv.org/abs/1508.06576
* [2] "Neural Style Transfer (NST) -- theory and implementation", 2021 -- https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d

There are still tons that can be improved:

* Other style losses.
* Work with higher resolution images.
* Work with videos.
* Style transfer with forward pass only.
* Read and implement newest papers...

## Samples from [Notebook](https://github.com/janpfeifer/styletransfer/blob/main/demo.ipynb)

Stylized GoMLX logo:

![image](https://github.com/user-attachments/assets/5d6ed0b0-7226-459d-af4a-7486482c6d7b)

Image from country side of France, while hiking the [Via Podiensis](https://en.wikipedia.org/wiki/Via_Podiensis):

![image](https://github.com/user-attachments/assets/4800fd8b-e297-4dbd-bec9-13291d758225)
