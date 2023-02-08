# Line Art Colorization
This project is a minimalistic implementation of AlacGan and it is based on the paper called User-Guided Deep Anime Line Art Colorization with Conditional Adversarial Networks (https://arxiv.org/pdf/1808.03240.pdf) as well as its github repository.

## Try it in Huggingface Spaces
[![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/N1ckQt/line-art-colorization)

## Differences from the original implementation
1. Less variants of line thickness (as it did not make the model performance significantly worse)
2. Different images in the dataset
3. No local features network added
4. All image pairs are acquired via the xdog algorithm whereas in the paper, real line art images were also used to train the model

Because of these differences, the results are slightly worse but the model was trained significantly faster and the process of collecting data did not take too long.

## Model weights
https://download938.mediafire.com/nd1xp1xdgitg/aig8n36f4vrne6t/gen_373000.pth

## Colorization examples
All these images were colorized by the alacgan neural network

![Results of colorization](https://i.imgur.com/qngw4BI.png)

