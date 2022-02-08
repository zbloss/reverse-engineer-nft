# reverse-engineer-nft
A project aimed at reverse-engineering trait-based NFTs. Many of the most popular NFT projects are known as trait-based projects. Think [CryptoPunks](https://www.larvalabs.com/cryptopunks), [Bored Apes](https://opensea.io/collection/boredapeyachtclub), or the NFTs used in this project, [Meebits](https://meebits.larvalabs.com/). The designers of these projects work extremely hard to first build assets: hats, skin colors, shirts, & other various accessories. Then they algorithmically (man I hate that word) generate a massive batch of these NFTs.

With this project, my goal was to start with those attributes, and reverse engineer the image NFT. This could then be used to recreate existing NFTs as well as to create brand new ones not seen in the original collection! An example input may look something like:

```json
{
  "name": "Meebit #1",
  "description": "Meebit #1",
  "image": "http://meebits.larvalabs.com/meebitimages/characterimage?index\u003d1\u0026type\u003dfull\u0026imageType\u003djpg",
  "attributes": [
    {
      "trait_type": "Type",
      "value": "Human"
    },
    {
      "trait_type": "Hair Style",
      "value": "Bald"
    },
    {
      "trait_type": "Hat",
      "value": "Backwards Cap"
    },
    {
      "trait_type": "Hat Color",
      "value": "Gray"
    },
    {
      "trait_type": "Shirt",
      "value": "Skull Tee"
    },
    {
      "trait_type": "Overshirt",
      "value": "Athletic Jacket"
    },
    {
      "trait_type": "Overshirt Color",
      "value": "Red"
    },
    {
      "trait_type": "Pants",
      "value": "Cargo Pants"
    },
    {
      "trait_type": "Pants Color",
      "value": "Camo"
    },
    {
      "trait_type": "Shoes",
      "value": "Workboots"
    }
  ]
}

```

With the target image being:

![Meebit #1](./images/1.png)


## Getting Started

First to generate the Metadata JSON object, you will run `get_metadata.py`. This will provide you a JSON file to feed into the `make_dataset.py` script. At that point you should have an obnoxious amount of NFTs and their corresponding metadata downloaded to your computer.

## The Model

The model architecture was heavily inspired from the original StyleGAN, but with a few tweaks. The first being, this isn't a GAN at all. Instead of starting from complete noise, I begin with the attributes. These attributes are encoded into a `[batch_size, 1, 21]` array then fed into an 8-layer Fully Connected neural network which scales them to `[batch_size, 1, 1024]`. Then a constant is applied to this array before it is fed into a series of Upsampling and Convolutional Layers, resulting in the Meebits original `[batch_size, 3, 1536, 1024]` size.

I have seen some really neat Neural Network architecture diagram tools floating around online lately. Once I get the time to sit down and create one I'll drop it here.

This architecture is fairly specific to this particular NFT project, but the same idea applies for scaling this model to other NFT projects. Most of the `fiddling` will come in making sure the output size matches the projects.