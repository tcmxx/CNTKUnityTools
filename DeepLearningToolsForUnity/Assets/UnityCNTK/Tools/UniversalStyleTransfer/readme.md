# Universal Style Transfer

## Overview
Here is the original paper: https://arxiv.org/pdf/1705.08086.pdf

The implementation here is somehow based on this: https://github.com/eridgd/WCT-TF and this:https://github.com/sunshineatnoon/PytorchWCT and this: https://github.com/Yijunmaverick/UniversalStyleTransfer

See https://www.youtube.com/watch?v=v1oWke0Qf1E for a brief explanation.

## Some notes of the algorithm
This section of helpful if you want to want to know what does the parameters mean in this algorithm. Please make sure you understand the youtube video above before you read the following.
### VGG
Universal Style Transfer uses [VGG](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77) as the encoder. The output of different layers of VGG represent the feature map of the input image with different receptive field. For example, a number from a deeper layer's output might mean whether a 64x64 patch on the original image has a dog shape while one from a shallower layer might mean whether a 3x3 patch on the original image has a vertical line.
### Feature Transform
Universal Style Transfer uses Whitening and Coloring operation on the output feature maps from VGG to blend the style and content image. (Whitening and Coloring is nothing about color)

Whitening(https://en.wikipedia.org/wiki/Whitening_transformation/) basically means to apply matrix multipliacations to a bunch of data point so that those data's covariance matrix is identity matrix, so that those data is uncorrelated.

Coloring is the inverse of Whitening. 

UST does whitening on content image's feature maps and color the whitened feature maps with the style image's transform matrices. The new feature map then transfered. The new feature map will be linearly blended with the original content feature map, and then decoded to a output image using the decoder.
### Multi-level Stylization
Universal Style Transfer does the encoding-whitening/coloring-decoding multiple times with different depth of VGG encoder to achieve a better looking result. See the figure below:
![style transfer pipeline](https://github.com/tcmxx/CNTKUnityTools/blob/master/Docs/Images/UST-pipeline.png)

## Usage:
### Unity Editor Tool
In Unity Editor menu go to Window/UnityDeepLearningTools/StyleTransferTool

### C# API

```csharp
//create the model. the styleTransferModelData.bytes is the binary files provided that contains all needed pretrained data of the network.
var styleTransferModel = new UniversalStyleTransferModel(CNTK.DeviceDescriptor.GPUDevice(0), styleTransferModelData.bytes);
//build it with specified dimentions
styleTransferModel.CreateModelWithDimensions(contentSize, styleSize);

// Get the raw data of content and style imagefrom the Unity Texture2D object using helper functions.
var tempContentTexture = Images.GetReadableTextureFromUnreadable(contentTexture);
byte[] contentBytes = tempContentTexture.GetRGB24FromTexture2D(contentResize);

var tempStyleTexture = Images.GetReadableTextureFromUnreadable(styleTexture);
byte[] styleBytes = tempStyleTexture.GetRGB24FromTexture2D(styleResize);
//destroy the tempeary texture if you want.
DestroyImmediate(tempStyleTexture);
DestroyImmediate(tempContentTexture);

//specify the which passes to run and what are the parameters in each pass
//PassIndex 1 means transfer with the VGG deepest level.
var styleTransferParams = new List<UniversalStyleTransferModel.ParameterSet>();
for (int i = 0; i < 5; ++i)
{
	styleTransferParams.Insert(0, new UniversalStyleTransferModel.ParameterSet((UniversalStyleTransferModel.PassIndex)i));
}
//execute the transfer and get the result image data
byte[] result = styleTransferModel.TransferStyle(contentBytes, styleBytes, styleTransferParams.ToArray());
//put it on a new texture
Texture2D tex = new Texture2D(contentResize.x, contentResize.y, TextureFormat.RGB24, false);
tex.LoadRawTextureData(result);
tex.Apply();
```
