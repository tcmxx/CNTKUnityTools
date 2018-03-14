
# AI and Deep learning tools for Unity using CNTK

## Content 
This rep contains some useful deep learning related tools implemented primarily using CNTK C# library.
Current contents:
- Helper functions to build/train neural network layers. (https://docs.microsoft.com/en-us/cognitive-toolkit/)
  - Layers definitions
  - Simple neural network
  - cGAN
- Universal Style Transfer(https://arxiv.org/pdf/1705.08086.pdf)
- Reinforcement Learning
  - PPO(https://arxiv.org/pdf/1707.06347.pdf)
  
## Platform and Installation
Currently it only works on Windows. If you need to use GPU for NN, you also need a proper Nvidia graphic card.
Installation steps:
1. Download the repo(Unity project)
2. Download the zip that includes necessary dlls https://drive.google.com/open?id=1VWEiXJw3PSdeXfBrimPevdCWdbUyHd_0
3. Put the dlls in correct places: (Adapted from https://github.com/aiunderstand/unity-cntk/blob/master/README.md.)
- Put those files/folders into any Plugins folder under yourproject/Assets.
    * Cntk.Core.Managed-2.4.dll
    * MathNet.Numerics.dll
    * MathNet.Numerics.MKL.dll
    * System.Drawing.dll
    * Accord folder
- Put other dlls and put them DIRECTLY under yourproject folder, or another place where Windows can find those dlls.
4. Done.

Note that the file Assets/UnityCNTK/Tools/UniversalStyleTransfer/Data/UST_combined.bytes uses Git LFS, be sure you download it correctly (It should be larger than 100MB)
