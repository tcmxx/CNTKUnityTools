using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
using UnityCNTK;
using System;
using Accord.Math;
using System.Diagnostics;
using MathNet.Numerics;

namespace UnityCNTK.Tools.StyleAndTexture
{
    public class UniversalStyleTransferModel {


        protected CNTK.DeviceDescriptor device;


        protected Function encoderLayersStyle;
        protected Function encoderLayersContent;

        protected List<Function> decoders;

        protected Vector2Int contentDimension;
        protected Vector2Int styleDimension;

        protected byte[] modelWeights;
        
        /// <summary>
        /// encoder or decoder layer in UST. 1 to 5
        /// </summary>
        public enum PassIndex
        {
            PassOne, PassTwo, PassThree, PassFour, PassFive
        }

        [Serializable]
        public class ParameterSet{
            public bool enabled = true;
            public PassIndex Pass;
            public float BlendFactor { get; set; }
            public float Beta { get; set; }
            public float Eps { get; set; }
            public float IgnoreSingularValue { get; set; }
            public ParameterSet(PassIndex pass, float blendFactor = 0.7f)
            {
                Pass = pass;
                BlendFactor = blendFactor;
                Beta = 0.5f;
                Eps = 0.00001f;
                IgnoreSingularValue = 0.00001f;
            }
        }


        public UniversalStyleTransferModel(DeviceDescriptor device, byte[] modelWithWeights)
        {
            this.device = device;
            modelWeights = modelWithWeights;
        }




        public byte[] TransferStyle(byte[] contentRGB24, byte[] styleRGB24, ParameterSet[] passes = null)
        {
            //use intel mkl as math.net linear algebra provider
            Control.UseNativeMKL();

            if (passes == null)
            {
                //default passes parameters
                passes = new ParameterSet[] { new ParameterSet(PassIndex.PassFive), new ParameterSet(PassIndex.PassFour),
                new ParameterSet(PassIndex.PassThree), new ParameterSet(PassIndex.PassTwo),
                new ParameterSet(PassIndex.PassOne)};
            }

            //test for couting time
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            UnityEngine.Debug.Assert(contentRGB24.Length == 3 * contentDimension.x * contentDimension.y, "Input content RGB24 data does not match the model dimensions");
            UnityEngine.Debug.Assert(styleRGB24.Length == 3 * styleDimension.x * styleDimension.y, "Input style RGB24 data does not match the model dimensions");

            float[] content = Images.RGB24ToFloat(contentRGB24);
            float[] style = Images.RGB24ToFloat(styleRGB24);
            
            IList<IList<float>> output = null; //result
            try
            {
                Value inputContent = new Value(new NDArrayView(new int[] { contentDimension.x, contentDimension.y, 3 }, content, device, true));
                Value inputStyle = new Value(new NDArrayView(new int[] { styleDimension.x, styleDimension.y, 3 }, style, device, true));
                stopwatch.Stop();
                //UnityEngine.Debug.Log("Input Conversion  completed. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);
                stopwatch.Restart();
                foreach (var p in passes)
                {
                    if (!p.enabled)
                        continue;
                    UnityEngine.Debug.Log("Start Pass" + ((int)p.Pass + 1));
                    inputContent = ExecutePass( inputContent, inputStyle, p);
                    //output = inputContent.GetDenseData<float>(decoders[(int)passes[passes.Length - 1].Pass].Output);    //test only
                    GC.Collect();
                    stopwatch.Stop();
                    UnityEngine.Debug.Log("Pass" + ((int)p.Pass+1)+"  completed. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);
                    stopwatch.Restart();
                }

                stopwatch.Stop();
                output = inputContent.GetDenseData<float>(decoders[(int)passes[passes.Length - 1].Pass].Output);
            }catch(Exception e)
            {
                UnityEngine.Debug.LogError(e.Message);
            }
            return Images.FloatToRGB24(output[0]);
        }


        protected Value ExecutePass(Value inputContent, Value inputStyle, ParameterSet parameters)
        {
            
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            int pass = (int)parameters.Pass + 1;
            //pass 5
            var inputDataMapContent = new Dictionary<Variable, Value>() { { encoderLayersContent.Arguments[0], inputContent } };
            var outputDataMapContent = new Dictionary<Variable, Value>() { { encoderLayersContent.Outputs[pass-1], null } };
            encoderLayersContent.Outputs[pass - 1].ToFunction(). Evaluate(inputDataMapContent, outputDataMapContent, false, device);
            stopwatch.Stop();
            //UnityEngine.Debug.Log("--Pass " + pass + " encoding content completed. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);

            stopwatch.Restart();
            var inputDataMapStyle = new Dictionary<Variable, Value>() { { encoderLayersStyle.Arguments[0], inputStyle } };
            var outputDataMapStyle = new Dictionary<Variable, Value>() { { encoderLayersStyle.Outputs[pass - 1], null } };
            encoderLayersStyle.Outputs[pass - 1].ToFunction().Evaluate(inputDataMapStyle, outputDataMapStyle, false, device);

            stopwatch.Stop();
            //UnityEngine.Debug.Log("--Pass " + pass + " encoding style completed. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);

            stopwatch.Restart();
            //whitening
            var whited = Images.WhitenAndColor(outputDataMapStyle[encoderLayersStyle.Outputs[pass - 1]], 
                outputDataMapContent[encoderLayersContent.Outputs[pass - 1]], device,parameters.Eps, parameters.IgnoreSingularValue,
                parameters.Beta, parameters.BlendFactor);
            stopwatch.Stop();
            //UnityEngine.Debug.Log("--Pass " + pass + " WCT completed. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);

            stopwatch.Restart();
            Value varWhited = whited.Item1;

            var finalInputDatamap = new Dictionary<Variable, Value>() { { decoders[pass - 1].Arguments[0], varWhited } };
            var finalOutputDatamap = new Dictionary<Variable, Value>() { { decoders[pass - 1].Output, null } };

            decoders[pass - 1].Evaluate(finalInputDatamap, finalOutputDatamap, false, device);
            stopwatch.Stop();
            //UnityEngine.Debug.Log("--Pass " + pass + " decoding completed. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);
            Value result =  finalOutputDatamap[decoders[pass - 1].Output];
            //decoders[pass - 1]["output"].Dispose();
            //decoders[pass - 1].Dispose();
            return result;
        }

        public void CreateModelWithDimensions(Vector2Int contentDimension, Vector2Int styleDimension)
        {
            this.contentDimension = contentDimension;
            this.styleDimension = styleDimension;

            encoderLayersContent = CreateEncoders(contentDimension);
            encoderLayersStyle = CreateEncoders(styleDimension);

            decoders = new List<Function>();
            decoders.Add(CreateDecoders(contentDimension, PassIndex.PassOne));
            decoders.Add(CreateDecoders(contentDimension, PassIndex.PassTwo));
            decoders.Add(CreateDecoders(contentDimension, PassIndex.PassThree));
            decoders.Add(CreateDecoders(contentDimension, PassIndex.PassFour));
            decoders.Add(CreateDecoders(contentDimension, PassIndex.PassFive));

            Function f = Function.Load(modelWeights, device);
            decoders[0].RestoreParametersByName(f);
            decoders[0] = decoders[0].Clone(ParameterCloningMethod.Freeze);
            decoders[1].RestoreParametersByName(f);
            decoders[1] = decoders[1].Clone(ParameterCloningMethod.Freeze);
            decoders[2].RestoreParametersByName(f);
            decoders[2] = decoders[2].Clone(ParameterCloningMethod.Freeze);
            decoders[3].RestoreParametersByName(f);
            decoders[3] = decoders[3].Clone(ParameterCloningMethod.Freeze);
            decoders[4].RestoreParametersByName(f);
            decoders[4] = decoders[4].Clone(ParameterCloningMethod.Freeze);

            encoderLayersStyle.RestoreParametersByName(f);
            encoderLayersStyle = encoderLayersStyle.Clone(ParameterCloningMethod.Freeze);

            encoderLayersContent.RestoreParametersByName(f);
            encoderLayersContent = encoderLayersContent.Clone(ParameterCloningMethod.Freeze);
        }



        /// <summary>
        /// Create the vgg19 convolutional encoders for UST model
        /// </summary>
        /// <param name="imageDimension"></param>
        /// <returns></returns>
        protected  Function CreateEncoders(Vector2Int imageDimension)
        {

            Function encoderLayers;
            VariableVector outputs = new VariableVector();
            //input variables
            Variable prev = Variable.InputVariable(new int[] { imageDimension.x, imageDimension.y, 3}, DataType.Float, "input");
            //encoderLayers["input"] = prev;

            //vgg preprocessing
            prev = Layers.Convolution2D(prev, 3, 1, 1, device, 1, false, true, "conv0_preprocessing");
            //encoderLayers["conv0_preprocessing"] = prev;

            //----conv1----
            //conv1_1
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 64, 3, 3, device, 1, false, true, "conv1_1");
            prev = CNTKLib.ReLU(prev, "relu1_1");
            //encoderLayers["relu1_1"] = prev;
            outputs.Add(prev);

            //conv1_2
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 64, 3, 3, device, 1, false, true, "conv1_2");
            prev = CNTKLib.ReLU(prev, "relu1_2");
            //maxpooling 1
            prev = CNTKLib.Pooling(prev, PoolingType.Max, new int[] { 2, 2 }, new int[] { 2, 2 }, BoolVector.Repeat(true, 2), false, true, "pool1");

            //----conv2----
            //conv2_1
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 128, 3, 3, device, 1, false, true, "conv2_1");
            prev = CNTKLib.ReLU(prev, "relu2_1");
            outputs.Add(prev);
            // encoderLayers["relu2_1"] = prev;
            //conv2_2
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 128, 3, 3, device, 1, false, true, "conv2_2");
            prev = CNTKLib.ReLU(prev, "relu2_2");
            //maxpooling 2
            prev = CNTKLib.Pooling(prev, PoolingType.Max, new int[] { 2, 2 }, new int[] { 2, 2 }, BoolVector.Repeat(true, 2), false, true, "pool2");

            //----conv3----
            //conv3_1
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "conv3_1");
            prev = CNTKLib.ReLU(prev, "relu3_1");
            outputs.Add(prev);
            //encoderLayers["relu3_1"] = prev;
            //conv3_2
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "conv3_2");
            prev = CNTKLib.ReLU(prev, "relu3_2");
            //conv3_3
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "conv3_3");
            prev = CNTKLib.ReLU(prev, "relu3_3");
            //conv3_4
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "conv3_4");
            prev = CNTKLib.ReLU(prev, "relu3_4");
            //maxpooling 3
            prev = CNTKLib.Pooling(prev, PoolingType.Max, new int[] { 2, 2 }, new int[] { 2, 2 }, BoolVector.Repeat(true, 2), false, true, "pool3");

            //----conv4----
            //conv4_1
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "conv4_1");
            prev = CNTKLib.ReLU(prev, "relu4_1");
            outputs.Add(prev);
            //encoderLayers["relu4_1"] = prev;
            //conv4_2
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "conv4_2");
            prev = CNTKLib.ReLU(prev, "relu4_2");
            //conv4_3
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "conv4_3");
            prev = CNTKLib.ReLU(prev, "relu4_3");
            //conv4_4
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "conv4_4");
            prev = CNTKLib.ReLU(prev, "relu4_4");
            //maxpooling 4
            prev = CNTKLib.Pooling(prev, PoolingType.Max, new int[] { 2, 2 }, new int[] { 2, 2 }, BoolVector.Repeat(true, 2), false, true, "pool4");

            //----conv5----
            //conv5_1
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1,1,0}), new SizeTVector(new uint[] { 1,1,0}));
            prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "conv5_1");
            prev = CNTKLib.ReLU(prev, "relu5_1");
            outputs.Add(prev);

            encoderLayers = CNTKLib.Combine(outputs);
            //encoderLayers["relu5_1"] = prev;
            return encoderLayers;
        }

        

        /// <summary>
        /// Create a Universal style transfer decoder. There are 5 decoders in UST, use index to specify which one to create.
        /// </summary>
        /// <param name="imageDimension">image dimension for the decoder to work with. Note that it might be different from  the decoder input dimemensions</param>
        /// <param name="index"></param>
        /// <returns></returns>
        protected Function CreateDecoders(Vector2Int imageDimension, PassIndex index)
        {

            Function encoderLayers;
            int ind = (int)index + 1;   //1 based index for the names
            Vector2Int inputDims = imageDimension;
            
            int[] xOffsets = new int[5];
            int[] yOffsets = new int[5];

            for(int i = 1; i < ind; ++i)
            {
                var temp = new Vector2Int(Mathf.CeilToInt(inputDims.x / 2.0f), Mathf.CeilToInt(inputDims.y / 2.0f));
                if(Mathf.Abs(inputDims.x/2.0f - temp.x) > 0.00001f)
                {
                    xOffsets[i] = 1;
                }
                if (Mathf.Abs(inputDims.y / 2.0f - temp.y ) > 0.00001f )
                {
                    yOffsets[i] = 1;
                }
                inputDims = temp;
            }

            //set the channel number
            int inputChannels = 64;
            switch (index)
            {
                case PassIndex.PassOne:
                    inputChannels = 64;
                    break;
                case PassIndex.PassTwo:
                    inputChannels = 128;
                    break;
                case PassIndex.PassThree:
                    inputChannels = 256;
                    break;
                case PassIndex.PassFour:
                    inputChannels = 512;
                    break;
                case PassIndex.PassFive:
                    inputChannels = 512;
                    break;
                default:
                    inputChannels = 64;
                    break;

            }
            
            //input variables
            Variable prev = Variable.InputVariable(new int[] { inputDims.x, inputDims.y, inputChannels }, DataType.Float, "input");
            //encoderLayers["input"] = prev;

            if (ind >= 5)
            {
                //decoder 5
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "de"+ ind + "conv5_1");
                prev = CNTKLib.ReLU(prev);
                prev = Layers.Upsample2D2(prev, xOffsets[4], yOffsets[4]);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "de" + ind + "conv4_4");
                prev = CNTKLib.ReLU(prev);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "de" + ind + "conv4_3");
                prev = CNTKLib.ReLU(prev);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 512, 3, 3, device, 1, false, true, "de" + ind + "conv4_2");
                prev = CNTKLib.ReLU(prev);
            }

            if (ind >= 4)
            {
                //decoder 4
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "de" + ind + "conv4_1");
                prev = CNTKLib.ReLU(prev);
                prev = Layers.Upsample2D2(prev, xOffsets[3], yOffsets[3]);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "de" + ind + "conv3_4");
                prev = CNTKLib.ReLU(prev);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "de" + ind + "conv3_3");
                prev = CNTKLib.ReLU(prev);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 256, 3, 3, device, 1, false, true, "de" + ind + "conv3_2");
                prev = CNTKLib.ReLU(prev);
            }

            if (ind >= 3)
            {
                //decoder 3
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 128, 3, 3, device, 1, false, true, "de" + ind + "conv3_1");
                prev = CNTKLib.ReLU(prev);
                prev = Layers.Upsample2D2(prev, xOffsets[2], yOffsets[2]);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 128, 3, 3, device, 1, false, true, "de" + ind + "conv2_2");
                prev = CNTKLib.ReLU(prev);
            }

            if (ind >= 2)
            {
                //decoder 2
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 64, 3, 3, device, 1, false, true, "de" + ind + "conv2_1");
                prev = CNTKLib.ReLU(prev);
                prev = Layers.Upsample2D2(prev, xOffsets[1], yOffsets[1]);
                prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
                prev = Layers.Convolution2D(prev, 64, 3, 3, device, 1, false, true, "de" + ind + "conv1_2");
                prev = CNTKLib.ReLU(prev);
            }

            //decoder 1
            prev = CNTKLib.Pad(prev, PaddingMode.REFLECTPAD, new SizeTVector(new uint[] { 1, 1, 0 }), new SizeTVector(new uint[] { 1, 1, 0 }));
            prev = Layers.Convolution2D(prev, 3, 3, 3, device, 1, false, true, "de" + ind + "conv1_1");
            prev = CNTKLib.ReLU(prev, "output");

            //encoderLayers["output"] = prev;
            encoderLayers = prev;
            return encoderLayers;
        }





   }
}