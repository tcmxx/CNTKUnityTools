using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

namespace UnityCNTK
{
    public static class Layers
    {
        //default name used for the suffix for all the bias parameters
        public static readonly string BiasSuffix = ".b";
        //default name used for the suffix for all the weight parameters
        public static readonly string WeightSuffix = ".W";



        public static Function Upsample2D2(Variable input, int xOffset = 0, int yOffset = 0, string name = "")
        {

            var xr = CNTKLib.Reshape(input, new int[] { 1, input.Shape[0], 1, input.Shape[1],  input.Shape[2] });
            var xx = CNTKLib.Splice(VariableVector.Repeat(xr,2), new Axis(0));
            var xy = CNTKLib.Splice(VariableVector.Repeat(xx, 2), new Axis(2));
            var r = CNTKLib.Reshape(xy, new int[] { input.Shape[0] * 2, input.Shape[1] * 2, input.Shape[2] });

            var sliceAxis = new AxisVector();
            sliceAxis.Add(new Axis(0)); sliceAxis.Add(new Axis(1));
            r = CNTKLib.Slice(r, sliceAxis, new IntVector(new int[] { xOffset, yOffset }), new IntVector(new int[] { input.Shape[0] * 2, input.Shape[1] * 2 }));
            //name
            if (!string.IsNullOrEmpty(name))
            {
                r.SetName(name);
            }
            return r;
        }

        public static Function ConvolutionTranspose2D(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight,DeviceDescriptor device,
            int stride = 1, bool pad = false, bool bias = true, string name = "")
        {
            int numInputChannels = input.Shape[input.Shape.Rank - 1];
            //kernal
            var convParams = new Parameter(new int[] { kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount },
                DataType.Float, CNTKLib.GlorotUniformInitializer(1, -1, 2), device, name + WeightSuffix);

            //deconv
            var convFunction = CNTKLib.ConvolutionTranspose(convParams, input, new int[] { stride, stride, numInputChannels }, BoolVector.Repeat(true, 3), new BoolVector(new bool[] { pad, pad, false }));

            //bias
            if (bias)
            {
                var b = new Parameter(new int[] { 1, 1, outFeatureMapCount }, DataType.Float, CNTKLib.ConstantInitializer(0), device, name + BiasSuffix);
                convFunction = b + convFunction;
            }
            //name
            if (!string.IsNullOrEmpty(name))
            {
                convFunction.SetName(name);
            }
            return convFunction;

        }

        /// <summary>
        /// conv2d. the 3rd dimension of input is the channel
        /// </summary>
        /// <param name="input"></param>
        /// <param name="outFeatureMapCount"></param>
        /// <param name="kernelWidth"></param>
        /// <param name="kernelHeight"></param>
        /// <param name="device"></param>
        /// <param name="stride"></param>
        /// <param name="pad"></param>
        /// <param name="bias"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Function Convolution2D(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, DeviceDescriptor device, 
            int stride=1, bool pad= false, bool bias = true, string name = "")
        {
            int numInputChannels = input.Shape[input.Shape.Rank - 1];


            //kernal
            var convParams = new Parameter(new int[] {  kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount },
                DataType.Float, CNTKLib.GlorotUniformInitializer(1, -1, 2), device, name + WeightSuffix);
            //conv
            var convFunction = CNTKLib.Convolution(convParams, input, new int[] { stride, stride, outFeatureMapCount}, new bool[] { true,true, true}, new bool[] { pad, pad, false});

            //bias
            if (bias)
            {
                var b = new Parameter(new int[] {1, 1, outFeatureMapCount}, DataType.Float, CNTKLib.ConstantInitializer(0), device, name + BiasSuffix);
                convFunction = b + convFunction;
            }
            if (!string.IsNullOrEmpty(name))
            {
                convFunction.SetName(name);
            }
            return convFunction;
        }


        public static Function Dense(Variable input, int outDim, DeviceDescriptor device, bool bias = true,  string name = "", float weightInitScale = 1)
        {
            List<int> pDimension = new List<int>(input.Shape.Dimensions);
            pDimension.Insert(0, outDim);

            var weightParam = new Parameter(pDimension.ToArray(), DataType.Float, CNTKLib.GlorotUniformInitializer(weightInitScale, 1, 0), device, name + WeightSuffix);

            //var normalizedWeight = Normalize(weightParam, new Axis(1));

            var f = CNTKLib.Times(weightParam, input);
            //var f = CNTKLib.Times(normalizedWeight, input);
            if (bias)
            {
                var biasParam = new Parameter(new int[] { outDim }, DataType.Float, 0, device, name + BiasSuffix);
                f = f + biasParam;
            }
            return f;
        }

        public static Function LayerNormalization(Variable input, DeviceDescriptor device, float initB = 0, float initScale = 1, string name = "", float eps = 0.00001f)
        {
            //get the mean first
            //var mean = CNTKLib.ReduceMean(input, Axis.AllStaticAxes());
            //var centered =  CNTKLib.Minus(input, mean);
            //var squared = CNTKLib.Square(centered);
            //var variance = CNTKLib.ReduceMean(squared, Axis.AllStaticAxes());

            var biasParams = new Parameter(new int[] { 1 }, initB, device, name + BiasSuffix);
            var scaleParams = new Parameter(new int[] { 1 }, initScale, device, name + WeightSuffix);
           // var epsConst = Constant.Scalar(DataType.Float, 0.00001f);

            //var std = CNTKLib.Sqrt(variance + epsConst);
            var normalized = Normalize(input,Axis.AllStaticAxes(),eps);
            var result = normalized * scaleParams + biasParams;
            return result;
        }

        public static Function Normalize(Variable input, Axis axis, float desiredMean = 0, float desriedStd = 1,float eps = 0.00001f)
        {
            var mean = CNTKLib.ReduceMean(input, axis);
            var centered = CNTKLib.Minus(input, mean);
            var squared = CNTKLib.Square(centered);
            var variance = CNTKLib.ReduceMean(squared, axis);
            var epsConst = Constant.Scalar(DataType.Float, 0.00001f);
            var std = CNTKLib.Sqrt(variance + epsConst);
            var normalized = CNTKLib.ElementDivide(centered, std);
            var result = Constant.Scalar(DataType.Float, desriedStd) *normalized + Constant.Scalar(DataType.Float,desiredMean);
            return result;
        }

        public static Function BatchNormalization(Variable input, double bnInitBias, double bnInitScale, int bnTimeConst, bool spatial, DeviceDescriptor device, string name = "", float eps = 0.00001f)
        {

            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)bnInitBias, device, name + BiasSuffix);
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)bnInitScale, device, name + WeightSuffix);
            var runningMean = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, device);
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, device);
            var runningCount = Constant.Scalar(0.0f, device);
            return CNTKLib.BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount,
                spatial, (double)bnTimeConst, 0.0, eps);
        }

        //add zeros to the input so that input has a larger width
        public static Function AddDummy(Variable input, int numToAdd, DeviceDescriptor device)
        {
            int inputSize = input.Shape[0];

            //create the dummy data
            float[] addValues = new float[numToAdd];
            for (int i = 0; i < numToAdd; i++)
                addValues[i] = 0;

            var addArray = new NDArrayView(DataType.Float, new int[] { numToAdd }, device);
            addArray.CopyFrom(new NDArrayView(new int[] { numToAdd }, addValues, (uint)addValues.Length, device));
            var dummyData = new Constant(addArray);

            //append the dummy data to the input
            var vars = new VariableVector();
            vars.Add(input);
            vars.Add(dummyData);
            var added = CNTKLib.Splice(vars, new Axis(0));
            return added;
        }


        public static Function NormalProbability(Variable input, Variable mean, Variable variance, DeviceDescriptor device)
        {
            //probability
            var diff = CNTKLib.Minus(input, mean);
            var temp1 = CNTKLib.ElementTimes(diff, diff);
            temp1 = CNTKLib.ElementDivide(temp1, CNTKLib.ElementTimes(Constant.Scalar<float>(2, device), variance));
            temp1 = CNTKLib.Exp(CNTKLib.Minus(Constant.Scalar<float>(0, device), temp1));

            var temp2 = CNTKLib.ElementDivide(
                Constant.Scalar<float>(1, device), 
                CNTKLib.Sqrt(
                    CNTKLib.ElementTimes(
                        variance, Constant.Scalar<float>(2 * Mathf.PI, device))));
            return CNTKLib.ElementTimes(temp1,temp2);
        }


        public static Function HuberLoss(Variable input1, Variable input2, DeviceDescriptor device)
        {
            var error = CNTKLib.Minus(input1, input2);
            var square = CNTKLib.ElementDivide(CNTKLib.Square(error), Constant.Scalar(2.0f, device));
            var linear = CNTKLib.Minus(CNTKLib.Abs(error),Constant.Scalar(0.5f, device));
            var useLinear = CNTKLib.Cast(CNTKLib.GreaterEqual(linear, Constant.Scalar(0.5f, device)),DataType.Float);
            return CNTKLib.ElementTimes(linear, useLinear).Output + CNTKLib.ElementTimes(square, CNTKLib.Minus(Constant.Scalar(1.0f,device), useLinear)).Output;

        }

    }


}
