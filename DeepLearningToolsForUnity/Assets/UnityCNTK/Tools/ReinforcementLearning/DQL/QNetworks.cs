using CNTK;
using System.Collections;
using System.Collections.Generic;
using UnityCNTK.LayerDefinitions;
using UnityEngine;

namespace UnityCNTK.ReinforcementLearning
{
    public abstract class QNetwork
    {
        public abstract int StateSize { get; protected set; }
        public abstract int ActionSize { get; protected set; }

        public abstract Variable InputState { get; protected set; }

        public abstract Variable OutputQs { get; protected set; }

        public abstract DeviceDescriptor Device { get; protected set; }

    }


    public class QNetworkSimple : QNetwork
    {
        public override int StateSize { get; protected set; }
        public override int ActionSize { get; protected set; }

        public override Variable InputState { get; protected set; }

        //actor outputs
        public override Variable OutputQs { get; protected set; }

        public override DeviceDescriptor Device { get; protected set; }
        public QNetworkSimple(int stateSize, int actionSize, int numLayers, int hiddenSize, DeviceDescriptor device, float initialWeightScale = 0.01f)
        {
            Device = device;
            StateSize = stateSize;
            ActionSize = actionSize;

            //create actor network part
            var inputA = new InputLayerDense(stateSize);
            var outputA = new OutputLayerDense(hiddenSize, null, OutputLayerDense.LossFunction.None);
            outputA.HasBias = false;
            outputA.InitialWeightScale = initialWeightScale;
            SequentialNetworkDense qNetwork = new SequentialNetworkDense(inputA, LayerDefineHelper.DenseLayers(numLayers, hiddenSize, false, NormalizationMethod.None, 0, initialWeightScale, new ReluDef()), outputA, device);

            //seperate the advantage and value part. It is said to be better
            var midStream = outputA.GetOutputVariable();
            var advantageStream = CNTKLib.Slice(midStream, AxisVector.Repeat(new Axis(0), 1), IntVector.Repeat(0, 1), IntVector.Repeat(hiddenSize / 2, 1));
            var valueStream = CNTKLib.Slice(midStream, AxisVector.Repeat(new Axis(0), 1), IntVector.Repeat(hiddenSize / 2, 1), IntVector.Repeat(hiddenSize, 1));
            var adv = Layers.Dense(advantageStream, actionSize, device, false, "QNetworkAdvantage", initialWeightScale);
            var value = Layers.Dense(valueStream, 1, device, false, "QNetworkValue", initialWeightScale);

            InputState = inputA.InputVariable;
            //OutputQs = outputA.GetOutputVariable();
            OutputQs = value.Output + CNTKLib.Minus(adv, CNTKLib.ReduceMean(adv, Axis.AllStaticAxes())).Output;

        }

    }


    public class QNetworkConvSimple : QNetwork
    {
        public override int StateSize { get; protected set; }
        public override int ActionSize { get; protected set; }
        public int[] InputDimension { get; private set; }
        public override Variable InputState { get; protected set; }

        //actor outputs
        public override Variable OutputQs { get; protected set; }

        public override DeviceDescriptor Device { get; protected set; }



        public QNetworkConvSimple(int inputWidth, int inputHeight, int inputDepth, int actionSize,  
            int[] filterSizes, int[] filterDepths,int[] strides, bool[] pooling,
            int densehiddenLayers, int densehiddenSize, bool denseUseBias, DeviceDescriptor device, float denseInitialWeightScale = 0.01f)
        {
            Device = device;
            StateSize = inputWidth*inputHeight*inputDepth;
            ActionSize = actionSize;
            InputDimension = new int[3] { inputWidth, inputHeight, inputDepth };


            //create actor network part
            InputState = CNTKLib.InputVariable(InputDimension, DataType.Float);

            Debug.Assert(filterSizes.Length == strides.Length && filterDepths.Length == filterSizes.Length, "Length of filterSizes,strides and filterDepth are not the same");

            var lastLayer = InputState;
            for(int i = 0; i < filterSizes.Length; ++i)
            {
                //conv layers. Use selu activaion and selu initlaization
                lastLayer = Layers.Convolution2D(lastLayer, filterDepths[i], 
                    filterSizes[i], filterSizes[i], device, strides[i], true, true, "QConv_"+i, Mathf.Sqrt((1.0f/ (filterSizes[i]* filterSizes[i]))));
                lastLayer = new SELUDef().BuildNew(lastLayer, device,"");
                //pooling
                if (pooling[i])
                {
                    lastLayer = CNTKLib.Pooling(lastLayer, PoolingType.Max, new int[] { 2, 2 }, new int[] { 2, 2 }, BoolVector.Repeat(true, 2), false, true, "pool2");
                }
            }

            lastLayer = CNTKLib.Flatten(lastLayer, new Axis(3),"Flatten");

            //dense layers
            var inputA = new InputLayerCNTKVar(lastLayer);
            var outputA = new OutputLayerDense(actionSize, null, OutputLayerDense.LossFunction.None);
            outputA.HasBias = false;
            outputA.InitialWeightScale = denseInitialWeightScale;
            SequentialNetworkDense qNetwork = new SequentialNetworkDense(inputA, LayerDefineHelper.DenseLayers(densehiddenLayers, densehiddenSize, denseUseBias, NormalizationMethod.None, 0, denseInitialWeightScale, new ReluDef()), outputA, device);
            
            //OutputQs = outputA.GetOutputVariable();
            OutputQs = outputA.GetOutputVariable();

        }

    }
}