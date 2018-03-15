using System.Collections;
using System.Collections.Generic;
using UnityEngine;


using CNTK;
using UnityCNTK.LayerDefinitions;
using MathNet.Numerics.Distributions;

namespace UnityCNTK
{
    public abstract class QNetwork
    {
        public abstract int StateSize { get; protected set; }
        public abstract int ActionSize { get; protected set; }
        
        public abstract Variable InputState { get; protected set; }

        public abstract Variable OutputQs { get; protected set; }
        
        public abstract DeviceDescriptor Device { get; protected set; }

    }


    /// <summary>
    /// PPO network similiar to one of Unity ML's python implementation
    /// https://github.com/Unity-Technologies/ml-agents
    /// </summary>
    public class QNetworkSimple : QNetwork
    {
        public override int StateSize { get; protected set; }
        public override int ActionSize { get; protected set; }
        
        public override Variable InputState { get; protected set; }

        //actor outputs
        public override Variable OutputQs { get; protected set; }

        protected SequentialNetworkDense qNetwork;

        public override DeviceDescriptor Device { get; protected set; }
        public QNetworkSimple(int stateSize, int actionSize, int numLayers, int hiddenSize, DeviceDescriptor device, float initialWeightScale = 0.01f)
        {
            Device = device;
            StateSize = stateSize;
            ActionSize = actionSize;

            //create actor network part
            var inputA = new InputLayerDense(stateSize);
            var outputA = new OutputLayerDense(actionSize, ActivationFunction.None, OutputLayerDense.LossFunction.None);
            outputA.InitialWeightScale = initialWeightScale;
            qNetwork = new SequentialNetworkDense(inputA, LayerDefineHelper.DenseLayers(numLayers, hiddenSize, NormalizationMethod.None, 0, initialWeightScale, ActivationFunction.Relu), outputA, device);
            InputState = inputA.InputVariable;
            OutputQs = outputA.GetOutputVariable();
        }

    }


    public class DQLModel
    {

        public QNetwork Network { get; protected set; }
        public int StateSize { get { return Network.StateSize; } }
        public int ActionSize { get { return Network.ActionSize; } }
        public DeviceDescriptor Device { get { return Network.Device; } }


        //---variables for training
        public Variable InputOldAction { get; protected set; }
        public Variable InputState { get { return Network.InputState; } }
        public Variable InputTargetQ { get; protected set; }
        
        public Variable OutputLoss { get; protected set; }
        public Variable OutputAction { get; protected set; }
        public Variable OutputQs { get { return Network.OutputQs; } }
        public Variable OutputMaxQ { get; protected set; }

        public Function CNTKFunction { get; protected set; }

        //public Variable testOutputProb;

        public DQLModel(QNetwork network)
        {
            Network = network;

            InputOldAction = CNTKLib.InputVariable(new int[] { Network.ActionSize }, DataType.Float);

            InputTargetQ = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);

            var oneHotOldAction = CNTKLib.OneHotOp(InputOldAction, (uint)ActionSize, false, new Axis(0));
            OutputLoss = CNTKLib.SquaredError(CNTKLib.ReduceSum(CNTKLib.ElementTimes(OutputQs, oneHotOldAction), Axis.AllStaticAxes()),InputTargetQ);

            OutputAction = CNTKLib.Argmax(OutputQs, new Axis(0));
            OutputMaxQ = CNTKLib.ReduceMax(OutputQs, new Axis(0));

            CNTKFunction = Function.Combine(new List<Variable>() { OutputLoss, OutputAction, OutputMaxQ });
        }

        public byte[] Save()
        {
            return OutputLoss.ToFunction().Save();
        }

        public void Restore(byte[] data)
        {
            Function f = Function.Load(data, Device);
            OutputLoss.ToFunction().RestoreParametersByName(f);
        }


        public int[] EvaluateAction(float[] state, out float[] maxQs)
        {
            //input data maps
            var inputDataMap = new Dictionary<Variable, Value>();

            Value inputStatedata = Value.CreateBatch(Network.InputState.Shape, state, Network.Device, true);
            inputDataMap.Add(Network.InputState, inputStatedata);

            //output datamaps
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(OutputMaxQ, null);
            outputDataMap.Add(OutputAction, null);

            CNTKFunction.Evaluate(inputDataMap, outputDataMap, Device);

            var maxQ = outputDataMap[OutputMaxQ].GetDenseData<float>(OutputMaxQ);
            var action = outputDataMap[OutputAction].GetDenseData<float>(OutputAction);

            int batchSize = maxQ.Count;

            int[] actions = new int[batchSize];
            maxQs = new float[batchSize];
            for (int i = 0; i < batchSize; ++i)
            {
                actions[i] = (int)action[i][0];
                maxQs[i] = maxQ[i][0];
            }
            return actions;
        }
    }






}