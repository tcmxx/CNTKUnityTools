using System.Collections;
using System.Collections.Generic;
using UnityEngine;


using CNTK;
using UnityCNTK.LayerDefinitions;
using MathNet.Numerics.Distributions;

namespace UnityCNTK.ReinforcementLearning
{



    public class PPOModel
    {

        public PPONetwork Network { get; protected set; }
        public bool IsActionContinuous { get { return Network.IsActionContinuous; } }
        public int StateSize { get { return Network.StateSize; } }
        public int ActionSize { get { return Network.ActionSize; } }
        public DeviceDescriptor Device { get { return Network.Device; } }


        //---variables for training
        public Variable InputAction { get; protected set; }
        public Variable InputState { get { return Network.InputState; } }
        public Variable InputOldProb { get; protected set; }
        public Variable InputTargetValue { get; protected set; }
        public Variable InputAdvantage { get; protected set; }
        public Variable InputClipEpsilon { get; protected set; }

        public Variable InputValuelossWeight { get; protected set; }
        public Variable InputEntropyLossWeight { get; protected set; }
        public Variable OutputEntropy { get; protected set; }
        public Variable OutputPolicyLoss { get; protected set; }
        public Variable OutputValueLoss { get; protected set; }
        public Variable OutputLoss { get; protected set; }

        //public Variable testOutputProb;

        public PPOModel(PPONetwork network)
        {
            Network = network;

            //inputs
            if (IsActionContinuous) {
                InputAction = CNTKLib.InputVariable(new int[] { Network.ActionSize }, DataType.Float);
                InputOldProb = CNTKLib.InputVariable(new int[] { Network.ActionSize }, DataType.Float);
            }
            else
            {
                InputAction = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
                InputOldProb = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
            }
            InputAdvantage = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
            InputTargetValue = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
            InputClipEpsilon = Constant.Scalar<float>(0.1f, Device);
            InputValuelossWeight = Constant.Scalar<float>(1f, Device);
            InputEntropyLossWeight = Constant.Scalar<float>(0f, Device);

            Variable actionProb = null;
            if (IsActionContinuous)
            {   
                //create the entropy loss part
                var temp = CNTKLib.ElementTimes(Constant.Scalar(DataType.Float, 2 * Mathf.PI * 2.7182818285), Network.OutputVariance);
                temp = CNTKLib.ElementTimes(Constant.Scalar(DataType.Float, 0.5), temp);
                OutputEntropy = CNTKLib.ReduceSum(temp, Axis.AllStaticAxes());
                //probability
                actionProb = Layers.NormalProbability(InputAction, Network.OutputMean, Network.OutputVariance, Device);
            }
            else
            {
                OutputEntropy = CNTKLib.Minus(Constant.Scalar<float>(0,Device), CNTKLib.ReduceSum(
                    CNTKLib.ElementTimes(
                        Network.OutputProbabilities, CNTKLib.Log(
                            Network.OutputProbabilities + Constant.Scalar<float>(0.000000001f, Device))), Axis.AllStaticAxes()));
                var oneHot = CNTKLib.OneHotOp(InputAction, (uint)Network.ActionSize, false, new Axis(0));
                actionProb = CNTKLib.ReduceSum(CNTKLib.ElementTimes(Network.OutputProbabilities, oneHot),Axis.AllStaticAxes());
            }
            //testOutputProb = actionProb;

            //value loss. Simple square loss

            OutputValueLoss = CNTKLib.SquaredError(Network.OutputValue, InputTargetValue);

            //policyloss
            //1. Clipped Surrogate loss
            var probRatio = CNTKLib.ElementDivide(actionProb, InputOldProb + Constant.Scalar<float>(0.0000000001f, Device));
            var p_opt_a = CNTKLib.ElementTimes(probRatio,InputAdvantage);
            var p_opt_b = CNTKLib.ElementTimes(
                CNTKLib.Clip(probRatio, CNTKLib.Minus(
                    Constant.Scalar<float>(1, Device), InputClipEpsilon), 
                    Constant.Scalar<float>(1, Device) + InputClipEpsilon),InputAdvantage);

            OutputPolicyLoss = CNTKLib.Minus(Constant.Scalar<float>(1, Device),CNTKLib.ReduceMean(CNTKLib.ElementMin(p_opt_a, p_opt_b, "min"), Axis.AllStaticAxes()));
            //OutputPolicyLoss = CNTKLib.ReduceMean(CNTKLib.ElementMin(p_opt_a, p_opt_b, "min"), Axis.AllStaticAxes());
            //OutputPolicyLoss = CNTKLib.Minus(Constant.Scalar<float>(1, Device), CNTKLib.ReduceMean(p_opt_a, Axis.AllStaticAxes()));

            //final weighted loss
            OutputLoss = OutputPolicyLoss + CNTKLib.ElementTimes(InputValuelossWeight,OutputValueLoss);
            OutputLoss = CNTKLib.Minus(OutputLoss,CNTKLib.ElementTimes(InputEntropyLossWeight,OutputEntropy));
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


        public int[] EvaluateActionDiscrete(float[] state, out float[] actionProbs, bool useProbability = true)
        {
            Debug.Assert(!IsActionContinuous, "Action is not discrete");
            //input data maps
            var inputDataMap = new Dictionary<Variable, Value>();

            Value inputStatedata = Value.CreateBatch(Network.InputState.Shape, state, Network.Device, true);
            inputDataMap.Add(Network.InputState, inputStatedata);

            //output datamaps
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(Network.OutputProbabilities, null);
            
            Network.PolicyFunction.Evaluate(inputDataMap, outputDataMap, Device);
            var result = outputDataMap[Network.OutputProbabilities].GetDenseData<float>(Network.OutputProbabilities);
            
            int batchSize = result.Count;;

            int[] actions = new int[batchSize];
            actionProbs = new float[batchSize];
            for (int i = 0; i < batchSize; ++i)
            {
                if (useProbability)
                {
                    actions[i] = MathUtils.IndexByChance(result[i]);
                }
                else
                {
                    actions[i] = MathUtils.IndexMax(result[i]);
                }
                actionProbs[i] = result[i][actions[i]];
            }
            return actions;
        }

        public float[] EvaluateActionContinuous(float[] state, out float[] actionProbs)
        {
            Debug.Assert(IsActionContinuous, "Action is not continuous");
            //input data maps
            var inputDataMap = new Dictionary<Variable, Value>();

            Value inputStatedata = Value.CreateBatch(Network.InputState.Shape, state, Network.Device, true);
            inputDataMap.Add(Network.InputState, inputStatedata);
            //test
            //inputDataMap.Add(InputAction, Value.CreateBatch(testOutputProb.Shape, new float[] { 1, 1.4f }, Device));

            //output datamaps
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(Network.OutputMean, null);
            outputDataMap.Add(Network.OutputVariance, null);
            //test
            //outputDataMap.Add(testOutputProb, null);
            //test
           // testOutputProb.ToFunction().Evaluate(inputDataMap, outputDataMap, Device);
            Network.PolicyFunction.Evaluate(inputDataMap, outputDataMap, Device);

            var means = outputDataMap[Network.OutputMean].GetDenseData<float>(Network.OutputMean);
            var vars = outputDataMap[Network.OutputVariance].GetDenseData<float>(Network.OutputVariance);

            
            int batchSize = means.Count;
            int actionSize = means[0].Count;
            int count = actionSize * batchSize;
            float[] actions = new float[count];
            actionProbs = new float[count];
            for (int i = 0; i < batchSize; ++i)
            {
                for (int j = 0;j < actionSize; ++j)
                {
                    float std = Mathf.Sqrt(vars[i][j]);
                    actions[i* batchSize + j] = (float)Normal.Sample(means[i][j], std);
                    actionProbs[i * batchSize + j] = (float)Normal.PDF(means[i][j], std, actions[i * batchSize + j]);
                }
            }
            
            //test
            //var test = outputDataMap[testOutputProb].GetDenseData<float>(testOutputProb)[0];
            //Debug.Log("test:" + string.Join(",", test));

            return actions;
        }

        public float[] EvaluateValue(float[] state)
        {
            //input data maps
            var inputDataMap = new Dictionary<Variable, Value>();

            Value inputStatedata = Value.CreateBatch(Network.InputState.Shape, state, Network.Device, true);
            inputDataMap.Add(Network.InputState, inputStatedata);

            //output datamaps
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(Network.OutputValue, null);

            Network.ValueFunction.Evaluate(inputDataMap, outputDataMap, Device);
            var values = outputDataMap[Network.OutputValue].GetDenseData<float>(Network.OutputValue);

            float[] result = new float[values.Count];
            for(int i = 0;i < result.Length; ++i)
            {
                result[i] = values[i][0];
            }
            return result;
        }
    }






}