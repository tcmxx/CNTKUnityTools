using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityCNTK.LayerDefinitions;
using CNTK;
namespace UnityCNTK
{
    public class GAN
    {

        public SequentialNetworkDense GeneratorSequentialModel { get; private set; }
        public SequentialNetworkDense DiscriminatorSequentialModel { get; private set; }

        public Variable GeneratorOutput { get; private set; }
        public Variable DiscriminatorFakeOutput { get; private set; }
        public Variable DiscriminatorRealOutput { get; private set; }
        public Function DiscriminatorMerged { get; private set; }

        public Variable InputNoiseGenerator { get; private set; }
        public Variable InputConditionGenerator { get; private set; }
        public Variable InputTargetGenerator { get; private set; }

        public Variable InputDataDiscriminatorReal { get; private set; }
        public Variable InputConditionDiscriminatorReal { get; private set; }
        public Variable InputConditionDiscriminatorFake { get; private set; }

        public Variable GeneratorLossL2 { get; private set; }
        public Variable GeneratorLossGAN { get; private set; }
        public Variable GeneratorLoss { get; private set; }
        public Variable DiscriminatorLoss { get; private set; }

        public readonly float EPS = 0.00001f;
        protected Variable epsConst;

        public int InputNoiseSize { get; private set; }
        public int OutputSize { get; private set; }
        public int InputConditionSize { get; private set; }

        public DeviceDescriptor Device { get; private set; }

        public GAN(int inputNoiseSize, int inputConditionSize, int outputSize, int generatorLayerSize, int generatorLayerCount, int discriminatorLayerSize, int discriminatorLayerCount, DeviceDescriptor device, float generatorL2lossFactor = 0)
        {
            Debug.Assert(inputNoiseSize > 0 || inputConditionSize > 0, "At least one of input noise or input condition should have non-zero size");
            Debug.Assert(outputSize > 0, "Output size should be larger than 0");

            Device = device;
            InputNoiseSize = inputNoiseSize;
            OutputSize = outputSize;
            InputConditionSize = inputConditionSize;


            CreateGenerator(inputNoiseSize, inputConditionSize, outputSize, generatorLayerSize, generatorLayerCount, device);
            CreateDiscriminators(GeneratorOutput, inputConditionSize, outputSize, discriminatorLayerSize, discriminatorLayerCount, device);

            //get losses
            epsConst = Constant.Scalar(DataType.Float, EPS);
            //loss for generator
            var l2Loss = GeneratorSequentialModel.OutputLayer.GetTrainingLossVariable();
            GeneratorLossL2 = CNTKLib.ElementTimes(l2Loss,Constant.Scalar(DataType.Float, generatorL2lossFactor));
            GeneratorLossGAN = CNTKLib.Minus(Constant.Scalar(DataType.Float, 1),  CNTKLib.Log(DiscriminatorFakeOutput + epsConst));
            GeneratorLoss = GeneratorLossL2 + GeneratorLossGAN;
            //loss for discriminator
            DiscriminatorLoss = CNTKLib.Minus(
                Constant.Scalar(DataType.Float, 0),
                CNTKLib.Log(DiscriminatorRealOutput + epsConst).Output +
                CNTKLib.Log(CNTKLib.Minus(Constant.Scalar(DataType.Float, 1),DiscriminatorFakeOutput) + epsConst).Output);
            
        }



        public IList<float> EvaluateOne(float[] condition)
        {
            return EvaluateOne(Utils.GenerateWhiteNoise(InputNoiseSize, -1, 1), condition);
        }
        public IList<float> EvaluateOne(float[] noise, float[] condition)
        {
            //input data maps
            var inputDataMap = new Dictionary<Variable, Value>();

            if(InputConditionSize >0)
            {
                Value inputConditiondata = Value.CreateBatch(InputConditionGenerator.Shape, condition, Device, true);
                inputDataMap.Add(InputConditionGenerator, inputConditiondata);
            }
            if(InputNoiseSize > 0)
            {
                Value inputNoisedata = Value.CreateBatch(InputNoiseGenerator.Shape, noise, Device, true);
                inputDataMap.Add(InputNoiseGenerator, inputNoisedata);
            }
            
            //output datamaps
            var outputDataMap = new Dictionary<Variable, Value>() { { GeneratorOutput, null } };


            GeneratorOutput.ToFunction().Evaluate(inputDataMap, outputDataMap, Device);
            var result = outputDataMap[GeneratorOutput].GetDenseData<float>(GeneratorOutput);
            return result[0];
        }
        /// <summary>
        /// Helper functio to create discriminators
        /// </summary>
        /// <param name="fakeDataFromGenerator"></param>
        /// <param name="inputConditionSize"></param>
        /// <param name="outputSize"></param>
        /// <param name=""></param>
        /// <param name="discriminatorLayerSize"></param>
        /// <param name="discriminatorLayerCount"></param>
        /// <param name="device"></param>
        protected void CreateDiscriminators(Variable fakeDataFromGenerator, int inputConditionSize, int outputSize, int discriminatorLayerSize, int discriminatorLayerCount, DeviceDescriptor device)
        {
            //create discriminator
            Variable concatenatedInput = null;
            //create input based on whether it is a conditional gan
            if (inputConditionSize > 0)
            {
                InputDataDiscriminatorReal = CNTKLib.InputVariable(new int[] { outputSize }, DataType.Float);
                InputConditionDiscriminatorReal = CNTKLib.InputVariable(new int[] { inputConditionSize }, DataType.Float);
                InputConditionDiscriminatorFake = CNTKLib.InputVariable(new int[] { inputConditionSize }, DataType.Float);
                var vsDiscriminator = new VariableVector();
                vsDiscriminator.Add(InputDataDiscriminatorReal);
                vsDiscriminator.Add(InputConditionDiscriminatorReal);
                concatenatedInput = CNTKLib.Splice(vsDiscriminator, new Axis(0));
            }
            else
            {
                InputDataDiscriminatorReal = CNTKLib.InputVariable(new int[] { outputSize }, DataType.Float);
                InputConditionDiscriminatorReal = null;
                InputConditionDiscriminatorFake = null;
                concatenatedInput = InputDataDiscriminatorReal;
            }

            var inputD = new InputLayerCNTKVar(concatenatedInput);
            var outputLayerD = new OutputLayerDense(1, ActivationFunction.Sigmoid, OutputLayerDense.LossFunction.Square);
            //create the discriminator sequential model
            DiscriminatorSequentialModel = new SequentialNetworkDense(inputD, LayerDefineHelper.DenseLayers(discriminatorLayerCount, discriminatorLayerSize, true,NormalizationMethod.None), outputLayerD, device);
            //real discriminator output
            DiscriminatorRealOutput = DiscriminatorSequentialModel.OutputLayer.GetOutputVariable();

            //clone the discriminator with shared parameters
            if (inputConditionSize > 0)
            {
                DiscriminatorFakeOutput = ((Function)DiscriminatorRealOutput).Clone(ParameterCloningMethod.Share,
                    new Dictionary<Variable, Variable>() { { InputDataDiscriminatorReal, fakeDataFromGenerator }, { InputConditionDiscriminatorReal, InputConditionDiscriminatorFake } });
            }
            else
            {
                DiscriminatorFakeOutput = ((Function)DiscriminatorRealOutput).Clone(ParameterCloningMethod.Share,
                    new Dictionary<Variable, Variable>() { { InputDataDiscriminatorReal, fakeDataFromGenerator } });
            }
            DiscriminatorMerged = Function.Combine(new List<Variable>() { DiscriminatorRealOutput, DiscriminatorFakeOutput });
        }
        /// <summary>
        /// Helper functoin to create gerantor;
        /// </summary>
        /// <param name="inputNoiseSize"></param>
        /// <param name="inputConditionSize"></param>
        /// <param name="outputSize"></param>
        /// <param name="generatorLayerSize"></param>
        /// <param name="generatorLayerCount"></param>
        /// <param name="device"></param>
        protected void CreateGenerator(int inputNoiseSize, int inputConditionSize, int outputSize, int generatorLayerSize, int generatorLayerCount, DeviceDescriptor device)
        {
            //create generator
            Variable concatenatedInput;
            if (inputNoiseSize > 0 && inputConditionSize > 0)
            {
                InputNoiseGenerator = CNTKLib.InputVariable(new int[] { inputNoiseSize }, DataType.Float);
                InputConditionGenerator = CNTKLib.InputVariable(new int[] { inputConditionSize }, DataType.Float);
                var vsgenerator = new VariableVector();
                vsgenerator.Add(InputNoiseGenerator);
                vsgenerator.Add(InputConditionGenerator);
                concatenatedInput = CNTKLib.Splice(vsgenerator, new Axis(0));
            }
            else if (inputNoiseSize > 0)
            {
                InputNoiseGenerator = CNTKLib.InputVariable(new int[] { inputNoiseSize }, DataType.Float);
                InputConditionGenerator = null;
                concatenatedInput = InputNoiseGenerator;
            }
            else
            {
                InputNoiseGenerator = null;
                InputConditionGenerator = CNTKLib.InputVariable(new int[] { inputConditionSize }, DataType.Float);
                concatenatedInput = InputConditionGenerator;
            }

            var inputG = new InputLayerCNTKVar(concatenatedInput);
            var outputLayerG = new OutputLayerDense(outputSize, ActivationFunction.None, OutputLayerDense.LossFunction.Square);
            GeneratorSequentialModel = new SequentialNetworkDense(inputG, LayerDefineHelper.DenseLayers(generatorLayerCount, generatorLayerSize, true, NormalizationMethod.None), outputLayerG, device);
            GeneratorOutput = GeneratorSequentialModel.OutputLayer.GetOutputVariable();
            InputTargetGenerator = GeneratorSequentialModel.OutputLayer.GetTargetInputVariable();
            

        }

    }
}