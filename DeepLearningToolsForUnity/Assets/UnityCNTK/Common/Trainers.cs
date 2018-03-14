using System.Collections;
using System.Collections.Generic;
using CNTK;
using System;
using System.Linq;

namespace UnityCNTK
{
    public class TrainerSimpleNN
    {
        protected Trainer trainer;
        protected List<Learner> learners;
        protected SequentialNetworkDense networkRef;

        protected DataBuffer dataBuffer;

        public DeviceDescriptor Device { get; set; }

        public float LastLoss { get { return (float)trainer.PreviousMinibatchLossAverage(); } }
        public int LastBatchSize { get { return (int)trainer.PreviousMinibatchSampleCount(); } }

        public TrainerSimpleNN(SequentialNetworkDense net, LearnerDefs.LearnerDef trainingLearner, DeviceDescriptor device, int maxDataBufferCount = 50000)
        {

            Device = device;
            networkRef = net;
            learners = new List<Learner>();

            var paramsToTrain = net.CNTKFunction.Parameters();
            Learner learner = trainingLearner.Create(paramsToTrain);
            learners.Add(learner);

            trainer = Trainer.CreateTrainer(networkRef.OutputLayer.GetOutputVariable(), networkRef.OutputLayer.GetTrainingLossVariable(),
                networkRef.OutputLayer.GetTrainingLossVariable(),//use training loss for eval error for now
                learners);

            //get the input shape for creating the buffer
            var inputDims = networkRef.InputLayer.InputVariable.Shape.Dimensions;
            int inputSize = 1;
            foreach (var d in inputDims)
            {
                inputSize *= d;
            }
            var targetDims = networkRef.OutputLayer.GetTargetInputVariable().Shape.Dimensions;
            int targetSize = 1;
            foreach (var d in targetDims)
            {
                targetSize *= d;
            }

            //create databuffer
            dataBuffer = new DataBuffer(maxDataBufferCount,
                new DataBuffer.DataInfo("Input", DataBuffer.DataType.Float, inputSize),
                new DataBuffer.DataInfo("Target", DataBuffer.DataType.Float, targetSize)
            );
        }


        public void AddData(float[] inputs, float[] targets)
        {
            //I am not checking the data size here because the dataBuffer.AddData will check it for me....tooo lazy
            dataBuffer.AddData(new Tuple<string, Array>("Input", inputs), new Tuple<string, Array>("Target", targets));
        }

        //clear all data
        public void ClearData()
        {
            dataBuffer.ClearData();
        }
        /// <summary>
        /// Train the network. 
        /// </summary>
        /// <param name="minibatchSize"></param>
        public void TrainMiniBatch(int minibatchSize)
        {
            var samples = dataBuffer.RandomSample(minibatchSize, new Tuple<string, int, string>("Input", 0, "Input"), new Tuple<string, int, string>("Target", 0, "Target"));
            TrainMiniBatch((float[])samples["Input"], (float[])samples["Target"]);
            /*var inputVar = networkRef.InputLayer.InputVariable;
            var inputValue = Value.CreateBatch(inputVar.Shape, (float[])samples["Input"], Device);

            var targetVar = networkRef.OutputLayer.GetTargetInputVariable();
            var targetValue = Value.CreateBatch(targetVar.Shape, (float[])samples["Target"], Device);

            var outputVar = networkRef.OutputLayer.GetOutputVariable();
            var lossVar = networkRef.OutputLayer.GetTrainingLossVariable();

            var inputDataMap = new Dictionary<Variable, Value>() { { inputVar, inputValue }, { targetVar, targetValue } };
            var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null }, { lossVar, null } };

            //train
            trainer.TrainMinibatch(inputDataMap, false, Device);*/
        }

        public void TrainMiniBatch(float[] inputs, float[] targets)
        {
            var inputVar = networkRef.InputLayer.InputVariable;
            var inputValue = Value.CreateBatch(inputVar.Shape, inputs, Device);

            var targetVar = networkRef.OutputLayer.GetTargetInputVariable();
            var targetValue = Value.CreateBatch(targetVar.Shape, targets, Device);

            //var outputVar = networkRef.OutputLayer.GetOutputVariable();
            //var lossVar = networkRef.OutputLayer.GetTrainingLossVariable();

            var inputDataMap = new Dictionary<Variable, Value>() { { inputVar, inputValue }, { targetVar, targetValue } };
            //var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null }, { lossVar, null } };

            //train
            trainer.TrainMinibatch(inputDataMap, false, Device);
        }

        public void SetLearningRate(float lr)
        {
            learners[0].SetLearningRateSchedule(new TrainingParameterScheduleDouble(lr));
        }
    }


    /// <summary>
    /// A class to hide CNTK learners api
    /// </summary>
    public static class LearnerDefs
    {


        public static SGDLearnerDef SGDLearner(float learningRate)
        {
            return new SGDLearnerDef(learningRate);
        }
        public static MomentumSGDLearnerDef MomentumSGDLearner(float learningRate, float momentum)
        {
            return new MomentumSGDLearnerDef(learningRate, momentum);
        }
        public static AdamLearnerDef AdamLearner(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 0.00001f)
        {
            return new AdamLearnerDef(learningRate, beta1, beta2, epsilon);
        }
        public abstract class LearnerDef
        {
            /// <summary>
            /// create CNTK learner
            /// </summary>
            /// <param name="parameters"></param>
            /// <returns></returns>
            public abstract Learner Create(IList<Parameter> parameters);
        }

        public class SGDLearnerDef : LearnerDef
        {
            private float lr;
            public SGDLearnerDef(float learningRate)
            {
                lr = learningRate;
            }
            public override Learner Create(IList<Parameter> parameters)
            {
                return Learner.SGDLearner(parameters, new TrainingParameterScheduleDouble(lr));
            }
        }

        public class MomentumSGDLearnerDef : LearnerDef
        {
            private float lr;
            private float mom;
            public MomentumSGDLearnerDef(float learningRate, float momentum)
            {
                lr = learningRate;
                mom = momentum;
            }
            public override Learner Create(IList<Parameter> parameters)
            {

                return Learner.MomentumSGDLearner(parameters, new TrainingParameterScheduleDouble(lr), new TrainingParameterScheduleDouble(mom), true);
            }
        }

        public class AdamLearnerDef : LearnerDef
        {
            private float lr;
            private float b1;
            private float b2;
            private float eps;
            public AdamLearnerDef(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 0.00001f)
            {
                lr = learningRate;
                b1 = beta1;
                b2 = beta2;
                eps = epsilon;
            }
            public override Learner Create(IList<Parameter> parameters)
            {

                return CNTKLib.AdamLearner(new ParameterVector(parameters.ToArray()), new TrainingParameterScheduleDouble(lr),
                    new TrainingParameterScheduleDouble(b1), true, new TrainingParameterScheduleDouble(b2), eps);
            }
        }
    }

}