using System.Collections;
using System.Collections.Generic;
using CNTK;
using System;
using System.Linq;
using UnityEngine;

namespace UnityCNTK
{
    public class TrainerGAN
    {
        protected Trainer trainerG;
        protected Trainer trainerD;

        protected List<Learner> learnersG;
        protected List<Learner> learnersD;

        protected GAN ganReference;
        
        public DeviceDescriptor Device { get; set; }

        protected DataBuffer dataBuffer;


        public float LastLossGenerator { get { return (float)trainerG.PreviousMinibatchLossAverage(); } }
        public float LastLossDiscriminator { get { return (float)trainerD.PreviousMinibatchLossAverage(); } }

        public float LearningRateGenerator { get; private set; }
        public float LearningRateDiscriminator { get; private set; }

        protected Dictionary<int,CNTKDictionary> savedLearners;
        protected Dictionary<int, Dictionary<Parameter, NDArrayView>> savedParameters;

        public bool usePredictionInTraining = false;

        public TrainerGAN(GAN gan, LearnerDefs.LearnerDef generatorLearner, LearnerDefs.LearnerDef discriminatorLearner, DeviceDescriptor device, int maxDataBufferCount = 50000)
        {

            Device = device;
            ganReference = gan;

            learnersG = new List<Learner>();
            learnersD = new List<Learner>();

            //trainer for  generator
            Learner learnerG = generatorLearner.Create(gan.GeneratorSequentialModel.Parameters);
            learnersG.Add(learnerG);
            trainerG = Trainer.CreateTrainer(gan.GeneratorOutput, gan.GeneratorLoss, gan.GeneratorLoss, learnersG);
            //trainer for  discriminator
            Learner learnerD = discriminatorLearner.Create(gan.DiscriminatorSequentialModel.Parameters);
            learnersD.Add(learnerD);
            trainerD = Trainer.CreateTrainer(gan.DiscriminatorMerged, gan.DiscriminatorLoss, gan.DiscriminatorLoss, learnersD);

            //create databuffer
            List<DataBuffer.DataInfo> dataInfos = new List<DataBuffer.DataInfo>();

            if(gan.InputConditionSize > 0)
            {
                dataInfos.Add(new DataBuffer.DataInfo("Condition", DataBuffer.DataType.Float, gan.InputConditionSize));
            }
            dataInfos.Add(new DataBuffer.DataInfo("Target", DataBuffer.DataType.Float, gan.OutputSize));
            dataBuffer = new DataBuffer(maxDataBufferCount, dataInfos.ToArray());

            //others
            savedLearners = new Dictionary<int, CNTKDictionary>();
            savedParameters = new Dictionary<int, Dictionary<Parameter, NDArrayView>>();
        }


        public void AddData(float[] inputConditions, float[] inputTargets)
        {
            //I am not checking the data size here because the dataBuffer.AddData will check it for me....tooo lazy
            List<Tuple<string, Array>> data = new List<Tuple<string, Array>>();
            if (ganReference.InputConditionSize > 0)
            {
                data.Add(new Tuple<string, Array>("Condition", inputConditions));
            }
            data.Add(new Tuple<string, Array>("Target", inputTargets));
            dataBuffer.AddData(data.ToArray());
        }

        //clear all data
        public void ClearData()
        {
            dataBuffer.ClearData();
        }


        public void TrainMiniBatch(int minibatchSize)
        {
            var fetches = new List<Tuple<string, int, string>>();
            fetches.Add(new Tuple<string, int, string>("Target", 0, "Target"));
            if (ganReference.InputConditionSize > 0)
            {
                fetches.Add(new Tuple<string, int, string>("Condition", 0, "Condition"));
            }
            var samples = dataBuffer.RandomSample(minibatchSize, fetches.ToArray());

            float[] targets = (float[])samples["Target"];
            float[] conditions = samples.ContainsKey("Condition") ? (float[])samples["Condition"] : null;
            TrainMiniBatch(Utils.GenerateWhiteNoise(ganReference.InputNoiseSize*minibatchSize,-1,1), conditions, targets);
        }

        public void TrainMiniBatch(float[] inputNoises,float[] inputConditions,float[] inputTargets)
        {

            //create input maps
            bool hasConditionInput = ganReference.InputConditionGenerator != null;
            bool hasNoiseInput = ganReference.InputNoiseGenerator != null;

            var inputMapGeneratorTrain = new Dictionary<Variable, Value>();
            var inputMapDiscriminatorTrain = new Dictionary<Variable, Value>();

            if (hasConditionInput)
            {
                var inputCondition = Value.CreateBatch(ganReference.InputConditionGenerator.Shape, inputConditions, Device);
                //generator training needed inputs
                inputMapGeneratorTrain.Add(ganReference.InputConditionGenerator, inputCondition);
                //discriminator training needed inputs
                inputMapDiscriminatorTrain.Add(ganReference.InputConditionGenerator, inputCondition);
                inputMapDiscriminatorTrain.Add(ganReference.InputConditionDiscriminatorFake, inputCondition);
                inputMapDiscriminatorTrain.Add(ganReference.InputConditionDiscriminatorReal, inputCondition);
                
            }

            if (hasNoiseInput)
            {
                var inputNoise = Value.CreateBatch(ganReference.InputNoiseGenerator.Shape, inputNoises, Device);
                //generator training needed inputs
                inputMapGeneratorTrain.Add(ganReference.InputNoiseGenerator, inputNoise);
                //discriminator training needed inputs
                inputMapDiscriminatorTrain.Add(ganReference.InputNoiseGenerator, inputNoise);
            }
            var inputTarget = Value.CreateBatch(ganReference.InputDataDiscriminatorReal.Shape, inputTargets, Device);
            //generator training needed inputs
            inputMapGeneratorTrain.Add(ganReference.InputTargetGenerator, inputTarget);
            //discriminator training needed inputs
            inputMapDiscriminatorTrain.Add(ganReference.InputDataDiscriminatorReal, inputTarget);

            if (usePredictionInTraining)
            {
                TrainWithPredictionBySaving(inputMapGeneratorTrain, inputMapDiscriminatorTrain);
            }
            else {
                TrainDefault(inputMapGeneratorTrain, inputMapDiscriminatorTrain);
            }
            //TrainWithReorder(inputMapGeneratorTrain, inputMapDiscriminatorTrain);
            //Debug.Log("G loss: " + trainerG.PreviousMinibatchLossAverage());
            //Debug.Log("D loss: " + trainerD.PreviousMinibatchLossAverage());
            //Debug.Log(learnersG[0].LearningRate());

        }

        public void SetLearningRateGenerator(float lr)
        {
            LearningRateGenerator = lr;
            learnersG[0].SetLearningRateSchedule(new TrainingParameterScheduleDouble(lr));
        }
        public void SetLearningRateDiscriminator(float lr)
        {
            LearningRateDiscriminator = lr;
            learnersD[0].SetLearningRateSchedule(new TrainingParameterScheduleDouble(lr));
        }



        protected void TrainWithPredictionBySaving(IDictionary<Variable,Value> inputMapGeneratorTrain, IDictionary<Variable, Value> inputMapDiscriminatorTrain)
        {
            SaveGenerator(0);
            trainerG.TrainMinibatch(inputMapGeneratorTrain, false, Device);
            SaveGenerator(1);
            RestoreGenerator(0);
            learnersG[0].SetLearningRateSchedule(new TrainingParameterScheduleDouble(LearningRateGenerator * 2));
            trainerG.TrainMinibatch(inputMapGeneratorTrain, false, Device);
            trainerD.TrainMinibatch(inputMapDiscriminatorTrain, false, Device);
            RestoreGenerator(1);
        }
        protected void TrainDefault(IDictionary<Variable, Value> inputMapGeneratorTrain, IDictionary<Variable, Value> inputMapDiscriminatorTrain)
        {
            trainerG.TrainMinibatch(inputMapGeneratorTrain, false, Device);
            trainerD.TrainMinibatch(inputMapDiscriminatorTrain, false, Device);
        }
        protected void TrainWithReorder(IDictionary<Variable, Value> inputMapGeneratorTrain, IDictionary<Variable, Value> inputMapDiscriminatorTrain)
        {
            SaveGenerator(0);
            learnersG[0].SetLearningRateSchedule(new TrainingParameterScheduleDouble(LearningRateGenerator));
            trainerG.TrainMinibatch(inputMapGeneratorTrain, false, Device);
            trainerD.TrainMinibatch(inputMapDiscriminatorTrain, false, Device);
            RestoreGenerator(0);
            trainerG.TrainMinibatch(inputMapGeneratorTrain, false, Device);
        }



        public void SaveGenerator(int key)
        {
            savedLearners[key] = learnersG[0].CreateCheckpoint();
            savedParameters[key] = new Dictionary<Parameter, NDArrayView>();
            foreach (var p in ganReference.GeneratorSequentialModel.Parameters)
            {
                savedParameters[key][p] = p.GetValue().DeepClone(Device,true);
            }
            
        }
        public void RestoreGenerator(int key)
        {
            if (savedLearners.ContainsKey(key) && savedParameters.ContainsKey(key))
            {
                learnersG[0].RestoreFromCheckpoint(savedLearners[key]);
                foreach (var p in ganReference.GeneratorSequentialModel.Parameters)
                {
                    p.SetValue(savedParameters[key][p]);
                }
            }

        }
        /*
        public void SaveDiscriminator(int key)
        {
            savedLearners[key] = learnersD[0].CreateCheckpoint();
            savedParameters[key] = new Dictionary<Parameter, NDArrayView>();
            foreach (var p in ganReference.DiscriminatorSequentialModel.Parameters)
            {
                savedParameters[key][p] = p.GetValue().DeepClone(Device, true);
            }
        }
        public void RestoreDiscriminator(int key)
        {
            if (savedLearners.ContainsKey(key) && savedParameters.ContainsKey(key))
            {
                learnersD[0].RestoreFromCheckpoint(savedLearners[key]);
                foreach (var p in ganReference.DiscriminatorSequentialModel.Parameters)
                {
                    p.SetValue(savedParameters[key][p]);
                }
            }

        }*/
    }
}