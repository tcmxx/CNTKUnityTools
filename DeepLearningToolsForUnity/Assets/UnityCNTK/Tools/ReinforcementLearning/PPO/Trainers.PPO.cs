using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

namespace UnityCNTK
{

    public interface IRLEnvironment
    {
        void Step(float[] action);
        float LastReward();
        float[] CurrentState();
        int CurrentStep();
        bool IsEnd();
        bool IsResolved();
        void Reset();
    }

    public class TrainerPPOSimple
    {
        
        protected DataBuffer dataBuffer;
        public int DataCountStored { get { return dataBuffer.CurrentCount; } }
        public PPOModel Model { get; protected set; }
       
        public int MaxStepHorizon { get; set; }


        public float RewardDiscountFactor { get; set; } = 0.99f;
        public float RewardGAEFactor { get; set; } = 0.95f;
        public float ValueLossWeight { get; set; } = 1f;
        public float EntroyLossWeight { get; set; } = 0.0f;
        public float ClipEpsilon { get; set; } = 0.2f;




        protected List<float> statesEpisodeHistory;
        protected List<float> rewardsEpisodeHistory;
        protected List<float> actionsEpisodeHistory;
        protected List<float> actionprobsEpisodeHistory;
        protected List<float> valuesEpisodeHistory;



        public float[] LastState { get; private set; }
        public float[] LastAction { get; protected set; }
        public float[] LastActionProbs { get; protected set; }
        public float LastValue { get; protected set; }


        public float LastLoss { get; protected set; }
        //public float LastEvalLoss { get; protected set; }

        protected Trainer trainer;

        protected List<Learner> learners;

        public TrainerPPOSimple(PPOModel model, LearnerDefs.LearnerDef learner, int bufferSize = 2048, int maxStepHorizon = 2048)
        {
            Model = model;
            MaxStepHorizon = maxStepHorizon;

            statesEpisodeHistory = new List<float>();
            rewardsEpisodeHistory = new List<float>();
            actionsEpisodeHistory = new List<float>();
            valuesEpisodeHistory = new List<float>();
            actionprobsEpisodeHistory = new List<float>();

            dataBuffer = new DataBuffer(bufferSize,
                new DataBuffer.DataInfo("State", DataBuffer.DataType.Float, Model.StateSize),
                new DataBuffer.DataInfo("Action", DataBuffer.DataType.Float, Model.IsActionContinuous?Model.ActionSize:1),
                new DataBuffer.DataInfo("ActionProb", DataBuffer.DataType.Float, Model.IsActionContinuous ? Model.ActionSize : 1),
                new DataBuffer.DataInfo("TargetValue", DataBuffer.DataType.Float, 1),
                new DataBuffer.DataInfo("Advantage", DataBuffer.DataType.Float, 1)
                );

            learners = new List<Learner>();
            List<Parameter> parameters = new List<Parameter>(Model.OutputLoss.ToFunction().Parameters());
            learners.Add(learner.Create(parameters));   

            //test
            trainer = Trainer.CreateTrainer(Model.OutputLoss, Model.OutputLoss, Model.OutputLoss, learners);
        }


        /// <summary>
        /// Step the enviormenet for training.
        /// </summary>
        /// <param name="environment"></param>
        public void Step(IRLEnvironment environment)
        {
            float[] action = null;
            float[] actionProbs = null;
            LastState = environment.CurrentState().CopyToArray();

            if (Model.IsActionContinuous)
            {
                action = Model.EvaluateActionContinuous(LastState, out actionProbs);
            }
            else
            {
                int[] tempAction = Model.EvaluateActionDiscrete(LastState, out actionProbs, true);
                action = new float[1] { tempAction[0] };
            }
            LastAction = action;
            LastActionProbs = actionProbs;

            LastValue = Model.EvaluateValue(LastState)[0];

            environment.Step(action);
        }

        /// <summary>
        /// called after step and when the enviorment is resolved. return whether the enviourment should reset
        /// </summary>
        /// <param name="environment"></param>
        public bool Record(IRLEnvironment environment)
        {
            Debug.Assert(environment.IsResolved());
            bool isEnd = environment.IsEnd();
            float reward = environment.LastReward();

            AddHistory(LastState, reward, LastAction, LastActionProbs, LastValue);
            if (isEnd || environment.CurrentStep() >= MaxStepHorizon) {
                float nextValue = 0;
                if (!isEnd)
                {
                    nextValue = Model.EvaluateValue(environment.CurrentState())[0];
                }
                ProcessEpisodeHistory(nextValue);
                return true;
            }
            return false;
        }


        public void TrainAllData(int batchsize, int epoch)
        {
            var fetches = new List<Tuple<string, int, string>>();
            fetches.Add(new Tuple<string, int, string>("State", 0, "State"));
            fetches.Add(new Tuple<string, int, string>("Action", 0, "Action"));
            fetches.Add(new Tuple<string, int, string>("ActionProb", 0, "ActionProb"));
            fetches.Add(new Tuple<string, int, string>("TargetValue", 0, "TargetValue"));
            fetches.Add(new Tuple<string, int, string>("Advantage", 0, "Advantage"));

            float lossEpoch = 0;
            float lossMean = 0;
            for (int i = 0; i < epoch; ++i)
            {
                var samples = dataBuffer.SampleBatchesReordered(batchsize, fetches.ToArray());
                float[] states = (float[])samples["State"];
                float[] actions = (float[])samples["Action"];
                float[] actionProbs = (float[])samples["ActionProb"];
                float[] targetValues = (float[])samples["TargetValue"];
                float[] advantages = (float[])samples["Advantage"];

                int batchCount = targetValues.Length / batchsize;
                for(int j =0;j < batchCount; ++j)
                {
                    TrainBatch(SubArray(states,j*batchsize*Model.StateSize, batchsize * Model.StateSize), 
                        SubArray(actions, j * batchsize * Model.ActionSize, batchsize * Model.ActionSize),
                        SubArray(actionProbs, j * batchsize * Model.ActionSize, batchsize * Model.ActionSize),
                        SubArray(targetValues, j * batchsize, batchsize),
                        SubArray(advantages, j * batchsize, batchsize));
                    lossEpoch += LastLoss;
                }
                lossMean += lossEpoch / batchCount;
            }
            LastLoss = lossMean / epoch;
        }

        public void TrainRandomBatch(int batchSize)
        {
            var fetches = new List<Tuple<string, int, string>>();
            fetches.Add(new Tuple<string, int, string>("State", 0, "State"));
            fetches.Add(new Tuple<string, int, string>("Action", 0, "Action"));
            fetches.Add(new Tuple<string, int, string>("ActionProb", 0, "ActionProb"));
            fetches.Add(new Tuple<string, int, string>("TargetValue", 0, "TargetValue"));
            fetches.Add(new Tuple<string, int, string>("Advantage", 0, "Advantage"));
            var samples = dataBuffer.RandomSample(batchSize, fetches.ToArray());

            float[] states = (float[])samples["State"];
            float[] actions = (float[])samples["Action"];
            float[] actionProbs = (float[])samples["ActionProb"];
            float[] targetValues = (float[])samples["TargetValue"];
            float[] advantages = (float[])samples["Advantage"];
            TrainBatch(states, actions, actionProbs, targetValues, advantages);
            
        }
        
        public void TrainBatch(float[] states,float[] actions,float[] actionProbs,float[] targetValues,float[] advantages)
        {
            //input map
            var inputMapGeneratorTrain = new Dictionary<Variable, Value>();
            var inputActions = Value.CreateBatch(Model.InputAction.Shape, actions, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputAction, inputActions);
            var inputStates = Value.CreateBatch(Model.Network.InputState.Shape, states, Model.Device);
            inputMapGeneratorTrain.Add(Model.Network.InputState, inputStates);
            var inputAdvantages = Value.CreateBatch(Model.InputAdvantage.Shape, advantages, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputAdvantage, inputAdvantages);
            var inputTargetValues = Value.CreateBatch(Model.InputTargetValue.Shape, targetValues, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputTargetValue, inputTargetValues);
            var inputOldProbs = Value.CreateBatch(Model.InputOldProb.Shape, actionProbs, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputOldProb, inputOldProbs);

            var inputClipEps = Value.CreateBatch(Model.InputClipEpsilon.Shape, new float[] { ClipEpsilon }, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputClipEpsilon, inputClipEps);
            var inputValueLossWeight = Value.CreateBatch(Model.InputValuelossWeight.Shape, new float[] { ValueLossWeight }, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputValuelossWeight, inputValueLossWeight);
            var inputEntropyLossWeight = Value.CreateBatch(Model.InputEntropyLossWeight.Shape, new float[] { EntroyLossWeight }, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputEntropyLossWeight, inputEntropyLossWeight);

            //output Map
            trainer.TrainMinibatch(inputMapGeneratorTrain, false, Model.Device);

            LastLoss = (float)trainer.PreviousMinibatchLossAverage();
            //LastEvalLoss = (float)trainer.PreviousMinibatchEvaluationAverage();
        }



        /// <summary>
        /// calcualte the discounted advantages for the current sequence of data, and add them to the databuffer
        /// </summary>
        protected void ProcessEpisodeHistory(float nextValue)
        {
            var advantages = RLUtils.GeneralAdvantageEst(rewardsEpisodeHistory.ToArray(), valuesEpisodeHistory.ToArray(), RewardDiscountFactor, RewardGAEFactor, nextValue);
            float[] targetValues = new float[advantages.Length];
            for(int i = 0; i < targetValues.Length; ++i)
            {
                targetValues[i] = advantages[i] + valuesEpisodeHistory[i];

                //test 
                //advantages[i] = 1;
            }
            //test
            //targetValues = RLUtils.DiscountedRewards(rewardsEpisodeHistory.ToArray(), RewardDiscountFactor);


            dataBuffer.AddData(Tuple.Create<string, Array>("State", statesEpisodeHistory.ToArray()),
                Tuple.Create<string, Array>("Action", actionsEpisodeHistory.ToArray()),
                Tuple.Create<string, Array>("ActionProb", actionprobsEpisodeHistory.ToArray()),
                Tuple.Create<string, Array>("TargetValue", targetValues),
                Tuple.Create<string, Array>("Advantage", advantages)
                );

            statesEpisodeHistory.Clear();
            rewardsEpisodeHistory.Clear();
            actionsEpisodeHistory.Clear();
            actionprobsEpisodeHistory.Clear();
            valuesEpisodeHistory.Clear();
        }



        protected void AddHistory(float[] state, float reward, float[] action, float[] actionProbs, float value)
        {
            statesEpisodeHistory.AddRange(state);
            rewardsEpisodeHistory.Add(reward);
            actionsEpisodeHistory.AddRange(action);
            actionprobsEpisodeHistory.AddRange(actionProbs);
            valuesEpisodeHistory.Add(value);
            
        }


        public void ClearData()
        {
            dataBuffer.ClearData();
        }


        public void SetLearningRate(float lr)
        {
            learners[0].SetLearningRateSchedule(new TrainingParameterScheduleDouble(lr));
        }



        public static T[] SubArray<T>(T[] data, int index, int length)
        {
            T[] result = new T[length];
            Array.Copy(data, index, result, 0, length);
            return result;
        }
    }
}