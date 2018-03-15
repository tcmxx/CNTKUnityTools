using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

namespace UnityCNTK
{
    
    public class TrainerDQLSimple
    {
        
        protected DataBuffer dataBuffer;
        public int DataCountStored { get { return dataBuffer.CurrentCount; } }
        public DQLModel Model { get; protected set; }
       
        public int MaxStepHorizon { get; set; }
        public float DiscountFactor { get; set; } = 0.99f;

        public int NumberOfActor { get; private set; } = 1;



        protected Dictionary<int, List<float>> statesEpisodeHistory;
        protected Dictionary<int,List<float>> rewardsEpisodeHistory;
        protected Dictionary<int, List<float>> actionsEpisodeHistory;
        protected Dictionary<int, List<float>> gameEndEpisodeHistory;


        public Dictionary<int, float[]> LastState { get; private set; }
        public  Dictionary<int, int> LastAction { get; protected set; }

        public float LastLoss { get; protected set; }
        //public float LastEvalLoss { get; protected set; }

        protected Trainer trainer;

        protected List<Learner> learners;

        public TrainerDQLSimple(DQLModel model, LearnerDefs.LearnerDef learner, int numberOfActor = 1, int bufferSize = 2048, int maxStepHorizon = 2048)
        {
            Model = model;
            MaxStepHorizon = maxStepHorizon;
            NumberOfActor = numberOfActor;

            statesEpisodeHistory = new Dictionary<int, List<float>>();
            rewardsEpisodeHistory = new Dictionary<int, List<float>>();
            actionsEpisodeHistory = new Dictionary<int, List<float>>();
            gameEndEpisodeHistory = new Dictionary<int, List<float>>();

            LastAction = new Dictionary<int, int>();


            dataBuffer = new DataBuffer(bufferSize,
                new DataBuffer.DataInfo("State", DataBuffer.DataType.Float, Model.StateSize),
                new DataBuffer.DataInfo("Action", DataBuffer.DataType.Float, 1),
                new DataBuffer.DataInfo("Reward", DataBuffer.DataType.Float, 1),
                new DataBuffer.DataInfo("GameEnd", DataBuffer.DataType.Float, 1)
                );

            learners = new List<Learner>();
            List<Parameter> parameters = new List<Parameter>(Model.OutputLoss.ToFunction().Parameters());
            learners.Add(learner.Create(parameters));   

            trainer = Trainer.CreateTrainer(Model.OutputLoss, Model.OutputLoss, Model.OutputLoss, learners);
        }


        /// <summary>
        /// Step the enviormenet for training.
        /// </summary>
        /// <param name="environment"></param>
        public void Step(IRLEnvironment environment)
        {
            float[][] actions = new float[NumberOfActor][];
            float[] statesAll = new float[NumberOfActor * Model.StateSize];
            for (int i = 0; i < NumberOfActor; ++i)
            {
                var states = environment.CurrentState(i).CopyToArray();
                LastState[i] = states;
                Array.Copy(states, 0, statesAll, i * Model.StateSize, Model.StateSize);
            }
            float[] maxQs;
            int[] tempAction = Model.EvaluateAction(statesAll, out maxQs);
            
            for(int i = 0; i < NumberOfActor; ++i)
            {
                actions[i] = new float[] { tempAction[i] };
                LastAction[i] = tempAction[i];
            }
            environment.Step(actions);
        }

        /// <summary>
        /// called after step and when the enviorment is resolved. return whether the enviourment should reset
        /// </summary>
        /// <param name="environment"></param>
        protected virtual bool Record(IRLEnvironment environment)
        {
            Debug.Assert(environment.IsResolved());
            bool isEnd = environment.IsEnd();
            
            if (isEnd || environment.CurrentStep() >= MaxStepHorizon) {
                for (int i = 0; i < NumberOfActor; ++i)
                {
                    float reward = environment.LastReward(i);
                    AddHistory(i,LastState[i], reward, LastAction[i], true);
                    UpdateReplayBuffer(i);
                }
                
                return true;
            }
            return false;
        }

        public void TrainRandomBatch(int batchSize)
        {
            var samples = SampleFromBufferAll(batchSize);

            float[] states = (float[])samples["State"];
            float[] actions = (float[])samples["Action"];
            float[] gameEnds = (float[])samples["GameEnd"];
            float[] nextStates = (float[])samples["NextState"];
            float[] rewards = (float[])samples["Reward"];
            TrainBatch(states, nextStates, actions, rewards, gameEnds);
            
        }
        
        public void TrainBatch(float[] states,float[] nextStates, float[] actions,float[] rewards,float[] gameEnd)
        {
            int batchSize = gameEnd.Length;
            //evaluate and calculate target Qs
            float[] targetQs = new float[batchSize];
            float[] nextMaxQs;
            Model.EvaluateAction(nextStates, out nextMaxQs);
            for (int i = 0; i < batchSize; ++i)
            {
                targetQs[i] = nextMaxQs[i] * gameEnd[i] * DiscountFactor + rewards[i];
            }

            //input map
            var inputMapGeneratorTrain = new Dictionary<Variable, Value>();

            var inputStates = Value.CreateBatch(Model.InputState.Shape, states, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputState, inputStates);
            
            var inputTargetQs = Value.CreateBatch(Model.InputTargetQ.Shape, targetQs, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputTargetQ, inputTargetQs);

            var inputOldAction = Value.CreateBatch(Model.InputOldAction.Shape, actions, Model.Device);
            inputMapGeneratorTrain.Add(Model.InputOldAction, inputOldAction);
            
            //train
            trainer.TrainMinibatch(inputMapGeneratorTrain, false, Model.Device);

            LastLoss = (float)trainer.PreviousMinibatchLossAverage();
        }



        public void ClearData()
        {
            dataBuffer.ClearData();
        }


        public void SetLearningRate(float lr)
        {
            learners[0].SetLearningRateSchedule(new TrainingParameterScheduleDouble(lr));
        }
        
        protected Dictionary<string, Array> SampleFromBufferAll(int size)
        {
            var samples = dataBuffer.RandomSample(size,
                Tuple.Create<string, int, string>("State", 0, "State"),
                Tuple.Create<string, int, string>("State", 1, "NextState"),
                Tuple.Create<string, int, string>("Action", 0, "Action"),
                Tuple.Create<string, int, string>("Reward", 0, "Reward"),
                Tuple.Create<string, int, string>("GameEnd", 0, "GameEnd"));
            return samples;
        }


        //this should be 
        protected void UpdateReplayBuffer(int agentNum)
        {
            //print("test");
            dataBuffer.AddData(Tuple.Create<string, Array>("State", statesEpisodeHistory[agentNum].ToArray()),
                Tuple.Create<string, Array>("Action", actionsEpisodeHistory[agentNum].ToArray()),
                Tuple.Create<string, Array>("Reward", rewardsEpisodeHistory[agentNum].ToArray()),
                Tuple.Create<string, Array>("GameEnd", gameEndEpisodeHistory[agentNum].ToArray())
                );

            statesEpisodeHistory.Clear();
            rewardsEpisodeHistory.Clear();
            actionsEpisodeHistory.Clear();
            gameEndEpisodeHistory.Clear();
        }

        void AddHistory(int agentNum, float[] state, float reward, float action, bool gameEnd)
        {
            statesEpisodeHistory[agentNum].AddRange(state);
            rewardsEpisodeHistory[agentNum].Add(reward);
            actionsEpisodeHistory[agentNum].Add(action);
            gameEndEpisodeHistory[agentNum].Add(gameEnd ? 0 : 1);
        }
    }
}