using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityCNTK.ReinforcementLearning;
using UnityCNTK;
using CNTK;
using UnityEngine.UI;
using System.IO;

public class PongPPORunner : MonoBehaviour {
    public PongEnvironment environment;
    public string saveDataPath = "PongPPO.bytes";
    public float learningRate = 0.001f;
    protected PPOModel model;
    protected TrainerPPOSimple trainer;

    public int episodesThisTrain = 0;
    public int trainedCount = 0;

    public int episodeToRunForEachTrain = 30;
    public int iterationForEachTrain = 50;
    public int minibatch = 32;
    public bool training = true;
    [Range(0, 100)]
    public float timeScale;

    public int Steps { get { return trainer.Steps; } }
    [Header("info")]
    public int currentEpisode = 0;
    public int leftWin = 0;
    public int rightWin = 0;

    public AutoAverage winningRate50Left = new AutoAverage(50);
    protected AutoAverage loss;
    public AutoAverage episodePointAve;
    protected float episodePoint;

    // Use this for initialization
    void Start()
    {
        var network = new PPONetworkDiscreteSimple(6, 3, 4, 64, DeviceDescriptor.CPUDevice, 0.01f);
        model = new PPOModel(network);
        trainer = new TrainerPPOSimple(model, LearnerDefs.AdamLearner(learningRate),1, 50000, 2000);

        //test
        //trainer.RewardDiscountFactor = 0.5f;

        loss = new AutoAverage(iterationForEachTrain);
        episodePointAve = new AutoAverage(episodeToRunForEachTrain);
    }

    // Update is called once per frame
    void Update()
    {
        Time.timeScale = timeScale;
        trainer.SetLearningRate(learningRate);
    }

    private void FixedUpdate()
    {
        RunStep();
    }

    protected void RunStep()
    {
        trainer.Step(environment);
        bool reset = trainer.Record(environment);
        episodePoint += environment.LastReward();

        //reset if end
        if (reset && training)
        {
            currentEpisode++;
            if (environment.GameWinPlayer == 0)
            {
                leftWin++;
                winningRate50Left.AddValue(1);
            }
            else
            {
                rightWin++;
                winningRate50Left.AddValue(0);
            }


            environment.Reset();
            episodesThisTrain++;
            episodePointAve.AddValue(episodePoint);
            episodePoint = 0;

            if (episodesThisTrain >= episodeToRunForEachTrain)
            {

                trainer.TrainAllData(minibatch, iterationForEachTrain);
                //record and print the loss
                print("Training Loss:" + trainer.LastLoss);
                trainedCount++;
                trainer.ClearData();
                episodesThisTrain = 0;
            }

        }
    }
    public void Save()
    {
        var data = model.Save();
        File.WriteAllBytes(saveDataPath, data);
    }
    public void Load()
    {
        var bytes = File.ReadAllBytes(saveDataPath);
        model.Restore(bytes);
    }
}
