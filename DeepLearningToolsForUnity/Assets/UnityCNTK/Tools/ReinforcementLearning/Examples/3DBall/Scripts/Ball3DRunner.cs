using System.Collections;
using System.Collections.Generic;
using UnityEngine;


using UnityCNTK;
using CNTK;
using UnityEngine.UI;

public class Ball3DRunner : MonoBehaviour {
    public Text scoreUI;
    public Ball3DEnviroment environment;
    public float learningRate = 0.001f;
    protected PPOModel model;
    protected TrainerPPOSimple trainer;

    public int episodesThisTrain = 0;
    public int trainedCount = 0;

    public int episodeToRunForEachTrain = 30;
    public int iterationForEachTrain = 50;
    public int minibatch = 32;

    [Range(0,100)]
    public float timeScale;
    public bool training = true;

    protected AutoAverage loss;
    protected AutoAverage episodePointAve;
    protected float episodePoint;

    // Use this for initialization
    void Start () {
        PPONetworkContinuousSimple network;
        if (environment.is3D)
        {
            network = new PPONetworkContinuousSimple(8, 2, 5, 64, DeviceDescriptor.CPUDevice, 0.01f);
        }
        else
        {
            network = new PPONetworkContinuousSimple(5, 1, 5, 64, DeviceDescriptor.CPUDevice, 0.01f);
        }
        model = new PPOModel(network);
        trainer = new TrainerPPOSimple(model, LearnerDefs.AdamLearner(learningRate), 10000, 200);

        //test
        //trainer.RewardDiscountFactor = 0.5f;

        loss = new AutoAverage(iterationForEachTrain);
        episodePointAve = new AutoAverage(episodeToRunForEachTrain); 
    }
	
	// Update is called once per frame
	void Update () {
        Time.timeScale = timeScale;
        trainer.SetLearningRate(learningRate);
    }

    private void FixedUpdate()
    {
        RunStep();
        //RunTest();
    }


    protected void RunStep()
    {
        trainer.Step(environment);
        bool reset = trainer.Record(environment);
        episodePoint += environment.LastReward();

        //reset if end
        if (reset && training)
        {
            environment.Reset();
            episodesThisTrain++;
            episodePointAve.AddValue(episodePoint);
            if (episodePointAve.JustUpdated)
            {
                scoreUI.text = "Average points:" + episodePointAve.Average;
            }
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
        }else if (environment.IsEnd())
        {
            environment.Reset();
        }
    }
    
}
