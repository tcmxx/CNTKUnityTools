using System.Collections;
using System.Collections.Generic;
using UnityEngine;


using UnityCNTK;
using CNTK;
using UnityEngine.UI;
using System.IO;

public class PongDQLRunner : MonoBehaviour {
    public PongEnvironment environment;
    public string saveDataPath;
    protected DQLModel model;
    protected DQLModel modelTarget;
    protected TrainerDQLSimple trainer;

    [Header("Training Settings")]
    public int experienceBufferSize = 5000000;
    public int batchSize = 64;
    public float discountFactor = 0.98f;
    public int trainingStepInterval = 10;
    public int stepsBeforeTrain = 100000;
    [Header("Changing RL")]
    public int changeRLSteps = 3000000;
    public float startLearningRate = 0.4f, endLearningRate = 0.05f;
    public float CurrentLearningRate { get {
            return startLearningRate + (endLearningRate - startLearningRate)*Mathf.Clamp01(((float)Steps - stepsBeforeTrain)/changeRLSteps);
        } }
    [Header("Random action settings")]
    public float randomChanceStart = 1.0f;
    public float randomChanceEnd = 0.05f;
    public int randomChanceDropSteps = 1000000;

    [Range(0,100)]
    public float timeScale = 1;
    public bool training = true;

    
    public int Steps { get { return trainer!= null?trainer.Steps:0; } }
    [Header("Information")]
    public int currentEpisode = 0;
    public int leftWin = 0;
    public int rightWin = 0;
    public AutoAverage winningRate50Left = new  AutoAverage(50);
    public AutoAverage reward50EpiLeft = new AutoAverage(50);
    protected float rewardLeftOneEpi = 0;
    public AutoAverage loss = new AutoAverage(500);
    // Use this for initialization
    void Start () {
        QNetworkSimple network = new QNetworkSimple(6,3,2,64,DeviceDescriptor.GPUDevice(0),0.4f);
        model = new DQLModel(network);
        QNetworkSimple networkTarget = new QNetworkSimple(6, 3, 2, 64, DeviceDescriptor.GPUDevice(0), 0.4f);
        modelTarget = new DQLModel(networkTarget);
        //trainer = new TrainerDQLSimple(model, null, LearnerDefs.SGDLearner(startLearningRate),1, experienceBufferSize, 2048);
        trainer = new TrainerDQLSimple(model, modelTarget, LearnerDefs.AdamLearner(startLearningRate), 1, experienceBufferSize, experienceBufferSize);
        //Save();//test
    }
	
	// Update is called once per frame
	void Update () {
        Time.timeScale = timeScale;
        UpdateAllParameters();
    }


    private void FixedUpdate()
    {
        RunStep();
    }


    protected void RunStep()
    {
        trainer.Step(environment);
        rewardLeftOneEpi += environment.LastReward(0);
        bool reset = trainer.Record(environment);

        //training
        if(trainer.Steps >= stepsBeforeTrain && trainer.Steps% trainingStepInterval == 0)
        {
            trainer.TrainRandomBatch(batchSize);

            //log the loss
            loss.AddValue(trainer.LastLoss);
            if (loss.JustUpdated)
            {
                print("Loss:" + loss.Average);
            }
        }
        //reset if end
        if (environment.IsEnd() || (reset && training))
        {
            currentEpisode++;
            if(environment.GameWinPlayer == 0)
            {
                leftWin++;
                winningRate50Left.AddValue(1);
            }
            else
            {
                rightWin++;
                winningRate50Left.AddValue(0);
            }
            reward50EpiLeft.AddValue(rewardLeftOneEpi);
            rewardLeftOneEpi = 0;
            environment.Reset();
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


    public void UpdateAllParameters()
    {
        trainer.DiscountFactor = discountFactor;

        trainer.SetLearningRate(CurrentLearningRate);

        trainer.RandomChanceDropStepInterval = randomChanceDropSteps;
        trainer.StartRandomChance = randomChanceStart;
        trainer.EndRandomChance = randomChanceEnd;
    }


}
