using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
using System.Linq;
using UnityCNTK;
using UnityCNTK.Tools.StyleAndTexture;
using UnityCNTK.LayerDefinitions;
using System.Threading;
using System;
using MathNet.Numerics;
using Accord.Math;

public class TestSeqNNTrain2D : MonoBehaviour {

    protected SequentialNetworkDense network;
    public DataPlane2D dataPlane;
    protected TrainerSimpleNN trainer;
    public float lr = 0.00001f;
    public bool training = false;
    protected OutputLayerDef outputLayer;
    // Use this for initialization
    void Start () {
        CreateResnode();

        //TestLayerNormalization();

    }
	

	// Update is called once per frame
	void Update () {
        if (training)
        {
            TrainOnce(50);
        }
    }
    
    

    public void CreateResnode()
    {
        var input = new InputLayerDense(2);

        //outputLayer = new OutputLayerDenseBayesian(1);
        outputLayer = new OutputLayerDense(1, ActivationFunction.None, OutputLayerDense.LossFunction.Square);

        network = new SequentialNetworkDense(input, LayerDefineHelper.DenseLayers(10,5,NormalizationMethod.None), outputLayer, DeviceDescriptor.CPUDevice);
        //network = new SequentialNetworkDense(input, LayerDefineHelper.ResNodeLayers(10, 5), outputLayer, DeviceDescriptor.CPUDevice);

        trainer = new TrainerSimpleNN(network, LearnerDefs.AdamLearner(lr),DeviceDescriptor.CPUDevice);

        dataPlane.network = this;
    }

    public void LoadTrainingData()
    {
        trainer.ClearData();
        trainer.AddData(dataPlane.GetDataPositions(), dataPlane.GetDataLabels());
        
    }

    public void TrainOnce(int episodes)
    {
        trainer.SetLearningRate(lr);
        for (int i = 0; i < episodes; ++i)
        {
            trainer.TrainMiniBatch(32);
        }
        print("Training loss:" + trainer.LastLoss);
    }

    public float EvalPosition(Vector2 position)
    {
        var pred = network.EvaluateOne(new float[] { position.x, position.y });
        //var variance = network.EvaluateOne(new float[] { position.x, position.y }, outputLayer.GetVarianceVariable());
        //print("Predicted: " + pred[0] + ", variance: " + variance[0]);
        print("Predicted: " + pred[0]);
        return pred[0];
    }
    
}
