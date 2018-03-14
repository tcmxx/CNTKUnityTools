using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

#if TensorFlow
using TensorFlow;
public static class TFSharpUtils {

    public static void SVD(float[,] covMat, out float[] s, out float[,] v)
    {
        TFShape shape = new TFShape(covMat.GetLength(0), covMat.GetLength(1));
        var reshaped = covMat.Reshape();
        var inputTensor = TFTensor.FromBuffer(shape, reshaped, 0, reshaped.Length);
        
        TFGraph svdGraph = new TFGraph();
        TFOutput input = svdGraph.Placeholder(TFDataType.Float, shape);
        var svdResult = (ValueTuple<TFOutput,TFOutput,TFOutput>)svdGraph.Svd(input, true);
        
        var sess = new TFSession(svdGraph);
        var runner = sess.GetRunner();
        runner.AddInput(input, inputTensor);
        runner.Fetch(svdResult.Item1);
        runner.Fetch(svdResult.Item2);

        TFTensor[] results = runner.Run();
        s = (float[])results[0].GetValue();
        v = (float[,])results[1].GetValue();
        TFStatus temp = new TFStatus();
        sess.CloseSession(temp);
        sess.DeleteSession(temp);
    }

}
#endif