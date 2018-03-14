using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using Accord.Math;
using Accord.Math.Decompositions;
using System;
using CNTK;
using System.Diagnostics;

namespace UnityCNTK
{
    public static partial class Images
    {

        /// <summary>
        /// from HxWxC input, reshape it to (H*W)xC, get mean and covariance matrix for channels
        /// Uses CNTK . This is priamarily used by whitening related
        /// </summary>
        /// <param name="input"></param>
        /// <param name="device"></param>
        /// <returns>tuple of cov, mean and reshaped inputs</returns>
        private static Tuple<Value, Value, Value> CovarianceMatrixFromImage(Value image, DeviceDescriptor device, out Tuple<Variable, Variable, Variable> outputs)
        {
            //create the CNTK model to multiply and reshape to get the covariance matrix, also get the mean
            NDShape inputShape = image.Shape;
            Variable input = Variable.InputVariable(inputShape, DataType.Float, "input");
            Variable reshaped = CNTKLib.Reshape(input, new int[] { inputShape[0] * inputShape[1], inputShape[2] });//change the shape to (W*H)xC
            Variable mean = CNTKLib.ReduceMean(reshaped, new Axis(0));
            Variable centered = CNTKLib.Minus(reshaped, mean);
            Variable covMat = CNTKLib.ElementDivide(CNTKLib.TransposeTimes(centered, centered),Constant.Scalar(DataType.Float, inputShape[0] * inputShape[1]-1.0));

            var inputDataMap = new Dictionary<Variable, Value>() { { input, image } };
            var outputDataMap = new Dictionary<Variable, Value>() { { covMat, null }, { mean, null }, { centered, null } };
            ((Function)covMat).Evaluate(inputDataMap, outputDataMap, false, device);

            outputs = new Tuple<Variable, Variable, Variable>(covMat, mean, reshaped);
            var result =  new Tuple<Value, Value, Value>(outputDataMap[covMat], outputDataMap[mean], outputDataMap[centered]);
            return result;
        }


        /// <summary>
        /// Get the E and D matrix for whitening or coloring. Implemented using Accord.net.
        /// </summary>
        /// <param name="covMat"></param>
        /// <param name="eps"></param>
        /// <param name="ignoredSingularValue"></param>
        /// <param name="beta"> when beta is 0.5, it is coloring. When beta is -0.5, it is whitening.You can also try other values</param>
        /// <returns>a tuple of E and D matrices as the first and second element</returns>
        private static Tuple<float[,], float[,]> GetWCMatrices(float[,] covMat, float eps = 0.00001f, float ignoredSingularValue = 0.00001f, float beta = -0.5f)
        {
            //no use the singular value decompositoin to do the whitening
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            //mathnet is a little faster tha accord.net
            float[] diag;
            float[,] E ;
            //bool useTF = false;  //test for tensorflow
            //if (useTF)
            //{
            //    //TFSharpUtils.SVD(covMat, out diag, out E);
                GC.Collect();
            //}
           // else
            //{
                var m = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfArray(covMat);
                var svdMathnet = m.Svd(true);
                diag = svdMathnet.S.ToArray();
                E = svdMathnet.U.ToArray();
            //}
            stopwatch.Stop();
            //UnityEngine.Debug.Log("------Otained Singular values and matrix using"+( useTF ?"tensorflow":"math.net")+ ". Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);
            stopwatch.Restart();


            //count the valid components based on the singular values and get rid of the too small ones
            int countValid = 0;
            for (int i = 0; i < diag.Length; ++i)
            {
                countValid = i;
                if (diag[i] < ignoredSingularValue)
                {
                    break;
                }
            }

            float[] validDiag = new float[countValid];
            Array.Copy(diag, validDiag, countValid);

            //create the diagonal matrix from singular value with pow of -0.5 by default(should not use other power normally)
            float[,] D = Matrix.Diagonal(countValid, countValid, validDiag.Add(eps).Pow(beta));
            //create the E matrix for valid singular values
            int[] indices = new int[countValid];
            for (int i = 0; i < countValid; ++i)
            {
                indices[i] = i;
            }
            float[,] validE = E.GetColumns(indices);
            stopwatch.Stop();
            //UnityEngine.Debug.Log("------Proccessed singular values and matrix. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);
            return new Tuple<float[,], float[,]>(validE, D);
        }


        /// <summary>
        /// Multiply the whiltening/coloring matrices and probably add the mean and reshape the output, then blend with the blend input with factor alpha
        /// see https://arxiv.org/pdf/1705.08086.pdf for the equations
        /// </summary>
        /// <param name="dataToTransform"></param>
        /// <param name="E1"></param>
        /// <param name="D1"></param>
        /// <param name="device"></param>
        /// <param name="E2"></param>
        /// <param name="D2"></param>
        /// <param name="finalAdd"></param>
        /// <param name="outputShape"></param>
        /// <param name="finalBlend"></param>
        /// <param name="blendFactor"></param>
        /// <returns></returns>
        private static Tuple<Value, Function> MultiplyWTC(Value dataToTransform, float[,] E1, float[,] D1, DeviceDescriptor device, float[,] E2 = null, float[,] D2 = null, Value finalAdd =null, NDShape outputShape=null, Value finalBlend = null, float blendFactor = 1)
        {
            //check the dimensions
            UnityEngine.Debug.Assert(D1.GetLength(1) == E1.GetLength(1), "E1 and D1 must have the same number of colunms");
            UnityEngine.Debug.Assert(D1.GetLength(0) == D1.GetLength(1), "D1 matrix must be square");

            NDShape inputShape = dataToTransform.Shape;

            UnityEngine.Debug.Assert(inputShape[1] == E1.GetLength(0), "dataToTransform's number of colunms must be the same as E1's number of rows");
   
            

            //create inputs
            //inputs for the input data
            Variable inputData = Variable.InputVariable(new int[] { inputShape[0], inputShape[1] }, DataType.Float);
            //inputs for the mean
            Variable inputMean = null;
            if (finalAdd != null)
            {
                inputMean = Variable.InputVariable(new int[] { finalAdd.Shape[0], finalAdd.Shape[1] }, DataType.Float);
            }
            //inputs for E1 and D1
            Variable inputE1 = Variable.InputVariable(new int[] { E1.GetLength(0), E1.GetLength(1) }, DataType.Float);
            Variable inputD1 = Variable.InputVariable(new int[] { D1.GetLength(0), D1.GetLength(1) }, DataType.Float);
            //inputs for E2 and D2
            Variable inputE2 = null;
            Variable inputD2 = null;
            if(D2 != null && E2 != null)
            {
                //check the dimensions
                UnityEngine.Debug.Assert(D2.GetLength(1) == E2.GetLength(1), "E2 and D2 must have the same number of colunms");
                UnityEngine.Debug.Assert(D2.GetLength(0) == D2.GetLength(1), "D2 matrix must be square");
                UnityEngine.Debug.Assert(inputShape[1] == E2.GetLength(0), "dataToTransform's number of colunms must be the same as E2's number of rows");
                inputE2 = Variable.InputVariable(new int[] { E2.GetLength(0), E2.GetLength(1) }, DataType.Float);
                inputD2 = Variable.InputVariable(new int[] { D2.GetLength(0), D2.GetLength(1) }, DataType.Float);
            }
            else if(D2 != null || E2 != null)
            {
                UnityEngine.Debug.LogWarning("Both D2 and E2 needs to be provide to use them");
            }
            //inputs for blending
            Variable inputBlend = null;
            if (finalBlend != null && outputShape != null)
            {
                UnityEngine.Debug.Assert(finalBlend.Shape[0] == outputShape[0] && finalBlend.Shape[1] == outputShape[1] && 
                    finalBlend.Shape[2] == outputShape[2], "value to blend has a different dimension as the outputshape");
                inputBlend = Variable.InputVariable(new int[] { finalBlend.Shape[0], finalBlend.Shape[1], finalBlend.Shape[2] }, DataType.Float);
            }

            //create the model
            var reshapedInputData = CNTKLib.Reshape(inputData, new int[] { inputShape[0], inputShape[1] });
            var finalOutput = CNTKLib.Times(reshapedInputData, CNTKLib.Times(inputE1, CNTKLib.Times(inputD1, CNTKLib.Transpose(inputE1))));
            //finalOutput = reshapedInputData;///TEset!!
            if (inputE2 != null && inputD2 != null)
            {
                finalOutput = CNTKLib.Times(finalOutput, CNTKLib.Times(inputE2, CNTKLib.Times(inputD2, CNTKLib.Transpose(inputE2))));
            }
            if (inputMean != null)
            {
                Variable reshapedMean = CNTKLib.Reshape(inputMean, new int[] { finalAdd.Shape[0], finalAdd.Shape[1] });
                finalOutput = finalOutput + reshapedMean;
            }
            //reshape
            if(outputShape != null)
                finalOutput = CNTKLib.Reshape(finalOutput, new int[] { outputShape[0], outputShape[1], outputShape[2] } );
            //blend
            if (inputBlend != null)
            {
                blendFactor = Mathf.Clamp01(blendFactor);
                finalOutput = CNTKLib.ElementTimes(finalOutput, Constant.Scalar(DataType.Float, blendFactor, device)).Output +
                    CNTKLib.ElementTimes(inputBlend, Constant.Scalar(DataType.Float, 1 - blendFactor, device));
            }

            //Create Values for E and D
            Value valueE1 = new Value(new NDArrayView(new int[] { E1.GetLength(0), E1.GetLength(1) }, E1.Transpose().Reshape(), device));
            Value valueD1 = new Value(new NDArrayView(new int[] { D1.GetLength(0), D1.GetLength(1) }, D1.Transpose().Reshape(), device));
            //input data map
            var inputDataMap = new Dictionary<Variable, Value>() { { inputData, dataToTransform },{ inputE1, valueE1 },{ inputD1, valueD1 } };
            if(inputMean != null)
            {
                inputDataMap[inputMean] = finalAdd;
            }
            if(inputD2 != null && inputE2 != null)
            {
                Value valueE2 = new Value(new NDArrayView(new int[] { E2.GetLength(0), E2.GetLength(1) }, E2.Transpose().Reshape(), device));
                Value valueD2 = new Value(new NDArrayView(new int[] { D2.GetLength(0), D2.GetLength(1) }, D2.Transpose().Reshape(), device));
                inputDataMap[inputD2] = valueD2;
                inputDataMap[inputE2] = valueE2;
            }
            if(inputBlend != null)
            {
                inputDataMap[inputBlend] = finalBlend;
            }
            var outputDataMap = new Dictionary<Variable, Value>() { { finalOutput.Output, null }};

            finalOutput.Evaluate(inputDataMap, outputDataMap,false, device);
            return new Tuple<Value, Function>(outputDataMap[finalOutput.Output], finalOutput);
        }


        public static Tuple<Value, Function> Whiten(Value image, DeviceDescriptor device, float eps = 0.00001f, float ignoredSingularValue = 0.00001f, float beta = -0.5f)
        {

            //create the CNTK model to multiply and reshape to get the covariance matrix, also get the mean
            Tuple<Variable, Variable, Variable> outputsVars;
            Tuple<Value, Value,Value> covAndMeanAndReshapeCentered = CovarianceMatrixFromImage(image, device,out outputsVars);
            NDShape imageShape = image.Shape;
            float[,] covMatArray = covAndMeanAndReshapeCentered.Item1.GetDenseData<float>(outputsVars.Item1)[0].CopyToArray().Reshape(imageShape[2], imageShape[2]);

            Tuple<float[,], float[,]> EAndD = GetWCMatrices(covMatArray, eps, ignoredSingularValue,  beta);

            var result = MultiplyWTC(covAndMeanAndReshapeCentered.Item3, EAndD.Item1, EAndD.Item2, device, null,null,null, imageShape);
            
            return result;
        }

        /// <summary>
        /// Whitening and coloring the data. image should be image with dimension :HxWxC.
        /// This uses CNTK for most of the operation except for svd because CNTK does not have that
        /// adapted from https://github.com/eridgd/WCT-TF/blob/master/ops.py
        /// </summary>
        /// <param name="imageStyle"></param>
        /// <param name="imageContent"></param>
        /// <param name="device"></param>
        /// <param name="eps"></param>
        /// <param name="ignoredSingularValue"></param>
        /// <param name="beta"></param>
        /// <returns></returns>
        public static Tuple<Value, Function> WhitenAndColor(Value imageStyle, Value imageContent, DeviceDescriptor device, float eps = 0.00001f, float ignoredSingularValue = 0.00001f, float beta = 0.5f, float blendFactor = 0.8f)
        {
            //create the CNTK model to multiply and reshape to get the covariance matrix, also get the mean
            //style images data
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            Tuple<Variable, Variable, Variable> outputsVarStyle;
            Tuple<Value, Value, Value> covAndMeanAndReshapeCenteredStyle = CovarianceMatrixFromImage(imageStyle, device, out outputsVarStyle);
            NDShape imageShapeStyle = imageStyle.Shape;
            float[,] covMatArrayStyle = covAndMeanAndReshapeCenteredStyle.Item1.GetDenseData<float>(outputsVarStyle.Item1)[0].CopyToArray().Reshape(imageShapeStyle[2], imageShapeStyle[2]);
            //content images data
            Tuple<Variable, Variable, Variable> outputsVarContent;
            Tuple<Value, Value, Value> covAndMeanAndReshapeCenteredContent = CovarianceMatrixFromImage(imageContent, device, out outputsVarContent);
            NDShape imageShapeContent = imageContent.Shape;
            float[,] covMatArrayContent = covAndMeanAndReshapeCenteredContent.Item1.GetDenseData<float>(outputsVarContent.Item1)[0].CopyToArray().Reshape(imageShapeContent[2], imageShapeContent[2]);
            stopwatch.Stop();
            //UnityEngine.Debug.Log("----Otained Cov matrices. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);

            stopwatch.Restart();

            Tuple<float[,], float[,]> EAndDStyle = GetWCMatrices(covMatArrayStyle, eps, ignoredSingularValue, beta);
            Tuple<float[,], float[,]> EAndDContent = GetWCMatrices(covMatArrayContent, eps, ignoredSingularValue, -beta);
            stopwatch.Stop();
            //UnityEngine.Debug.Log("----Otained Whitening/Coloring matrices. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);

            stopwatch.Restart();
            var result = MultiplyWTC(covAndMeanAndReshapeCenteredContent.Item3, EAndDContent.Item1, EAndDContent.Item2, 
                device, EAndDStyle.Item1, EAndDStyle.Item2, covAndMeanAndReshapeCenteredStyle.Item2, imageShapeContent, imageContent, blendFactor);
            stopwatch.Stop();
            //UnityEngine.Debug.Log("----WC matrices multiplied. Time cost: " + (float)stopwatch.ElapsedMilliseconds / 1000);

            return result;
        }
    }

}
