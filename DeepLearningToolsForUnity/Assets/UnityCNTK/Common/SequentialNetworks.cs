using System.Collections;
using System.Collections.Generic;
using CNTK;

namespace UnityCNTK
{
    public class SequentialNetworkDense
    {

        //the reference to the CNTK function of the resnet
        public LayerDefinitions.InputLayerDef InputLayer { get; private set; } = null;
        public LayerDefinitions.LayerDef[] HiddenLayers { get; private set; } = null;
        public LayerDefinitions.OutputLayerDef OutputLayer { get; private set; } = null;

        public DeviceDescriptor Device { get; set; }
        public Function CNTKFunction { get { return OutputLayer.CNTKFunction; } }

        public List<Parameter> Parameters { get {
                return CNTKFunction.FindParametersByName(ParameterNames.ToArray());
            } }

        public List<string> ParameterNames;

        /// <param name="input">input specs for the resnet</param>
        /// <param name="hiddenLayers">hidden layers specs for the resnet.</param>
        /// <param name="output">output specs for the resnet</param>
        /// Note that those input well not be copied.
        public SequentialNetworkDense(LayerDefinitions.InputLayerDef input, LayerDefinitions.LayerDef[] hiddenLayers, LayerDefinitions.OutputLayerDef output, DeviceDescriptor device, string name="")
        {
            this.InputLayer = input;
            this.HiddenLayers = hiddenLayers;
            OutputLayer = output;
            Device = device;
            ParameterNames = new List<string>();
            //start to build the network
            var temp = input.Build("NetworkInput");
            for (int i = 0; i < hiddenLayers.Length; ++i)
            {
                temp = hiddenLayers[i].Build(temp, device, name+".Hidden." + i);
                ParameterNames.AddRange(hiddenLayers[i].ParameterNames);
            }
            
            temp = OutputLayer.Build(temp, device, name + ".NetworkOutput");
            ParameterNames.AddRange(OutputLayer.ParameterNames);

        }

        public IList<float> EvaluateOne(float[] input)
        {

            var inputVar = InputLayer.InputVariable;
            var outputVar = OutputLayer.GetOutputVariable();


            Value inputdata = Value.CreateBatch(inputVar.Shape, input, Device, true);


            var inputDataMap = new Dictionary<Variable, Value>() { { inputVar, inputdata } };
            var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null } };


            outputVar.ToFunction().Evaluate(inputDataMap, outputDataMap, Device);
            var result = outputDataMap[outputVar].GetDenseData<float>(outputVar);
            return result[0];
        }

        public IList<float> EvaluateOne(float[] input, Variable outputVar)
        {

            var inputVar = InputLayer.InputVariable;


            Value inputdata = Value.CreateBatch(inputVar.Shape, input, Device, true);


            var inputDataMap = new Dictionary<Variable, Value>() { { inputVar, inputdata } };
            var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null } };


            outputVar.ToFunction().Evaluate(inputDataMap, outputDataMap, Device);

            return outputDataMap[outputVar].GetDenseData<float>(outputVar)[0];
        }

        public byte[] Save()
        {
            return CNTKFunction.Save();
        }

        public void Restore(byte[] data)
        {
            Function f = Function.Load(data, Device);
            CNTKFunction.RestoreParametersByName(f);
        }

    }





}