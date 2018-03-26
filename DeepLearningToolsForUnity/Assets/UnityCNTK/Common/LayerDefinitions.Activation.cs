using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

namespace UnityCNTK.LayerDefinitions
{

    public class ReluDef:LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        /// <summary>
        /// Construct a relu layer data for build
        /// </summary>
        public ReluDef()
        {
            ParameterNames = new List<string>();
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.ReLU(input,name);
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }



    public class SELUDef : LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        public float Alpha { get; private set; }
        public float Gamma { get; private set; }
        /// <summary>
        /// Construct a relu layer data for build
        /// </summary>
        public SELUDef(float gamma = 1.0507f,float alpha = 1.6732f)
        {
            ParameterNames = new List<string>();
            Alpha = alpha;
            Gamma = gamma;
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.SELU(input, Gamma,Alpha, name);
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }
    public class ELUDef : LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        /// <summary>
        /// Construct a relu layer data for build
        /// </summary>
        public ELUDef()
        {
            ParameterNames = new List<string>();
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.ELU(input, name);
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }

    public class SwishDef : LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        public float Beta { get; private set; }
        /// <summary>
        /// Construct a relu layer data for build
        /// </summary>
        public SwishDef(float beta = 1)
        {
            Beta = beta;
            ParameterNames = new List<string>();
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.ElementTimes(input, CNTKLib.ElementTimes(Constant.Scalar(DataType.Float,Beta),CNTKLib.Sigmoid(input, name)));
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }


    public class LeakyReLUDef : LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        public float Alpha { get; private set; }
        /// <summary>
        /// Construct a relu layer data for build
        /// </summary>
        public LeakyReLUDef(float alpha)
        {
            ParameterNames = new List<string>();
            Alpha = alpha;
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.LeakyReLU(input, Alpha, name);
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }

    public class TanhDef : LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        /// <summary>
        /// Construct a tanh layer data for build
        /// </summary>
        public TanhDef()
        {
            ParameterNames = new List<string>();
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.Tanh(input, name);
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }

    public class SigmoidDef : LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        /// <summary>
        /// Construct a sigmoid layer data for build
        /// </summary>
        public SigmoidDef()
        {
            ParameterNames = new List<string>();
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.Sigmoid(input, name);
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }

    public class SoftmaxDef : LayerDef
    {
        public override List<string> ParameterNames { get; protected set; }
        /// <summary>
        /// Construct a relu layer data for build
        /// </summary>
        public SoftmaxDef()
        {
            ParameterNames = new List<string>();
        }
        protected override Function BuildNetwork(Variable input, DeviceDescriptor device, string name)
        {
            var c1 = CNTKLib.Softmax(input, name);
            return c1;
        }
        public override Function BuildNew(Variable input, DeviceDescriptor device, string name)
        {
            return BuildNetwork(input, device, name);
        }
    }

}